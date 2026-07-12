# web/api/query_service.py
"""
查询流程的接口定义

[兼容修复] 必须在任何间接依赖 torchvision 的库（FlagEmbedding / transformers 等）之前，
先干净地导入 torch / torchvision，使其内部 meta 算子（如 roi_align）一次性注册并缓存，
避免后续被重复注册而抛出 "already a kernel registered ... roi_align" 的 RuntimeError
（torch 2.11 + torchvision 0.26 在 transformers 间接导入时偶发）。
"""
import torch  # noqa: F401
import torchvision  # noqa: F401

import os
import sys
import uuid
from pathlib import Path

# [修复] 让「直接运行本文件」也能正确导入项目内的 config/processor/utils 等包。
# 直接执行 python web/api/query_service.py 时，Python 只会把脚本所在目录 web/api 加入 sys.path，
# 而 config、processor、utils、tool 都在项目根目录下，会抛 ModuleNotFoundError。
# 这里在导入项目内模块之前，把项目根目录显式加入 sys.path，使直接运行与 python main.py 行为一致。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # web/api/query_service.py -> 项目根目录
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from processor.query_processor.main_graph_v2 import KBQueryWorkflowV2
from utils.mongo_history_utils import clear_history, get_recent_messages
from utils.answer_cache import get_cached_answer, set_cached_answer  # [P1-方案7] 答案短时缓存
from utils.sse_utils import create_sse_queue, SSEEvent, push_to_session, sse_generator
from utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, get_task_result, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED
from tool.logger import logger
# 1. 创建应用
app = FastAPI(
    title="掌柜智库-查询API",
    description="此文档是掌柜智库查询流程的API接口说明"
)

# 2. 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的源
    allow_credentials=True,  # 允许携带cookie
    allow_methods=["*"],  # 允许的请求方法
    allow_headers=["*"],  # 允许的请求头
)

# 3. 静态页面路由
@app.get("/chat.html")  # 对外访问地址
async def chat():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    html_path = current_dir_parent_path / "page" / "chat.html"
    # 如果不存在，抛出404异常
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{html_path}")
    return FileResponse(html_path)


# 3.1 其余静态页面路由：与查询服务同源托管，保证侧边栏跨页导航可达
@app.get("/import.html")  # 对外访问地址
async def get_import_page():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    html_path = current_dir_parent_path / "page" / "import.html"
    # 如果不存在，抛出404异常
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{html_path}")
    return FileResponse(html_path)


@app.get("/overview.html")  # 对外访问地址
async def get_overview_page():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    html_path = current_dir_parent_path / "page" / "overview.html"
    # 如果不存在，抛出404异常
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{html_path}")
    return FileResponse(html_path)


# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="查询内容")  # ...必须填写
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")

@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    1 解析参数
    2 更新任务状态
    3 调用处理流程图
    4 返回结果
    :param background_tasks:
    :param request:
    :return:
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    # 处理是不是流式返回结果
    is_stream = request.is_stream

    if is_stream:
        # 创建一个字典 存储对一个session_id : queue 结果队列
        create_sse_queue(session_id)
    # 更新任务状态
    # 当前会话id作为key! 整体装填处于运行中！
    update_task_status(session_id, TASK_STATUS_PROCESSING,is_stream)

    print("开始处理流程... 是否流式:", is_stream, f"其他参数:{user_query}, session_id:{session_id}")

    if is_stream:
        # 如果是流式，则返回一个流式响应，过程不断地推送
        # 运行执行图对象方法
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        # 返回结果
        print("开始处理结果....")
        return {
            "message":"结果正在处理中...",
            "session_id":session_id
        }
    else:
        # 同步运行
        # [P1-方案7] 答案短时缓存：相同 (session_id, query) 在 TTL(60s) 内重复请求直接返回，
        # 避免重复跑昂贵的「检索 + 重排 + 生成」链路（用户重复点击/网络重发等场景）。
        # 说明：缓存命中即代表首次调用已成功写入历史，无需重复写历史，直接返回即可。
        cached_answer = get_cached_answer(session_id, user_query)
        if cached_answer is not None:
            logger.info(f"答案缓存命中，直接返回。session_id={session_id}")
            return {
                "message": "处理完成！",
                "session_id": session_id,
                "answer": cached_answer,
                "done_list": [],
                "cached": True
            }
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id,"answer","")
        # 写入缓存（空答案不缓存，由 set_cached_answer 内部拦截）
        set_cached_answer(session_id, user_query, answer)
        return {
            "message":"处理完成！",
            "session_id":session_id,
            "answer":answer,
            "done_list":[]
        }

# 定义查询接口
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    print(f"开始流程图处理...{session_id} {user_query} {is_stream}")

    # [BUG修复] 初始化 state 时补全 QueryGraphState TypedDict 声明的全部必填字段为默认值
    # 避免下游节点使用方括号访问未设置字段时抛出 KeyError
    init_state = {
        "session_id": session_id,
        "message_id": "",
        "original_query": user_query,
        "embedding_chunks": [],
        "hyde_embedding_chunks": [],
        "hyde_doc": "",
        "web_search_docs": [],
        # [P1-方案6] 上游批处理节点写入的预计算向量，初始为 None
        "query_dense": None,
        "query_sparse": None,
        "hyde_dense": None,
        "hyde_sparse": None,
        "rrf_chunks": [],
        "reranked_docs": [],
        "prompt": "",
        "answer": "",
        "item_names": [],
        "rewritten_query": "",
        "history": [],
        "is_stream": is_stream
    }

    try:
        workflow = KBQueryWorkflowV2()
        for chunk in workflow.run(init_state, stream=is_stream):
            logger.debug(chunk)
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        print(f"流程执行异常: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})



@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):

    print("调用流式/stream...")
    """
    sse 实时返回结果
    """
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """
    清空指定会话的历史记录
    """
    count = clear_history(session_id)
    return {"message": "历史会话已清空", "deleted_count": count}

@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """
    查询当前会话历史记录
    """
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


# 证明服务器启动即可
@app.get("/health")
async def health():
    """
    检查服务是否正常
    """
    return {"ok": True}

# 静态资源挂载：前端独立的 CSS/JS 资产以相对路径 assets/... 引用，
# 页面与资源同处 web/page 下（Vite 风格的 dist 布局），既支持 FastAPI 托管，
# 也支持 file:// 直接打开或任意 docroot 下的静态服务器，彻底避免绝对路径 404。
app.mount("/assets", StaticFiles(directory=str(Path(__file__).absolute().parent.parent / "page" / "assets")), name="assets")


if __name__ == "__main__":
    # 端口与 main.py 保持一致：优先读取 .env 中的 KB_PORT_QUERY（默认 8001，
    # 当前因 8001 被「清衡舆情分析平台」占用而改为 8101），避免直接运行与一键启动端口不一致
    _query_port = int(os.getenv("KB_PORT_QUERY", 8001))
    uvicorn.run(app, host="127.0.0.1", port=_query_port)