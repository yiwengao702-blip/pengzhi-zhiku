# web/api/query_service.py
"""
查询流程的接口定义

[轻量变体] 已移除 torch / torchvision 预导入：嵌入改走云端 API（见 utils/embedding_utils.py），
本地不再加载任何神经模型，原先为规避 torchvision meta 算子重复注册而做的兼容导入不再需要。
"""

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
from utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, get_task_result, set_task_result, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED
from tool.logger import logger
# 1. 创建应用
app = FastAPI(
    title="掌柜智库-查询API",
    description="此文档是掌柜智库查询流程的API接口说明"
)

# [安全加固 Phase0] 跨域收紧：同 import_service，禁止 `*` + credentials 的非法组合。
def _kb_cors_origins() -> list:
    raw = os.getenv("KB_CORS_ORIGINS", "")
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://127.0.0.1:8100", "http://127.0.0.1:8101",
        "http://localhost:8100", "http://localhost:8101",
    ]


# 2. 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=_kb_cors_origins(),  # 显式白名单
    allow_credentials=False,  # 同源无需凭证，关闭以根除凭证泄露面
    allow_methods=["*"],  # 允许的请求方法
    allow_headers=["*"],  # 允许的请求头
)


# [安全加固 Phase0] 可选 API 密钥闸门（防御纵深）：设置 KB_API_KEY 后，
# /query、/stream、/history 等问答/历史接口必须携带 `X-API-Key` 或 `Authorization: Bearer <key>`。
# 未设置时保持开发态开放（服务已默认绑定 127.0.0.1），启动打印告警提醒生产务必配置。
_KB_API_KEY = os.getenv("KB_API_KEY", "").strip()
if _KB_API_KEY:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class _ApiKeyGuard(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            p = request.url.path
            if p == "/query" or p.startswith("/stream") or p.startswith("/history"):
                auth = request.headers.get("Authorization", "")
                token = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("X-API-Key", "").strip()
                if token != _KB_API_KEY:
                    return JSONResponse(
                        status_code=401,
                        content={"ok": False, "detail": "缺少或错误的 API 密钥"},
                    )
            return await call_next(request)

    app.add_middleware(_ApiKeyGuard)
else:
    import logging as _logging
    _logging.getLogger("knowledge_base").warning(
        "KB_API_KEY 未设置：问答/历史接口当前无任何鉴权（仅依赖 127.0.0.1 绑定）。\n"
        "生产环境务必启用 API Key 闸门，具体设置步骤：\n"
        "  1) 生成强随机密钥 —— PowerShell: $k=-join((1..64)|%{'0123456789abcdef'[(Get-Random -Max 16)]});$k\n"
        "                         Bash/Linux: python -c 'import secrets;print(secrets.token_hex(32))'\n"
        "  2) 写入项目 .env：KB_API_KEY=<上一步生成的密钥>\n"
        "  3) 调用方请求头携带：X-API-Key: <密钥>  或  Authorization: Bearer <密钥>\n"
        "未设置前，任何能访问 127.0.0.1:8101 的进程均可免鉴权调用问答/历史接口。"
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
        # [检索指标] 非流式：meta 由 run_query_graph 内组装并写入任务结果，随响应一并返回
        meta = get_task_result(session_id, "meta", None)
        return {
            "message":"处理完成！",
            "session_id":session_id,
            "answer":answer,
            "done_list":[],
            "meta": meta
        }

# 定义查询接口
def _build_retrieval_meta(final_state: dict, timings: dict, total_ms: float) -> dict:
    """组装检索指标（检索速度 / 召回 / 相关度 / 命中），随答案推送给前端展示。

    口径说明：
    - 检索耗时 search_ms：三路召回（普通向量 / HyDE / 联网搜索）在 LangGraph 中是并行超步，
      同一超步内各节点共享耗时，因此取三者的 max（真实墙钟）而非 sum，再加上游 HyDE
      预处理、RRF 融合与重排在 meta 中单列，前端按需展示；
    - 命中 hit：只要最终答案不是以「抱歉」开头的兜底/拒答话术（未找到相关产品 /
      知识库未收录 / 生成出错）即视为命中。实时单次问答无法计算真实"正确率"，
      「召回数 + Top1 相关分 + 命中标志」是最接近的可观测代理指标；
    - top_score：重排后 Top1 的相关性分数，即该次检索的"最高相关度"。
    """
    def t(node: str) -> float:
        return float(timings.get(node, 0.0))

    search_wall_ms = max(
        t("node_search_embedding"),
        t("node_search_embedding_hyde"),
        t("node_web_search_mcp"),
    )
    # 是否真正进入过检索链路（主体确认阶段直接反问/拒答时，四类检索节点均未执行）
    retrieved = (t("node_search_embedding") + t("node_search_embedding_hyde")
                 + t("node_rrf") + t("node_rerank")) > 0
    docs = final_state.get("reranked_docs") or []
    scores = []
    for d in docs:
        try:
            if d.get("score") is not None:
                scores.append(float(d.get("score")))
        except (TypeError, ValueError):
            continue
    answer = (final_state.get("answer") or "").strip()
    hit = bool(answer) and not answer.startswith("抱歉")
    return {
        "total_ms": round(total_ms),
        "confirm_ms": round(t("node_item_name_confirm")),
        "prep_ms": round(t("node_hyde_embed_prep")),
        "search_ms": round(search_wall_ms),
        "rrf_ms": round(t("node_rrf")),
        "rerank_ms": round(t("node_rerank")),
        "llm_ms": round(t("node_answer_output")),
        "rrf_count": len(final_state.get("rrf_chunks") or []),
        "recall_count": len(docs),
        "top_score": round(max(scores), 4) if scores else None,
        "retrieved": retrieved,
        "hit": hit,
    }


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
        # [检索指标] 统一走带计时的执行通道：同时拿到最终状态与逐节点耗时，
        # 用于组装「检索速度/召回/相关度/命中」指标。节点内部的流式推送行为
        # 仍由 state["is_stream"] 控制，与原 invoke/stream 语义完全一致。
        final_state, timings, total_ms = workflow.run_with_timing(init_state, stream=is_stream)
        # [检索指标] 组装 meta 并登记到任务结果；流式时必须在置 completed 之前推送，
        # 保证 meta 事件先于任何关闭信号到达前端（前端 final 后延迟收口等待 meta）
        meta = _build_retrieval_meta(final_state, timings, total_ms)
        set_task_result(session_id, "meta", meta)
        if is_stream:
            push_to_session(session_id, SSEEvent.META, meta)
        # [检索指标] 回写进历史：使指标随会话持久化，刷新后前端可据此重绘指标行与遥测侧栏
        try:
            from utils.mongo_history_utils import update_latest_assistant_meta
            update_latest_assistant_meta(session_id, meta)
        except Exception as _e:
            print(f"回写检索指标到历史失败(不影响主链路): {_e}")
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
                "image_urls": r.get("image_urls", []),
                "ts": r.get("ts"),
                # 检索遥测指标：随历史返回，刷新后前端据此重绘指标行与遥测侧栏
                "meta": r.get("meta")
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
# 静态资源缓存策略：显式声明「每次回源校验」。
# 背景：Starlette 的 StaticFiles 只发 ETag/Last-Modified、不发 Cache-Control，
# 浏览器据此按启发式规则长时间复用旧 JS，出现过「已修复 app-config.js 但用户仍走
# 127.0.0.1」的假故障。加 no-cache 后浏览器每次带 If-None-Match 回源，未变 304、变了立即生效。
@app.middleware("http")
async def _static_no_cache(request, call_next):
    response = await call_next(request)
    _p = request.url.path
    if _p.endswith((".js", ".css", ".html")) or _p == "/":
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


app.mount("/assets", StaticFiles(directory=str(Path(__file__).absolute().parent.parent / "page" / "assets")), name="assets")


if __name__ == "__main__":
    # 端口与 main.py 保持一致：优先读取 .env 中的 KB_PORT_QUERY（默认 8001，
    # 当前因 8001 被「清衡舆情分析平台」占用而改为 8101），避免直接运行与一键启动端口不一致
    _query_port = int(os.getenv("KB_PORT_QUERY", 8001))
    uvicorn.run(app, host=os.getenv("KB_HOST", "127.0.0.1"), port=_query_port)