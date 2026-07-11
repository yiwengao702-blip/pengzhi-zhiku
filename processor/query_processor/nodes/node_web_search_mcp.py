# processor/query_processor/nodes/node_web_search_mcp.py
import asyncio
import concurrent.futures
import json

from agents.mcp import MCPServerStreamableHttp

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.task_utils import add_done_task


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        query = state.get("rewritten_query", "")
        docs = []
        # 如果没有查询内容，直接返回
        if query:
            # 兼容事件循环场景：如果当前已有运行中的事件循环（如 FastAPI async 端点），
            # 则通过独立线程执行 asyncio.run() 以避免 "cannot be called from a running event loop" 错误
            try:
                result = self._run_async_safe(self._mcp_call(query))
            except Exception as e:
                # [健壮修复] Web 搜索只是问答的增强项，外部 MCP 鉴权/网络失败（如 401）时
                # 不应拖垮整条问答链路；捕获后降级为空结果，继续走 KB 检索 + LLM 回答。
                logger.warning(f"Web 搜索 MCP 调用失败，已降级跳过: {e}")
                result = None
            if result: # 判断是否从web搜索里面得到结果
                try:
                    # 安全解析 MCP 返回的 JSON（防御非法 JSON）
                    pages = json.loads(result.content[0].text).get("pages") or []
                except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as e:
                    logger.error(f"MCP 搜索结果解析失败: {e}")
                    pages = []
                # 统一输出结构化结果，供后续 rerank/引用使用
                # 每条：{title, url, snippet}
                # pages 5条消息
                for item in pages:
                    snippet = (item.get("snippet") or "").strip()
                    url = (item.get("url") or "").strip()
                    title = (item.get("title") or "").strip()
                    if not snippet:
                        continue
                    docs.append({"title": title, "url": url, "snippet": snippet})

                logger.info(f"MCP 搜索结果:{docs}")
        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
        # [BUG修复] 统一返回格式：即使无结果也返回 {"web_search_docs": []}
        # 避免 state 中 web_search_docs 键缺失，防止下游代码方括号访问时抛出 KeyError
        return {"web_search_docs": docs}

    @staticmethod
    def _run_async_safe(coro):
        """
        安全执行协程：自动检测事件循环状态，兼容同步和异步两种运行上下文。
        - 无运行中的事件循环 → 直接 asyncio.run()
        - 有运行中的事件循环（如 FastAPI async 端点）→ 在独立线程中 asyncio.run()
        注意：LangGraph 所有节点都是同步调用，而本节点需要通过 MCP 调用外部异步服务，
              因此需要在此处完成同步→异步的桥接。
        """
        try:
            # 检测当前线程是否有运行中的事件循环
            asyncio.get_running_loop()
            # 已有事件循环：在独立线程中运行，避免嵌套冲突
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            # 无事件循环：直接使用 asyncio.run()
            return asyncio.run(coro)

    async def _mcp_call(self, query):
        # MCPServerStreamableHttp 发起一个请求
        search_mcp = MCPServerStreamableHttp(
            name="search_mcp",  # 要使用的插件名称(可自定义) 比如我的搜索服务
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                "timeout": 10,
            },
            cache_tools_list=True,  # 第1次请求,后续缓存不重复请求(服务器上有哪些工具)
            max_retry_attempts=3,  # 网络异常时重试次数
        )

        try:
            await search_mcp.connect() # 链接
            result = await search_mcp.call_tool(
                tool_name="bailian_web_search",  # 使用的插件名称(不能错)
                arguments={"query": query, "count": 5},
            )
            return result
        finally:
            await search_mcp.cleanup()


if __name__ == "__main__":
    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？"
    }

    # 执行节点的业务调用
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(serialize_json(result, indent=4))
