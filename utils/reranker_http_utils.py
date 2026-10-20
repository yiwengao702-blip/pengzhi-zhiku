# utils/reranker_http_utils.py
"""
重排（Rerank）工具

双通道实现：
  1) 主通道 DashScope TextReRank —— 配了 MCP_API_KEY 时走这里（保持原有行为）；
  2) 备用通道 OpenAI 兼容 /rerank —— 未配 DashScope 时改走 OPENAI_API_BASE 指向的兼容端点
     （SiliconFlow 的 BAAI/bge-reranker-v2-m3 即属此类），行为与主通道一致：
     输入 (query, documents) 输出与 documents 同序的相关性分数列表。

任一路径失败都会抛出异常，由上层 node_rerank 捕获后按 RRF 顺序降级，
因此重排不可用不会中断问答链路。
"""

import requests
from dotenv import load_dotenv

import dashscope

from config.reranker_config import reranker_config

load_dotenv()

# 备用通道请求超时（秒）
_FALLBACK_TIMEOUT = 30


def _rerank_via_dashscope(query: str, documents: list, api_key: str) -> list:
    """主通道：DashScope TextReRank。"""
    dashscope.api_key = api_key
    response = dashscope.TextReRank.call(
        model=reranker_config.text_rerank_model,
        query=query,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        instruct=reranker_config.text_rerank_instruct,
    )

    status_code = response.get("status_code")
    if status_code != 200:
        message = response.get("message")
        raise RuntimeError(f"DashScope rerank 调用失败: {message}")

    results = response.output.get("results", [])
    scores = [0.0] * len(documents)
    for item in results:
        index = item.get("index")
        score = item.get("relevance_score")
        scores[int(index)] = float(score)
    return scores


def _rerank_via_openai_compatible(query: str, documents: list) -> list:
    """
    备用通道：OpenAI 兼容的 /rerank 端点（SiliconFlow / Cohere 兼容网关等）。
    请求与响应结构遵循 Cohere 风格：results[].index / results[].relevance_score。
    """
    base = (reranker_config.fallback_api_base or "").rstrip("/")
    api_key = reranker_config.fallback_api_key or ""
    if not base or not api_key:
        raise RuntimeError("未配置可用的 rerank 通道（MCP_API_KEY 与 OPENAI_API_KEY 均为空）")

    resp = requests.post(
        f"{base}/rerank",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": reranker_config.fallback_model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        },
        timeout=_FALLBACK_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"兼容端点 rerank 调用失败: HTTP {resp.status_code} {resp.text[:200]}")

    payload = resp.json()
    scores = [0.0] * len(documents)
    for item in payload.get("results", []):
        idx = int(item.get("index", 0))
        if 0 <= idx < len(scores):
            scores[idx] = float(item.get("relevance_score", 0.0))
    return scores


def rerank_documents(query: str, documents: list[str]) -> list[float]:
    """
    对候选文档做交叉编码器精排，返回与 documents 同序的相关性分数。
    :param query: 用户查询（已改写/融合后的查询）
    :param documents: 候选文档文本列表
    :return: 相关性分数列表（与 documents 等长、同序）
    """
    # [BUG修复] 入参守卫：query 为空或文档列表为空时，远端接口会直接报错
    # "query and documents are required!"，此处提前返回空评分列表，由上层走降级逻辑
    if not query or not documents:
        return []

    # 通道选择：DashScope 优先，其次 OpenAI 兼容端点
    if reranker_config.text_rerank_api_key:
        return _rerank_via_dashscope(query, documents, reranker_config.text_rerank_api_key)

    return _rerank_via_openai_compatible(query, documents)
