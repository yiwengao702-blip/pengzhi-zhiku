# config/reranker_config.py
"""
重排（Rerank）配置

主通道：DashScope TextReRank（MCP_API_KEY）
备用通道：任意 OpenAI 兼容的 /rerank 端点（如 SiliconFlow 的 BAAI/bge-reranker-v2-m3）

【为什么要备用通道】
轻量部署环境没有 DashScope 密钥，而重排是「召回后精排」的关键一环。
检测到 DashScope 未配置时，自动改用 OPENAI_API_BASE + OPENAI_API_KEY 指向的兼容端点，
使部署环境同样具备真实的交叉编码器精排能力（无需本地模型权重）。
"""
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RerankerConfig:
    text_rerank_api_key: str      # DashScope API Key
    text_rerank_model: str        # DashScope 模型名称
    text_rerank_instruct: str     # 是否使用指令
    # ===== 备用通道：OpenAI 兼容 /rerank 端点 =====
    fallback_api_key: str = ""
    fallback_api_base: str = ""
    fallback_model: str = "BAAI/bge-reranker-v2-m3"


reranker_config = RerankerConfig(
    text_rerank_api_key=os.getenv("MCP_API_KEY"),
    text_rerank_model=os.getenv("TEXT_RERANK_MODEL"),
    text_rerank_instruct=os.getenv("TEXT_RERANK_INSTRUCT"),
    fallback_api_key=os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
    fallback_api_base=os.getenv("RERANK_API_BASE") or os.getenv("OPENAI_API_BASE", ""),
    fallback_model=os.getenv("RERANK_MODEL") or "BAAI/bge-reranker-v2-m3",
)
