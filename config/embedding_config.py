# config/embedding_config.py
"""
嵌入配置（轻量变体）

与原始版本相比新增云端 embeddings 三项配置：
  EMBEDDING_API_KEY  / OPENAI_API_KEY   —— OpenAI 兼容的 embeddings 密钥
  EMBEDDING_API_BASE / OPENAI_API_BASE  —— 接口基址（默认 SiliconFlow）
  EMBEDDING_MODEL                       —— 模型名（默认 BAAI/bge-m3，1024 维）
  EMBEDDING_DIM                         —— 稠密向量维度，必须与模型输出一致

bge_* 字段保留仅为向后兼容（历史脚本仍在读取），轻量变体不再加载本地模型。
"""
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


def _env(*names, default=""):
    """按优先级取第一个非空环境变量，便于新旧变量名共存。"""
    for name in names:
        val = os.getenv(name)
        if val:
            return val.strip()
    return default


@dataclass
class EmbeddingConfig:
    bge_m3_path: str
    bge_m3: str
    bge_device: str
    bge_fp16: bool
    # ===== 轻量变体新增：云端稠密嵌入 =====
    embedding_api_key: str
    embedding_api_base: str
    embedding_model: str
    embedding_dim: int


embedding_config = EmbeddingConfig(
    bge_m3_path=os.getenv("BGE_M3_PATH"),
    bge_m3=os.getenv("BGE_M3"),
    bge_device=os.getenv("BGE_DEVICE"),
    # 特殊处理：将.env中的1/0转为布尔值，兼容常见的数字/字符串格式
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true", 1),
    embedding_api_key=_env("EMBEDDING_API_KEY", "OPENAI_API_KEY"),
    embedding_api_base=_env("EMBEDDING_API_BASE", "OPENAI_API_BASE",
                            default="https://api.siliconflow.cn/v1"),
    embedding_model=_env("EMBEDDING_MODEL", default="BAAI/bge-m3"),
    embedding_dim=int(_env("EMBEDDING_DIM", default="1024")),
)

if __name__ == '__main__':
    print("api_base =", embedding_config.embedding_api_base)
    print("model    =", embedding_config.embedding_model)
    print("dim      =", embedding_config.embedding_dim)
    print("api_key  =", (embedding_config.embedding_api_key or "")[:8] + "***")
