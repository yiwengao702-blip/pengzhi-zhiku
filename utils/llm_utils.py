# utils/llm_utils.py

from langchain_openai import ChatOpenAI  # 保留：视觉/多模态节点仍直接使用

# [融合] 文本 LLM 收敛到共享底座网关（见 sakura_llm_bridge）
from config.lm_config import lm_config
from utils.sakura_llm_bridge import SakuraGatewayChatModel

_llm_client_cache = {}

# 自由切换模型

def get_llm_client(model: str | None = None, json_mode: bool = False) -> SakuraGatewayChatModel:
    """
    获取 LLM 客户端实例（文本路径，已收敛到 sakura_base.llm.gateway）。
    - model: 允许不同节点使用不同模型
    - json_mode: True 时要求输出 JSON（透传 response_format）
    视觉/多模态调用（vl_model，图片识别）由各节点直接 new ChatOpenAI，不经此函数。
    """
    m = model or lm_config.llm_model
    key = (m, json_mode)  # key = ('ds',false)    元祖
    if key in _llm_client_cache:# 存在 我就不做任何操作
        return _llm_client_cache[key]

    # 是否深度思考（DeepSeek 等兼容端点透传）
    extra_body = {"enable_thinking": False}

    # [融合] 文本路径统一走网关适配器；json_mode 透传 response_format，extra_body 透传 enable_thinking
    client = SakuraGatewayChatModel(
        provider="openai",
        api_key=lm_config.api_key,
        base_url=lm_config.base_url,
        model=m,
        temperature=lm_config.llm_temperature,
        max_tokens=2048,
        json_mode=json_mode,
        extra_body=extra_body,
    )
    _llm_client_cache[key] = client
    return client