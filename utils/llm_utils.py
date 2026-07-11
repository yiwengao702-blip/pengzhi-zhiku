# utils/llm_utils.py

from langchain_openai import ChatOpenAI

from config.lm_config import lm_config

_llm_client_cache = {}

# 自由切换模型

def get_llm_client(model: str | None = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取 LangChain ChatOpenAI 客户端实例
    - model: 允许不同节点使用不同模型
    - json_mode: True 时要求输出 JSON
    """
    m = model or lm_config.llm_model
    key = (m, json_mode)  # key = ('ds',false)    元祖
    if key in _llm_client_cache:# 存在 我就不做任何操作
        return _llm_client_cache[key]

    # 是否深度思考
    extra_body = {"enable_thinking": False}

    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
    # 创建LLM
    client = ChatOpenAI(
        model=m,
        temperature=lm_config.llm_temperature,
        api_key=lm_config.api_key,
        base_url=lm_config.base_url,
        extra_body=extra_body,# 是否深度
        model_kwargs=model_kwargs,# json
    )
    _llm_client_cache[key] = client
    return client