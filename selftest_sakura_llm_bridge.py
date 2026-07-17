# selftest_sakura_llm_bridge.py
# knowledge_base 文本 LLM 网关收敛自测：验证 SakuraGatewayChatModel 是 LangChain 兼容接口
# （invoke / stream / 角色映射 / json_mode / extra_body 透传），以及 get_llm_client() 工厂
# 按 (model, json_mode) 返回网关适配器并带缓存。
#
# 运行：从 knowledge_base 根目录用项目 venv 执行
#   .venv/Scripts/python.exe selftest_sakura_llm_bridge.py

import os
import sys
from types import SimpleNamespace

# 让 import utils / config / sakura_base 可用
KB_ROOT = os.path.dirname(os.path.abspath(__file__))
if KB_ROOT not in sys.path:
    sys.path.insert(0, KB_ROOT)
_SAKURA_PKG = r"D:/WorkBuddy/2026-10-14-11-06-38/sakura_platform/packages/sakura_base"
if _SAKURA_PKG not in sys.path:
    sys.path.insert(0, _SAKURA_PKG)

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage  # noqa: E402

import utils.sakura_llm_bridge as bridge  # noqa: E402
from utils.sakura_llm_bridge import SakuraGatewayChatModel  # noqa: E402


def _check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        sys.exit(1)


def test_offline_mock():
    m = SakuraGatewayChatModel(provider="mock", api_key="", base_url="", model="mock")
    out = m.invoke([HumanMessage(content="你好")])
    _check("mock invoke 返回 AIMessage", isinstance(out, AIMessage))
    _check("mock 回显包含输入", "你好" in out.content)


def test_stream():
    m = SakuraGatewayChatModel(provider="mock", api_key="", base_url="", model="mock")
    chunks = list(m.stream([HumanMessage(content="流式")]))
    _check("stream 产出 AIMessageChunk", all(isinstance(c, AIMessageChunk) for c in chunks))
    text = "".join(getattr(c, "content", "") or "" for c in chunks)
    _check("stream 文本包含输入", "流式" in text)


def test_role_mapping():
    msgs = bridge.SakuraGatewayChatModel._to_chat_messages([
        SystemMessage(content="sys"),
        HumanMessage(content="usr"),
        AIMessage(content="ai"),
    ])
    _check("system->system", msgs[0].role == "system")
    _check("human->user", msgs[1].role == "user")
    _check("ai->assistant", msgs[2].role == "assistant")


def test_json_mode_and_extra_body_passthrough():
    captured = {}

    class FakeGateway:
        def __init__(self, **kw):
            pass

        def complete(self, messages, *, temperature, max_tokens, json_mode=False, extra_body=None):
            captured.update(json_mode=json_mode, extra_body=extra_body, max_tokens=max_tokens, temperature=temperature)
            return SimpleNamespace(text="ok-text")

    orig = bridge.LLMGateway
    bridge.LLMGateway = FakeGateway
    try:
        m = SakuraGatewayChatModel(
            provider="openai", api_key="k", base_url="http://x/v1", model="m",
            temperature=0.1, max_tokens=8192, json_mode=True, extra_body={"enable_thinking": False},
        )
        out = m.invoke([HumanMessage(content="q")])
        _check("json_mode 透传给网关", captured.get("json_mode") is True)
        _check("extra_body 透传给网关", captured.get("extra_body") == {"enable_thinking": False})
        _check("max_tokens 透传", captured.get("max_tokens") == 8192)
        _check("温度透传", captured.get("temperature") == 0.1)
        _check("返回文本", out.content == "ok-text")
    finally:
        bridge.LLMGateway = orig


def test_get_llm_client_factory():
    import utils.llm_utils as lu  # noqa: E402
    # 用假配置覆盖（避免依赖 .env / 真实端点）
    fake_cfg = SimpleNamespace(base_url="http://x/v1", api_key="k", llm_model="default-model", llm_temperature=0.2)
    saved = lu.lm_config
    lu.lm_config = fake_cfg
    try:
        lu._llm_client_cache.clear()
        # 默认模型 + json_mode=False
        c1 = lu.get_llm_client()
        _check("默认模型回退", c1.model == "default-model")
        _check("默认 json_mode=False", c1.json_mode is False)
        _check("extra_body 保留 enable_thinking", c1.extra_body == {"enable_thinking": False})
        # 指定模型 + json_mode=True
        c2 = lu.get_llm_client(model="m2", json_mode=True)
        _check("指定模型生效", c2.model == "m2")
        _check("json_mode=True 生效", c2.json_mode is True)
        # 缓存：相同 key 返回同一实例
        c3 = lu.get_llm_client(model="m2", json_mode=True)
        _check("相同 key 命中缓存", c3 is c2)
        # 不同 key 不混用
        c4 = lu.get_llm_client(model="m2", json_mode=False)
        _check("不同 key 不命中缓存", c4 is not c2)
    finally:
        lu.lm_config = saved
        lu._llm_client_cache.clear()


def main():
    print("== knowledge_base 文本 LLM 网关适配器自测 ==")
    test_offline_mock()
    test_stream()
    test_role_mapping()
    test_json_mode_and_extra_body_passthrough()
    test_get_llm_client_factory()
    print("\nALL PASS ✅  knowledge_base 文本 LLM 已收敛到 sakura_base.llm.gateway（节点调用点不变；视觉调用保留 ChatOpenAI）。")


if __name__ == "__main__":
    main()
