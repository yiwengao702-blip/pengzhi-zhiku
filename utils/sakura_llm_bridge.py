# knowledge_base/utils/sakura_llm_bridge.py
#
# [融合] knowledge_base 的「文本」LLM 调用收敛到 sakura_base.llm.gateway。
# 以 LangChain BaseChatModel 适配网关，保持 get_llm_client() 的 invoke/stream 调用点不变
# （不重写任何节点）。仅接管「文本」路径；视觉/多模态调用（vl_model，图片识别）仍由各节点
# 既有 ChatOpenAI 处理——网关是纯文本传输，无法承载图片，故有意保留该部分不收敛。
#
# 收敛收益：消除 knowledge_base 自写的 ChatOpenAI 客户端，统一到共享底座的 HTTP 传输与
# 供应商/降级策略；json_mode 与 extra_body(enable_thinking) 透传，保持既有行为。

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

# [融合] 复用共享底座的 LLM 网关，消除自写 OpenAI 客户端
# [部署兼容] 云端部署环境里没有 sakura_platform 仓库，也就装不上 editable 的 sakura_base。
#   为此把 sakura_base.llm（仅 gateway/types 两个纯标准库文件）vendor 进本项目 vendor/ 目录，
#   本机开发仍优先使用已安装的完整底座，缺包时自动回落到 vendor 副本，实现「部署零外部依赖」。
try:
    from sakura_base.llm.gateway import LLMGateway
    from sakura_base.llm.types import ChatMessage
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path

    _vendor_dir = _Path(__file__).resolve().parents[1] / "vendor"
    if _vendor_dir.is_dir() and str(_vendor_dir) not in _sys.path:
        _sys.path.append(str(_vendor_dir))
    try:
        from sakura_base.llm.gateway import LLMGateway
        from sakura_base.llm.types import ChatMessage
    except ImportError as _imp_err:
        raise RuntimeError(
            "knowledge_base 需要 sakura_base 共享底座（或项目内 vendor/sakura_base 副本），请先安装：\n"
            "  pip install -e <sakura_platform>/packages/sakura_base[web]\n"
            f"原始错误：{_imp_err}"
        ) from _imp_err


class SakuraGatewayChatModel(BaseChatModel):
    """LangChain 适配：把网关 complete() 暴露成 invoke/ainvoke/stream。

    网关以标准库 urllib 同步请求，适配器在异步/流式路径用阻塞调用包裹，并保留至多 3 次
    退避重试，对齐原 ChatOpenAI 的链路韧性。json_mode / extra_body 透传给网关。
    """

    provider: str = "openai"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    json_mode: bool = False
    extra_body: Optional[Dict[str, Any]] = None

    @property
    def _llm_type(self) -> str:
        return "sakura-gateway"

    @property
    def _identifying_params(self) -> dict:
        return {
            "llm_type": "sakura-gateway",
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "json_mode": self.json_mode,
        }

    @staticmethod
    def _to_chat_messages(messages: List[BaseMessage]) -> List[ChatMessage]:
        """LangChain 消息 -> 网关 ChatMessage；角色映射 human->user、ai->assistant、system->system。"""
        out: List[ChatMessage] = []
        for m in messages:
            role = getattr(m, "type", None) or m.__class__.__name__.lower().replace("message", "")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            content = m.content if isinstance(m.content, str) else str(m.content)
            out.append(ChatMessage(role=role, content=content))
        return out

    def _complete(self, messages: List[BaseMessage]) -> str:
        # 网关为轻量 dataclass，每次重建开销可忽略；便于在重试间不持有跨线程状态。
        gw = LLMGateway(
            provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            timeout=30.0,
        )
        last_exc: Optional[Exception] = None
        for _ in range(2):
            try:
                res = gw.complete(
                    self._to_chat_messages(messages),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    json_mode=self.json_mode,
                    extra_body=self.extra_body,
                )
                return res.text
            except Exception as exc:  # 网关已把网络错误包成 RuntimeError
                last_exc = exc
                continue
        # 3 次均失败：透传最后一次异常
        assert last_exc is not None
        raise last_exc

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = self._complete(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = await asyncio.to_thread(self._complete, messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # 网关为同步非流式传输，这里按 LangChain 约定以单个 chunk 形式吐出完整文本；
        # 节点对 chunk.content 的累加逻辑不受影响（仅失去逐 token 渐进，功能等价）。
        text = self._complete(messages)
        yield ChatGenerationChunk(message=AIMessageChunk(content=text))
