# utils/answer_cache.py
"""
[P1/P2-方案7] 答案短时缓存：相同 (会话, 问题) 在 TTL 内重复提问时，直接返回缓存答案，
免去整条检索 + 生成链路（LLM / Rerank / Embedding）开销。仅用于非流式、完全相同的提问。
"""
import time
from threading import Lock

_ANSWER_CACHE: dict = {}
_LOCK = Lock()
TTL_SECONDS: int = 60          # 缓存有效期（秒）
MAX_ENTRIES: int = 500         # 最大缓存条数


def _make_key(session_id: str, query: str) -> tuple:
    return (session_id, (query or "").strip().lower())


def get_cached_answer(session_id: str, query: str):
    """命中且未过期返回答案字符串，否则返回 None。"""
    key = _make_key(session_id, query)
    with _LOCK:
        item = _ANSWER_CACHE.get(key)
        if item and (time.time() - item[1]) < TTL_SECONDS:
            return item[0]
    return None


def set_cached_answer(session_id: str, query: str, answer: str) -> None:
    """写入答案缓存（附带时间戳）；空答案不缓存。"""
    if not answer:
        return
    key = _make_key(session_id, query)
    with _LOCK:
        _ANSWER_CACHE[key] = (answer, time.time())
        # 超限或过期时清理，控制内存占用
        if len(_ANSWER_CACHE) > MAX_ENTRIES:
            now = time.time()
            expired = [k for k, v in _ANSWER_CACHE.items() if now - v[1] >= TTL_SECONDS]
            for k in expired:
                del _ANSWER_CACHE[k]
            if len(_ANSWER_CACHE) > MAX_ENTRIES:
                for k in list(_ANSWER_CACHE.keys())[:MAX_ENTRIES // 2]:
                    _ANSWER_CACHE.pop(k, None)
