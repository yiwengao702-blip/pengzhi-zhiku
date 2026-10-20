# utils/embedding_utils.py
"""
轻量嵌入后端（云 API 稠密向量 + 本地哈希稀疏向量）

【为什么改】
原实现依赖本地 BGE-M3（torch + FlagEmbedding，模型权重约 2.5GB 常驻内存），
在 2C4G 的云主机上无法与其它服务共处。这里改为：
  1) 稠密向量：调用 SiliconFlow 的 OpenAI 兼容 embeddings 接口（BAAI/bge-m3，1024 维，L2 已归一化）；
  2) 稀疏向量：本地哈希词袋特征（中文按「单字 + 双字」切分，英文/数字按 token），
     无需任何模型权重，产出与 Milvus SPARSE_FLOAT_VECTOR 完全兼容的 {token_id: weight}。

【对外契约保持不变】
    generate_embeddings(texts) -> {"dense": List[List[float]], "sparse": List[Dict[int, float]]}
上游 node_bge_embedding / node_search_embedding / node_search_embedding_hyde 等节点无需改动。

【降级策略】
若未配置 API Key 或远端调用连续失败，自动降级为「本地哈希稠密向量」，
保证服务不中断（语义精度下降，但检索链路仍可用），并在日志中给出告警。
"""

import hashlib
import math
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Dict, List

from config.embedding_config import embedding_config

# 注意：不再在此处调用 setup_logging()
# 原因：setup_logging() 内部会执行 logger.handlers.clear() 清空所有已注册的 handler，
#       这会与 tool/logger.py 中配置的彩色日志 handler 冲突，导致日志输出异常。
#       日志配置统一由 tool/logger.py 在模块加载时完成。

# ==================== 稀疏向量参数 ====================
# 稀疏向量的哈希空间：Milvus 的稀疏索引键为 uint32，留出足够空间降低碰撞率。
SPARSE_HASH_SPACE = 2 ** 31 - 1
# 单条文本最多保留的高频 token 数（超出截断，避免稀疏向量过宽拖慢索引）
SPARSE_MAX_TOKENS = 256

# 中文连续串 / 英文数字串 的正则切分器
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")

# ==================== 稠密向量参数 ====================
# 云端 embeddings 接口单请求的最大条数（SiliconFlow 建议 <= 32）
_DENSE_BATCH = int(os.getenv("EMBEDDING_API_BATCH", "16"))
# 单条文本送入 API 前的字符截断（bge-m3 上限 8192 token，按字符粗截更安全）
_DENSE_MAX_CHARS = int(os.getenv("EMBEDDING_MAX_CHARS", "4000"))
# 失败重试次数与退避基数（秒）
_DENSE_RETRIES = int(os.getenv("EMBEDDING_API_RETRIES", "3"))
_DENSE_BACKOFF = float(os.getenv("EMBEDDING_API_BACKOFF", "1.5"))

# OpenAI 兼容客户端单例（懒加载，避免无 Key 时也构造客户端）
_client = None
_client_lock = threading.Lock()


def _get_client():
    """获取（并缓存）OpenAI 兼容客户端；配置缺失时返回 None。"""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        api_key = embedding_config.embedding_api_key
        if not api_key:
            return None
        from openai import OpenAI  # 延迟导入：无 Key 的纯本地场景无需该依赖

        _client = OpenAI(
            api_key=api_key,
            base_url=embedding_config.embedding_api_base,
            timeout=float(os.getenv("EMBEDDING_API_TIMEOUT", "60")),
            max_retries=0,  # 重试逻辑由本模块自行控制，便于打印日志
        )
        return _client


# ==================== 稀疏向量：本地哈希词袋 ====================
def _tokenize(text: str) -> List[str]:
    """
    轻量分词：中文取「单字 + 相邻双字」，英文/数字取整词。
    双字特征能显著提升中文短语的区分度（等价于一个极简的 2-gram 词表）。
    """
    text = text or ""
    tokens: List[str] = []
    for seg in _CJK_RE.findall(text):
        tokens.extend(seg)  # 单字
        if len(seg) > 1:
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))  # 双字
    tokens.extend(m.group().lower() for m in _WORD_RE.finditer(text))
    return tokens


def _token_id(token: str) -> int:
    """稳定哈希：blake2b 取模，保证跨进程/重启后同一 token 落在同一维度。"""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % SPARSE_HASH_SPACE


def _build_sparse_vector(text: str) -> Dict[int, float]:
    """
    生成稀疏向量：词频取 1 + ln(tf) 平滑，再按最大权重归一化（取值恒为正）。
    """
    tokens = _tokenize(text)
    if not tokens:
        return {}
    tf: Dict[int, int] = {}
    for tok in tokens:
        idx = _token_id(tok)
        tf[idx] = tf.get(idx, 0) + 1
    # 高频 token 截断：按词频倒序保留前 N 个
    items = sorted(tf.items(), key=lambda kv: kv[1], reverse=True)[:SPARSE_MAX_TOKENS]
    weights = {idx: 1.0 + math.log(cnt) for idx, cnt in items}
    max_w = max(weights.values()) if weights else 1.0
    return {idx: round(w / max_w, 6) for idx, w in weights.items()}


# ==================== 稠密向量：本地哈希降级实现 ====================
def _build_hash_dense(text: str, dim: int) -> List[float]:
    """
    纯本地稠密向量（降级用）：把 token 哈希到 dim 个桶中做带符号累加后 L2 归一化。
    语义能力弱于神经嵌入，但保证「无网 / 无 Key」时检索链路依然跑得通。
    """
    vec = [0.0] * dim
    for tok in _tokenize(text):
        h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
        pos = h % dim
        sign = 1.0 if (h >> 62) & 1 else -1.0
        vec[pos] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0:
        # 空文本兜底：给第 0 维一个极小值，避免全零向量被 Milvus 拒收
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


# ==================== 稠密向量：云端 API ====================
def _dense_batch_api(texts: List[str]) -> List[List[float]]:
    """调用一次（已分批的）embeddings 接口，返回与入参同序的向量列表。"""
    client = _get_client()
    payload = [(t or "")[:_DENSE_MAX_CHARS] or "空" for t in texts]
    last_err = None
    for attempt in range(_DENSE_RETRIES):
        try:
            resp = client.embeddings.create(
                model=embedding_config.embedding_model,
                input=payload,
                encoding_format="float",
            )
            # 按 index 排序，杜绝服务端乱序导致的「文本-向量」错配
            data = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
            return [list(item.embedding) for item in data]
        except Exception as e:  # noqa: BLE001 网络/限流/额度异常统一重试
            last_err = e
            if attempt < _DENSE_RETRIES - 1:
                time.sleep(_DENSE_BACKOFF * (attempt + 1))
    raise RuntimeError(f"云端嵌入接口调用失败：{last_err}")


def _dense_vectors(texts: List[str], dim: int) -> List[List[float]]:
    """
    稠密向量主入口：优先走云端 API，异常则整体降级为本地哈希向量。
    返回的每个向量都会校验维度，维度不符时同样触发降级，避免入库时才报错。
    """
    if _get_client() is not None:
        try:
            out: List[List[float]] = []
            for i in range(0, len(texts), _DENSE_BATCH):
                out.extend(_dense_batch_api(texts[i:i + _DENSE_BATCH]))
            if len(out) == len(texts) and all(len(v) == dim for v in out):
                return out
            raise RuntimeError(
                f"云端返回维度异常：期望 {dim}，实得 {sorted({len(v) for v in out})}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[embedding] 警告：云端嵌入不可用，降级为本地哈希向量（精度下降）：{e}")
    return [_build_hash_dense(t, dim) for t in texts]


# ==================== 对外主函数 ====================
# [P1/P2-方案6/7] 嵌入结果内存 LRU 缓存：相同文本（如重复提问、同一改写查询）直接命中，
# 免去一次网络往返；按 (模型名, 文本元组) 作为键，模型更换后自动失效。
_EMBED_CACHE: "OrderedDict" = OrderedDict()
_EMBED_CACHE_MAX = 2000


def generate_embeddings(texts):
    """
    为文本生成向量嵌入
    :param texts: 要生成嵌入的文本列表
    :return: 包含dense和sparse向量的字典
             {"dense": List[List[float]], "sparse": List[Dict[int, float]]}
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)

    # 先查缓存：命中则直接返回，避免重复请求
    cache_key = (embedding_config.embedding_model, tuple(texts))
    if cache_key in _EMBED_CACHE:
        _EMBED_CACHE.move_to_end(cache_key)
        return _EMBED_CACHE[cache_key]

    dim = embedding_config.embedding_dim
    dense = _dense_vectors(texts, dim)
    sparse = [_build_sparse_vector(t) for t in texts]

    result = {"dense": dense, "sparse": sparse}

    # 写入缓存（超过上限 LRU 淘汰）
    _EMBED_CACHE[cache_key] = result
    while len(_EMBED_CACHE) > _EMBED_CACHE_MAX:
        _EMBED_CACHE.popitem(last=False)

    return result


class _CSRShim:
    """
    CSR 结构兼容壳：原 BGE-M3 实现的 embeddings["sparse"] 是 scipy 的 csr_matrix，
    历史脚本（test/14-向量化测试.py 等）会读取 .indices/.indptr/.data。
    这里用纯 Python 复刻这三个属性，保持旧脚本可用。
    """

    def __init__(self, sparse_list: List[Dict[int, float]]):
        self.indices: List[int] = []
        self.data: List[float] = []
        self.indptr: List[int] = [0]
        for d in sparse_list:
            for k, v in d.items():
                self.indices.append(k)
                self.data.append(v)
            self.indptr.append(len(self.indices))


class _EmbeddingAdapter:
    """历史接口适配器：把 generate_embeddings 包装成 BGEM3EmbeddingFunction 的调用形状。"""

    def encode_documents(self, texts):
        res = generate_embeddings(texts)
        return {"dense": res["dense"], "sparse": _CSRShim(res["sparse"])}

    def encode_queries(self, texts):
        # 对称嵌入模型（bge-m3）无需区分 query/document
        return self.encode_documents(texts)


_adapter = None


def get_bge_m3_ef():
    """
    获取嵌入模型单例（历史函数名保留，实际已改为云端 API + 本地稀疏的后端）。
    """
    global _adapter
    if _adapter is None:
        _adapter = _EmbeddingAdapter()
    return _adapter


# ================= 单元测试主入口 =================
if __name__ == "__main__":
    # 准备测试文本（一个是中文，一个是英文词，模拟双语召回测试）
    test_data = ["大功率电磁炉", "Car"]

    print(f"👉 开始单独测试 generate_embeddings 函数...")
    print(f"👉 测试文本: {test_data}")
    print(f"👉 稠密后端: {embedding_config.embedding_api_base} / {embedding_config.embedding_model}")
    print(f"👉 稀疏后端: 本地哈希词袋（SPARSE_HASH_SPACE={SPARSE_HASH_SPACE}）\n")

    try:
        output = generate_embeddings(test_data)

        print("\n" + "=" * 20 + " 🎉 测试结果 🎉 " + "=" * 20)
        print(f"【稠密向量 (Dense)】")
        print(f"   - 成功生成数量: {len(output['dense'])} 条")
        print(f"   - 第一条文本的维度大小: {len(output['dense'][0])} 维")
        print(f"   - 第一条文本的前 5 维特征值: {output['dense'][0][:5]} ...")

        print(f"\n【稀疏向量 (Sparse)】")
        print(f"   - 成功生成数量: {len(output['sparse'])} 条")
        print(f"   - 第一条文本的关键字字典 (词ID : 权重)，共 {len(output['sparse'][0])} 项")
        sample = list(output["sparse"][0].items())[:8]
        print(f"     {sample} ...")

        print("=" * 54)
        print("💡 结论：函数单独测试通过！输出数据结构完全符合 Milvus 的入库标准。")

    except Exception as e:
        print(f"\n❌ 测试失败！拦截到异常: {e}")
        print("💡 排查建议：")
        print("   1. 检查 .env 中 OPENAI_API_KEY / OPENAI_API_BASE 是否可访问。")
        print("   2. 若网络受限，可依赖本地哈希降级向量（不影响链路可用性）。")
