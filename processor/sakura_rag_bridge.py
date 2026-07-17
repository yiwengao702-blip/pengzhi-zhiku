# knowledge_base/processor/sakura_rag_bridge.py
#
# knowledge_base → sakura_base.rag 桥接（融合 R4：消除 Milvus 同构检索重复）。
#
# 作用：把 knowledge_base 的「Milvus 稠密+稀疏混合检索」收敛到统一的 MilvusRetriever，
#       复用项目既有 BGE-M3（generate_embeddings 返回 {dense, sparse}）。
# 原则：不重写 node_import_milvus / node_search_embedding，仅提供等价能力的统一接入层；
#       生产切换时把这两个节点的「建集合/插入/混合检索」替换为本桥接即可（配置项一一对应）。
from __future__ import annotations

from typing import List, Optional

# [融合] 收敛到共享底座；未安装时给出清晰安装提示
try:
    from sakura_base.rag import MilvusRetriever
except ImportError as _imp_err:
    raise RuntimeError(
        "knowledge_base 现已依赖 sakura_base 共享底座，请先安装：\n"
        "  pip install -e <sakura_platform>/packages/sakura_base[rag]\n"
        f"原始错误：{_imp_err}"
    ) from _imp_err


def _embed(texts: List[str]):
    """embedding 契约：复用项目既有 BGE-M3，返回 {dense, sparse} 混合形态。

    惰性导入 generate_embeddings，避免 torch / BGE-M3 在桥接模块 import 期被强制加载。
    """
    from utils.embedding_utils import generate_embeddings  # noqa: WPS433 惰性导入
    emb = generate_embeddings(texts)
    return {"dense": emb["dense"], "sparse": emb["sparse"]}


def build_retriever(*, uri: str, collection: str, dim: int = 1024,
                    dense_weight: float = 0.5, sparse_weight: float = 0.5,
                    text_field: str = "content") -> MilvusRetriever:
    """构造对齐 knowledge_base 的 Milvus 检索器（稠密 COSINE + 稀疏 IP + WeightedRanker）。"""
    return MilvusRetriever(uri=uri, collection=collection, dim=dim,
                          embedder=_embed, dense_weight=dense_weight,
                          sparse_weight=sparse_weight, text_field=text_field)


def add_chunks(retriever: MilvusRetriever, chunks: List[dict]) -> None:
    """批量入库：chunks 为含 content + 业务字段（title/file_title/item_name...）的 dict 列表，
    形态与 node_import_milvus 的 chunks 一致；业务字段经 Milvus 动态列落库，检索时回带。"""
    texts = [c.get("content", "") for c in chunks]
    retriever.add(texts, metas=chunks)


def search(retriever: MilvusRetriever, query: str, top_k: int = 5) -> List[dict]:
    """检索：返回与 node_search_embedding 同构的结果列表
    [{content, distance, item_name, chunk_id, meta}, ...]，便于下游 rerank/LLM 直接消费。"""
    hits = retriever.search(query, top_k)
    out = []
    for h in hits:
        out.append({
            "content": h.text,
            "distance": h.score,
            "item_name": h.meta.get("item_name"),
            "chunk_id": h.meta.get("_id"),
            "meta": h.meta,
        })
    return out
