# config/retrieval_config.py
"""
检索相关阈值统一配置（P0-方案2）。

此前存在两类问题：
1. .env 中的 MILVUS_MIN_COSINE_SCORE=0.75 没有任何代码读取，是「死配置」；
2. 型号对齐阈值 0.85 / 0.6 硬编码在 node_item_name_confirm.py 中，无法动态调参。

本配置把它们集中到这里，统一通过 .env 调整，便于灰度调参与运维。
"""
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RetrievalConfig:
    # 切片相似度下限：低于该分数的召回切片直接丢弃（或整体偏低时回退保留），不进入 rerank/LLM，降噪省 token
    min_cosine_score: float
    # 型号确认高置信阈值：单条匹配 > 该分数则直接确认型号
    item_confirm_high: float
    # 型号确认中置信阈值：无高置信结果时，>= 该分数的匹配作为候选让用户明确
    item_confirm_mid: float
    # 切片混合检索权重：稠密/稀疏（BGE-M3 dense/sparse），默认 0.8/0.2，可按业务调优
    search_dense_weight: float
    search_sparse_weight: float
    # RRF 融合中 HyDE 路权重（embedding 路保持 1.0），默认 0.7，用于削弱假设文档噪声
    rrf_hyde_weight: float


retrieval_config = RetrievalConfig(
    # 读取 .env 的 MILVUS_MIN_COSINE_SCORE；缺失时默认 0.75
    min_cosine_score=float(os.getenv("MILVUS_MIN_COSINE_SCORE", "0.75")),
    # 读取 .env 的 ITEM_CONFIRM_HIGH_SCORE；缺失时默认 0.85（与原硬编码一致）
    item_confirm_high=float(os.getenv("ITEM_CONFIRM_HIGH_SCORE", "0.85")),
    # 读取 .env 的 ITEM_CONFIRM_MID_SCORE；缺失时默认 0.60（与原硬编码一致）
    item_confirm_mid=float(os.getenv("ITEM_CONFIRM_MID_SCORE", "0.60")),
    # 读取 .env 的 SEARCH_DENSE_WEIGHT / SEARCH_SPARSE_WEIGHT；缺失时默认 0.8/0.2
    search_dense_weight=float(os.getenv("SEARCH_DENSE_WEIGHT", "0.8")),
    search_sparse_weight=float(os.getenv("SEARCH_SPARSE_WEIGHT", "0.2")),
    # 读取 .env 的 RRF_HYDE_WEIGHT；缺失时默认 0.7
    rrf_hyde_weight=float(os.getenv("RRF_HYDE_WEIGHT", "0.7")),
)

if __name__ == "__main__":
    print(retrieval_config)
