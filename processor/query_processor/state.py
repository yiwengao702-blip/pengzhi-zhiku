# processor/query_processor/state.py

from typing import TypedDict, List

class QueryGraphState(TypedDict):
    """
    查询流程图状态
    包含整个查询流程中传递的所有数据。
    """

    session_id: str  # 会话ID
    message_id: str  # 消息ID

    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # 已向量化的假设性问题切片
    hyde_doc: str  # HyDE 节点生成的假设性答案文档，用于检索增强
    web_search_docs: list  # 网络搜索回来的文档

    # [P1-方案6] 上游 node_hyde_embed_prep 批处理好的向量，供两路检索节点直接复用，避免重复向量化
    query_dense: list  # 改写查询的稠密向量
    query_sparse: dict  # 改写查询的稀疏向量
    hyde_dense: list  # (改写查询 + HyDE 文档) 的稠密向量
    hyde_sparse: dict  # (改写查询 + HyDE 文档) 的稀疏向量

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: List[str]  # 提取出的商品名称
    rewritten_query: str  # 改写后的问题   理解用户的意图
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出