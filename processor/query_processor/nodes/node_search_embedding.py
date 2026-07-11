# processor/query_processor/nodes/node_search_embedding.py
from config.milvus_config import milvus_config
from config.retrieval_config import retrieval_config  # [P0] 检索阈值统一配置
from processor.query_processor.base import NodeBase, T
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.json_format_utils import serialize_json
from utils.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search, escape_milvus_string
from utils.task_utils import add_done_task


class NodeSearchEmbedding(NodeBase):
    """
   节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
   """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        核心节点函数：基于已确认商品名+改写后的用户问题，执行Milvus向量数据库混合检索
        流程：用户问题向量化 → 构造带商品名过滤的混合搜索请求 → 执行稠密+稀疏混合检索 → 返回检索结果
        :param state: Dict - 会话状态字典，包含上游传递的核心信息，关键字段：
                      {
                          "rewritten_query": str,   # step4改写后的完整用户问题（含商品名）
                          "item_names": list[str],  # step7已确认的标准化商品名列表
                      }

        :return: Dict - 检索结果字典，仅包含embedding_chunks字段，供下游节点使用：
                 {
                     "embedding_chunks": List[Dict]  # Milvus检索结果列表，无结果则为空列表
                                                     # 每个元素为一条匹配的向量数据，含业务字段
                 }
        """

        try:

            # 1、用户问题和已确认商品名
            query = state.get("rewritten_query")
            item_names = state.get("item_names")

            # 2、生成向量 (Dense + Sparse)
            # [P1-方案6] 优先复用上游 node_hyde_embed_prep 批处理好的 query 向量；
            # 缺失（如单测/历史直接调用）时回退到独立向量化，保证兼容。
            q_dense = state.get("query_dense")
            q_sparse = state.get("query_sparse")
            if q_dense is not None and q_sparse is not None:
                dense_vec = q_dense
                sparse_vec = q_sparse
                logger.info("复用上游批处理的 query 向量（方案6）")
            else:
                embeddings = generate_embeddings([query])
                dense_vec = embeddings.get("dense")[0]
                sparse_vec = embeddings.get("sparse")[0]

            # 3. 获取Milvus的集合
            collection_name = milvus_config.chunks_collection

            # 4、处理 item_names：对每个值做安全转义后构造 Milvus 过滤表达式
            #    转义函数可防止商品名中的特殊字符（引号、反斜杠等）导致 Milvus 解析报错
            expr = None
            if item_names:
                # 逐个转义：先将 item_name 中的 \ → \\, " → \", ' → \', 再包裹双引号
                escaped_names = [escape_milvus_string(str(v)) for v in item_names]
                # Milvus 表达式要求字符串值用双引号包裹，如: item_name in ["A","B"]
                quoted = ", ".join(f'"{v}"' for v in escaped_names)
                expr = f"item_name in [{quoted}]"
                logger.info(f"过滤条件: {expr}")
            else:
                logger.info("未指定商品名过滤，将全库检索")

            # 5、构造Milvus混合搜索请求对象
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                expr=expr,# 表达式： BrotherHAK-180烫金机","BrotherHAK180烫金机
                limit=10  # 底层检索返回数量（后续会再过滤为5，预留更多结果做重排序）
            )

            # 6、执行混合向量检索
            logger.info("开始执行 Milvus 混合检索...")
            client = get_milvus_client()
            res = hybrid_search(# 检索 混合检索
                client=client,
                collection_name=collection_name,  # 检索的目标集合名（文本片段向量集合）
                reqs=reqs,  # 构造好的混合搜索请求对象（稠密+稀疏）
                ranker_weights=(retrieval_config.search_dense_weight, retrieval_config.search_sparse_weight),  # [P1/P2-方案8] 权重配置化
                output_fields=["chunk_id", "content", "item_name"]  # 指定返回的业务字段
            )
            # 7、构造并返回结果：[P0-方案2] 按 min_cosine_score 过滤低质切片，避免噪声进入 rerank/LLM
            min_score = retrieval_config.min_cosine_score
            if res and res[0]:
                hits = res[0]
                kept = [h for h in hits if (h.get("distance") or 0) >= min_score]
                # 安全网：过滤后为空（整体相似度偏低）时保留原始召回，避免阈值过严导致全量拒答
                if not kept and hits:
                    kept = hits
                logger.info(f"按相似度下限 {min_score} 过滤：{len(hits)} → {len(kept)} 条")
                return {"embedding_chunks": kept}
            return {"embedding_chunks": []}

        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            # 返回标准格式空列表，避免下游 get() 时拿到 None（TypedDict 类型一致性）
            return {"embedding_chunks": []}
        finally:
            # [BUG修复] add_done_task 移入 finally 块，确保异常/正常路径下都能标记任务完成
            # 避免前端进度条因异常而永久卡在"切片搜索"运行中
            add_done_task(state.get("session_id"), self.name, state.get("is_stream"))


if __name__ == "__main__":
    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["BrotherHAK180烫金", "BrotherHAK-180烫金机","BrotherHAK180烫金扫描仪"]
    }
    node_search_embedding = NodeSearchEmbedding()
    result = node_search_embedding(init_state)
    logger.info(serialize_json(result, indent=4))# 打印
