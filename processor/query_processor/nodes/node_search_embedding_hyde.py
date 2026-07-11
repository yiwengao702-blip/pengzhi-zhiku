# processor/query_processor/nodes/node_search_embedding_hyde.py

# processor/query_processor/nodes/node_search_embedding_hyde.py
from config.milvus_config import milvus_config
from config.retrieval_config import retrieval_config  # [P0] 检索阈值统一配置
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.json_format_utils import serialize_json
from utils.llm_utils import get_llm_client
from utils.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search, escape_milvus_string
from utils.task_utils import add_done_task


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        HyDE (Hypothetical Document Embedding) 检索节点
        核心思想：通过LLM生成假设性答案（HyDE文档），将其向量化后用于检索，以解决短查询语义稀疏问题。

        执行步骤：
        1. 参数提取：从会话状态中获取改写后的查询（rewritten_query）和已确认的商品名（item_names）。
        2. 生成假设文档 (Step 1)：调用LLM，基于用户问题生成一段假设性的理想回答（即HyDE文档）。
        3. 混合检索 (Step 2)：
           - 将“用户问题 + 假设文档”合并，生成BGE-M3稠密+稀疏向量。
           - 在Milvus中执行混合检索（带商品名过滤），召回最相似的知识切片。
        4. 结果封装：返回检索到的切片列表和生成的假设文档，更新会话状态。

        :param state: 会话状态字典，包含 rewritten_query, item_names 等
        :return: 包含 hyde_embedding_chunks (检索结果) 和 hyde_doc (假设文档) 的字典
        """

        # 1、用户问题和已确认商品名
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        #
        try:

            # [P1-方案6] hyde_doc 由上游 node_hyde_embed_prep 预生成并写入 state，这里直接复用；
            # 缺失（如单测/历史直接调用）时回退到自行生成，保证降级可用。
            hyde_doc = state.get("hyde_doc")
            if not hyde_doc:
                hyde_doc = self._step_1_create_hyde_doc(rewritten_query)
            logger.info("hyde_doc假设性回答: {}".format(hyde_doc))
            # 3、用"重写问题 + 假设文档"检索切片（向量复用上游批处理好的 hyde_dense/hyde_sparse）
            res = self._step_2_search_embedding_hyde(
                rewritten_query=rewritten_query,
                hyde_doc=hyde_doc,
                item_names=item_names,
                state=state
            )
            # 4、结果封装
            return {
                "hyde_embedding_chunks": res,
                "hyde_doc": hyde_doc,
            }

        except Exception as e:
            logger.exception(f"假设性文档向量搜索失败: {e}")
            # 返回标准格式空列表，避免下游 get() 时拿到 None（TypedDict 类型一致性）
            return {"hyde_embedding_chunks": [], "hyde_doc": ""}
        finally:
            # [BUG修复] add_done_task 移入 finally 块，确保异常/正常路径下都能标记任务完成
            # 避免前端进度条因异常而永久卡在"切片搜索(假设性文��)"运行中
            add_done_task(state.get("session_id"), self.name, state.get("is_stream"))

    def _step_2_search_embedding_hyde(
            self,
            rewritten_query: str,
            hyde_doc: str,
            item_names=None,
            state: dict = None
    ):
        """
        阶段2：利用“重写问题 + 假设性文档”生成 embedding，并到向量库检索切片。

        :param rewritten_query: 改写后的查询
        :param hyde_doc: Step 1 生成的假设性文档
        :param item_names: 商品名称列表，用于元数据过滤 (item_name in [...])
        :param state: 工作流状态，用于读取上游批处理好的 hyde_dense/hyde_sparse 向量
        :return: 检索结果列表
        """

        try:
            # 1、拼接查询与假设文档，形成更丰富的语义上下文
            # 这里把用户问题 + 假设答案拼在一起生成向量，相当于：
            # 既保留了用户的原始意图（rewritten_query）
            # 又增强了语义丰富度（hyde_doc）
            # combined
            combined_text = rewritten_query + " " + hyde_doc

            # 2、生成向量 (Dense + Sparse)
            # [P1-方案6] 优先复用上游 node_hyde_embed_prep 批处理好的 combined 向量；
            # 缺失（如单测/历史直接调用）时回退到独立向量化，保证兼容。
            h_dense = (state or {}).get("hyde_dense")
            h_sparse = (state or {}).get("hyde_sparse")
            if h_dense is not None and h_sparse is not None:
                dense_vec = h_dense
                sparse_vec = h_sparse
                logger.info("复用上游批处理的 hyde 向量（方案6）")
            else:
                embeddings = generate_embeddings([combined_text])
                dense_vec = embeddings.get("dense")[0]# 稠密向量
                sparse_vec = embeddings.get("sparse")[0]#稀疏向量

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
                logger.info(f"步骤2: 过滤条件: {expr}")
            else:
                logger.info("步骤2: 未指定商品名过滤，将全库检索")

            # 5、构造Milvus混合搜索请求对象
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                expr=expr,
                limit=10  # 底层检索返回数量（后续会再过滤为5，预留更多结果做重排序）
            )

            # 6、执行混合向量检索
            logger.info("步骤2: 开始执行 Milvus 混合检索...")
            client = get_milvus_client()
            res = hybrid_search(
                client=client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(retrieval_config.search_dense_weight, retrieval_config.search_sparse_weight),  # [P1/P2-方案8] 权重配置化
                output_fields=["chunk_id", "content", "item_name"],
            )

            # [P0-方案2] 按 min_cosine_score 过滤低质切片（安全网：整体偏低时保留原始召回）
            min_score = retrieval_config.min_cosine_score
            chunks = []
            if res and res[0]:
                hits = res[0]
                kept = [h for h in hits if (h.get("distance") or 0) >= min_score]
                chunks = kept if kept else hits
                logger.info(f"HyDE 按相似度下限 {min_score} 过滤：{len(hits)} → {len(chunks)} 条")
            return chunks

        except Exception as e:
            logger.error(f"步骤2: 检索过程发生异常: {e}")
            raise e

    def _step_1_create_hyde_doc(self, rewritten_query: str) -> str:
        """
        阶段1：利用大模型根据用户查询生成假设性文档（Hypothetical Document）。
        HyDE的核心在于：利用LLM生成一个“虚构但相关”的文档，用该文档的向量去检索真实的文档，
        从而缓解短查询（Query）与长文档（Document）在语义空间不匹配的问题。

        :param rewritten_query: 用户改写后的查询语句
        :return: LLM生成的假设性文档内容
        """

        logger.info("步骤1: 开始生成假设性文档")

        try:
            llm = get_llm_client()  # 默认  封装程度越高
            hyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)
            hyde_doc = llm.invoke(hyde_prompt).content
            return hyde_doc

        except Exception as e:
            logger.exception(f"步骤1: 生成假设文档失败: {e}")
            raise e


if __name__ == "__main__":
    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["BrotherHAK180烫金机", "BrotherHAK-180烫金机","BrotherHAK180烫金扫描仪"]
    }
    node_search_embedding_hyde = NodeSearchEmbeddingHyde()
    result = node_search_embedding_hyde(init_state)
    logger.info(serialize_json(result, indent=4))
