# processor/query_processor/nodes/node_hyde_embed_prep.py
"""
[P1-方案6] HyDE 文档生成 + 向量批处理预处理节点。

在并行召回（node_search_embedding / node_search_embedding_hyde）之前，统一完成两件事：
1. 调用 LLM 生成 HyDE 假设性文档（hyde_doc）；
2. 将「改写查询」与「改写查询 + hyde_doc」一次性批处理向量化，
   把两条 dense/sparse 向量写入 state，供下游两路检索节点直接复用，避免各自独立向量化。

收益：把原本分散在两路检索节点里的 2 次独立 BGE-M3 embedding 调用合并为 1 次批处理，
BGE-M3 在 CPU 上单批推理比两次单条推理吞吐更高，整体延迟下降。
"""
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.task_utils import add_done_task


class NodeHydeEmbedPrep(NodeBase):
    """
    节点功能：HyDE 文档生成与向量批处理预处理。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_hyde_embed_prep"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        生成 HyDE 假设文档，并把 query / (query+hyde_doc) 一次性批处理向量化，写入 state。
        :param state: 含 rewritten_query 等字段
        :return: 更新后的 state（新增 hyde_doc / query_dense / query_sparse / hyde_dense / hyde_sparse）
        """
        # 优先用改写后的查询；缺失时回退原始查询
        rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""

        try:
            # 1. 生成 HyDE 假设性文档（理想回答范文）
            hyde_doc = self._step_1_create_hyde_doc(rewritten_query)
            # 2. 拼接 query 与假设文档，构造更丰富的检索文本
            combined_text = (rewritten_query + " " + hyde_doc).strip()

            # 3. [P1-方案6] 批处理向量化：一次调用同时得到 query 与 combined 的 dense/sparse，
            #    下游 node_search_embedding 用 query 向量、node_search_embedding_hyde 用 combined 向量，
            #    相比两路各调一次 generate_embeddings 更省时。
            embeddings = generate_embeddings([rewritten_query, combined_text])
            dense_list = embeddings.get("dense")
            sparse_list = embeddings.get("sparse")
            query_dense = dense_list[0]
            query_sparse = sparse_list[0]
            hyde_dense = dense_list[1]
            hyde_sparse = sparse_list[1]

            # 写入 state，供下游检索节点复用（缺失时下游会自行回退向量化，保证降级可用）
            state["hyde_doc"] = hyde_doc
            state["query_dense"] = query_dense
            state["query_sparse"] = query_sparse
            state["hyde_dense"] = hyde_dense
            state["hyde_sparse"] = hyde_sparse
            logger.info("HyDE 文档生成 + 向量批处理完成")

        except Exception as e:
            logger.exception(f"HyDE 文档生成/批处理向量化失败: {e}")
            # 降级：清空预计算向量，下游检索节点会各自回退到独立向量化，保证链路不中断
            state["hyde_doc"] = ""
            state["query_dense"] = None
            state["query_sparse"] = None
            state["hyde_dense"] = None
            state["hyde_sparse"] = None
        finally:
            # 确保无论正常/异常都标记任务完成，避免前端进度条卡死
            add_done_task(state.get("session_id"), self.name, state.get("is_stream"))

        return state

    def _step_1_create_hyde_doc(self, rewritten_query: str) -> str:
        """
        利用大模型根据用户查询生成假设性文档（Hypothetical Document）。
        HyDE 核心：用 LLM 生成一个"虚构但相关"的文档，用其向量去检索真实文档，
        缓解短查询与长文档在语义空间不匹配的问题。
        :param rewritten_query: 用户改写后的查询语句
        :return: LLM 生成的假设性文档内容
        """
        logger.info("开始生成假设性文档（HyDE 预处理）")
        try:
            llm = get_llm_client()
            hyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)
            hyde_doc = llm.invoke(hyde_prompt).content
            return hyde_doc
        except Exception as e:
            logger.exception(f"生成假设文档失败: {e}")
            raise e


if __name__ == "__main__":
    init_state = {
        "session_id": "test_prep_001",
        "original_query": "如何使用万用表测量电压？",
        "rewritten_query": "如何使用万用表测量电压？",
    }
    node = NodeHydeEmbedPrep()
    result = node(init_state)
    logger.info(f"hyde_doc 长度: {len(result.get('hyde_doc') or '')}")
    logger.info(f"query_dense 维度: {len(result.get('query_dense') or [])}")
    logger.info(f"hyde_dense 维度: {len(result.get('hyde_dense') or [])}")
