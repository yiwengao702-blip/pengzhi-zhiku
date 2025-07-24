# processor/query_processor/nodes/node_rrf.py
from typing import List, Tuple, Dict

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.task_utils import add_done_task
from config.retrieval_config import retrieval_config  # [P1/P2-方案8] 检索权重配置化

# [P0-方案3] RRF 融合上限：与 node_rerank 的 RERANK_MAX_TOPK(10) 对齐，避免前置截断压制相关性上限
RRF_MAX_RESULTS: int = 10
# [P0-方案3 / P1/P2-方案8] HyDE 路权重：默认 0.7，可被 .env 的 RRF_HYDE_WEIGHT 覆盖。
# 注意：这里**不再**在模块加载时固化，而是每次 process 调用实时读取 retrieval_config，
# 以便评测脚本（tools/eval_retrieval.py --grid-search）能运行时改写权重并立即生效，
# 也契合「改后重启后端即生效」的设计意图（连运行时动态调参都不必重启）。


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE）进行加权融合排序。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 1. 获取各搜索路的搜索结果
        embedding_chunks_list = state.get("embedding_chunks") or []
        hyde_embedding_chunks_list = state.get("hyde_embedding_chunks") or []

        # 2. 清洗数据
        # embedding_search_list = [
        #     {"chunk_id": "chunk_1", "content": "向量搜索结果#1"},
        #     {"chunk_id": "chunk_2", "content": "向量搜索结果#2"},
        #     {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}
        # ]
        # embedding_search_list = []
        # for doc in embedding_chunks_list:
        #     if isinstance(doc, dict):
        #         embedding_search_list.append(doc.get("entity"))

        embedding_search_list = [doc.get("entity") for doc in embedding_chunks_list if isinstance(doc, dict)]
        hyde_embedding_search_list = [doc.get("entity") for doc in hyde_embedding_chunks_list if isinstance(doc, dict)]

        # rrf_inputs = [
        #     {"list":embedding_search_list, "weight":1.0},
        #     {"list":hyde_embedding_search_list, "weight":1.0}
        # ]

        # 3. 构建融合前的数据结构
        rrf_inputs = [
            (embedding_search_list, 1.0),# embedding 路权重保持 1.0
            # [P0-方案3] HyDE 路权重：实时读取配置（默认 0.7），支持运行时调参与 GridSearch
            (hyde_embedding_search_list, retrieval_config.rrf_hyde_weight)
        ]

        # 4. 根据RRF公式进行数据的融合
        rrf_merge_results = self._rrf_merge(rrf_inputs, max_results=RRF_MAX_RESULTS)  # [P0-方案3] 上限 3 → 10

        # 5. 将分数清洗出去（下一个步骤重新排序，不参考这个步骤的分数）
        rrf_chunks = [doc for doc, _ in rrf_merge_results]

        state["rrf_chunks"] = rrf_chunks
        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
        return state

    def _rrf_merge(self, rrf_inputs, k: int = 60, max_results: int = None) -> List[Tuple[Dict, float]]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法
        :param rrf_inputs: 输入数据列表，每个元素是一个元组，包含一个列表和一个权重
        :param k: RRF中的平滑参数
        :param max_results: 最大结果数量限制，如果是None，则获取所有结果
        :return: 融合后的结果列表 [({},得分),({},得分),...]
        """

        # 1. 遍历需要融合的数据，并组装数据
        chunk_data = {}  # 最后数据
        chunk_scores = {} # 综合得分
        for rrf_input, weight in rrf_inputs:
            # 对当前这一路计算
            for rank, doc in enumerate(rrf_input, start=1):
                chunk_id = doc.get("chunk_id")# 拿片段 名称
                # [BUG修复] 防御 chunk_id 为 None 的情况，使用 id(doc) 生成唯一标识
                # 避免多个缺失 chunk_id 的文档共用 None 作为字典 key，导致 RRF 分数错误累加和内容覆盖
                safe_id = chunk_id if chunk_id is not None else f"_no_id_{id(doc)}"
                # 计算每个文档的得分，��将得分组织在list中
                chunk_scores[safe_id] = chunk_scores.get(safe_id, 0.0) + weight / (k + rank)
                # 将所有文档组织在list中
                chunk_data.setdefault(# setdefault 如果这个键不存在就创建 吧综合得分写进去 如果存在就忽略 片段名 和内容
                    safe_id,
                    doc
                )# 只保留第一次遇到的那个文档实体，不会覆盖。"如果 chunk_id 这个键还不存在，就把它存进字典，值设为 doc；如果已经存在了，就什么也不做。"

        # 2. 将得分和文档组装在一起
        unsorted_results = [(chunk_data[chunk_id], score) for chunk_id, score in chunk_scores.items()]

        # 3. 按照得分降序排列
        sorted_results = sorted(unsorted_results, key=lambda x:x[1], reverse=True)
        # topN  取前几条
        return sorted_results[:max_results] if max_results else sorted_results

if __name__ == '__main__':

    # 模拟两路检索结果
    mock_state = {
        # 直接硬搜得到相似度高的片段
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        # 通过假设性文档搜到的相似度高的片段   比如 chunk_1 两路排名里面都高
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#3"}},
        ]
    }

    node_rrf = NodeRrf()# __call__  process
    result = node_rrf(mock_state)
    logger.info(serialize_json(result, indent=4))# 打印日志