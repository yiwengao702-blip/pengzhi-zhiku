from typing import Optional

from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph
from processor.query_processor.nodes.node_answer_output import NodeAnswerOutput
from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.node_rerank import NodeRerank
from processor.query_processor.nodes.node_rrf import NodeRrf
from processor.query_processor.nodes.node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_processor.nodes.node_web_search_mcp import NodeWebSearchMcp
from processor.query_processor.nodes.node_hyde_embed_prep import NodeHydeEmbedPrep  # [P1-方案6] 上游批处理节点
from processor.query_processor.state import QueryGraphState
from tool.logger import logger

# 初始化环境变量（类加载前执行，保证全局生效）
load_dotenv()


class KBQueryWorkflowV2:
    """
    知识库查询工作流类
    封装LangGraph工作流的构建、编译、执行逻辑，支持自定义配置和多实例运行
    """
    def __init__(self):
        """初始化工作流：创建状态图、注册节点、定义路由规则"""
        # 1. 初始化LangGraph状态图
        self.workflow = StateGraph(QueryGraphState)
        # 2. 初始化所有业务节点（实例属性，支持多实例隔离）
        self._init_nodes()
        # 3. 注册节点到工作流
        self._register_nodes()
        # 4. 设置入口和路由规则
        self._setup_routes()
        # 5. 编译工作流（懒加载，首次执行时编译）
        self._compiled_app: Optional[object] = None

    def _init_nodes(self):
        """初始化所有业务节点（私有方法，封装节点创建逻辑）"""
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_hyde_embed_prep = NodeHydeEmbedPrep()  # [P1-方案6]
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()

    def _register_nodes(self):
        """注册所有节点到工作流（私有方法，统一管理节点注册）"""
        # 节点标识与实例属性名保持一致，便于维护
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm) # 确认主体
        self.workflow.add_node("node_search_embedding", self.node_search_embedding) # 向量搜索
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde) #假设性答案向量搜索
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp) # 联网搜索
        self.workflow.add_node("node_hyde_embed_prep", self.node_hyde_embed_prep) # [P1-方案6] HyDE 文档+向量批处理
        self.workflow.add_node("node_rrf", self.node_rrf) # 排序
        self.workflow.add_node("node_rerank", self.node_rerank) # 重排
        self.workflow.add_node("node_answer_output", self.node_answer_output) # 生成

    def _route_after_item_name_confirm(self, state: QueryGraphState) -> str:
        """主体名称确认后的条件路由函数"""
        if state.get("answer"):
            """
            这主要发生在 node_item_name_confirm 节点无法直接确定唯一的商品型号，从而需要“反问用户”或“拒绝回答”的场景。
            具体来说，有以下两种情况会导致 state 中直接出现 answer ，从而跳过后续的检索流程，直接输出：
            1. 多选一（反问用户） ：
            - 场景 ：用户问得太模糊（比如“华为P60”），系统发现数据库里有“华为P60 128G”和“华为P60 Art”两个型号，不足以直接确认。
            - 处理 ：节点会生成一条反问句作为 answer ，例如：“您是想问以下哪个产品：华为P60 128G、华为P60 Art？请明确一下型号。”
            - 结果 ：此时不需要去检索文档了，直接把这句话发给用户让他选。

            2. 查无此人（拒绝回答） ：
            - 场景 ：用户问了一个系统里压根没有的商品（比如“小米15”，库里只有华为的数据）。
            - 处理 ：节点会生成一条拒绝句作为 answer ，例如：“抱歉，未找到相关产品，请提供准确型号以便我为您查询。”
            - 结果 ：同样不需要后续检索，直接结束流程。
            """
            return "node_answer_output"

        # 否则进入 [P1-方案6] 上游批处理节点（生成 HyDE 文档 + 批处理向量），再并行召回
        return "node_hyde_embed_prep"


    def _setup_routes(self):
        """设置工作流路由规则"""
        # 1、设置入口节点
        self.workflow.set_entry_point("node_item_name_confirm")

        # 2、注册条件路由边
        self.workflow.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,
            {
                "node_answer_output": "node_answer_output",
                "node_hyde_embed_prep": "node_hyde_embed_prep",
            }
        )

        # 2.5 [P1-方案6] 批处理节点完成后，并行扇出到三路召回
        self.workflow.add_edge("node_hyde_embed_prep", "node_search_embedding")
        self.workflow.add_edge("node_hyde_embed_prep", "node_search_embedding_hyde")
        self.workflow.add_edge("node_hyde_embed_prep", "node_web_search_mcp")

        # 3. 多路搜索结果合并
        self.workflow.add_edge("node_search_embedding", "node_rrf")
        self.workflow.add_edge("node_search_embedding_hyde", "node_rrf")
        self.workflow.add_edge("node_web_search_mcp", "node_rrf")

        # 4. 排序 -> 重排 -> 生成 -> 结束
        self.workflow.add_edge("node_rrf", "node_rerank")
        self.workflow.add_edge("node_rerank", "node_answer_output")
        self.workflow.add_edge("node_answer_output", END)


    def compile(self):
        """编译工作流（公开方法，支持手动触发编译）"""
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    def run(self, initial_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        """
        统一执行入口，支持切换invoke/stream
        :param initial_state:  初始状态对象
        :param stream: 是否是流式输出
        :return: 执行完成后的状态对象
        """
        """"""
        if not self._compiled_app:
            self.compile()

        # [BUG修复] 注释掉生产环境的 ASCII 图打印，避免每次查询污染 stdout 日志
        # 如需调试流程图，可取消下面注释
        # self._compiled_app.get_graph().print_ascii()

        if stream:
            return self._compiled_app.stream(initial_state)
        else:
            return self._compiled_app.invoke(initial_state)

    def run_with_timing(self, initial_state: QueryGraphState, stream: bool = False):
        """带逐节点耗时的执行入口（「检索速度/命中指标」的数据源）。

        使用 stream_mode=["updates","values"] 双通道执行：
        - updates：每个超步（节点）完成时产出 {节点名: 状态增量}。相邻两次 updates 之间的
          墙钟间隔即该超步的执行耗时；并行超步（三路召回）会在同一个 chunk 里同时出现
          多个节点名，它们共享同一份耗时（对并行分支取 max 而非 sum，见服务层组装逻辑）。
        - values：每个超步完成后的全量状态快照，最后一个快照即最终状态，
          服务层从中读取 answer / reranked_docs / rrf_chunks 组装指标。

        返回 (final_state, timings_ms, total_ms)：
        - final_state: 最终状态字典
        - timings_ms: {节点名: 累计耗时毫秒}
        - total_ms: 全部超步耗时之和（端到端墙钟，不含服务层开销）

        说明：无论 stream 真假都统一走 stream 通道执行——节点内部的流式推送行为
        由 state["is_stream"] 控制，与这里的执行通道无关，语义与 invoke 完全一致。
        原有 run() 保持不变，评测脚本等既有调用方不受影响。
        """
        if not self._compiled_app:
            self.compile()

        import time as _time
        timings: dict = {}
        total_ms = 0.0
        final_state: dict = {}
        t_prev = _time.perf_counter()
        for mode, chunk in self._compiled_app.stream(initial_state, stream_mode=["updates", "values"]):
            t_now = _time.perf_counter()
            if mode == "updates" and isinstance(chunk, dict):
                # 距上个超步完成事件的时间差 ≈ 当前超步的执行耗时
                dt_ms = (t_now - t_prev) * 1000.0
                total_ms += dt_ms
                for node_name in chunk:
                    if node_name == "__end__":
                        continue
                    timings[node_name] = timings.get(node_name, 0.0) + dt_ms
                # 仅在 updates 时校准计时基准；values 快照的产出间隔不参与计时
                t_prev = t_now
            elif mode == "values" and isinstance(chunk, dict):
                final_state = chunk
        return final_state, timings, total_ms

# ===================== 用法示例 =====================
if __name__ == "__main__":

    # 定义初始状态
    init_state = { "original_query": "如何使用万用表测量电压？"}

    workflow = KBQueryWorkflowV2()
    # final_state = workflow.run(init_state)
    # logger.info(final_state)
    # 流式输出
    for chunk in workflow.run(init_state, stream=True):
        logger.warning(chunk)

    # 打印编译后的图结构
    logger.info(workflow.compile().get_graph().draw_ascii())