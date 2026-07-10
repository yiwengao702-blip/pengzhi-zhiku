# processor\import_processor\main_graph.py
"""
导入功能的主流程图
"""
import logging

from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.import_processor.base import setup_logging
from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.node_entry import NodeEntry
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus
from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.node_md_img import NodeMDImg
from processor.import_processor.nodes.node_pdf_to_md import NodePDFToMD
from processor.import_processor.state import ImportGraphState


class KBImportWorkflow:

    def __init__(self):
        """
        初始化工作流对象
        """
        self._compiled_graph = None

    @property
    def graph(self):
        """
        延迟加载（懒加载）：只在第一次使用工作流对象的时候编译图
        :return:
        """

        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph()
        return self._compiled_graph

    def build_graph(self):

        # 1.初始化图（工作流）
        graph = StateGraph(ImportGraphState)

        # 2.在图中注册节点
        graph.add_node("node_entry", NodeEntry())  # 入口：判断文档类型
        graph.add_node("node_pdf_to_md", NodePDFToMD())  # pdf转换为markdown
        graph.add_node("node_md_img", NodeMDImg())  # 解析文档中的图片的含义，并将其嵌入文档中
        graph.add_node("node_document_split", NodeDocumentSplit())  # 文档切分
        graph.add_node("node_item_name_recognition", NodeItemNameRecognition())  # 识别商品名称
        graph.add_node("node_bge_embedding", NodeBGEEmbedding())  # 将文本转成向量
        graph.add_node("node_import_milvus", NodeImportMilvus())  # 将向量和标量存入milvus

        # 3. 设置入口节点
        graph.set_entry_point("node_entry")
        # 4. 注册条件边
        graph.add_conditional_edges(
            "node_entry",
            self.route_after_entry,
            {
                # key：路由函数的返回值，value：节点的名字
                "node_pdf_to_md": "node_pdf_to_md",
                "node_md_img": "node_md_img",
                END: END
            }
        )

        # 5.注册顺序边
        graph.add_edge("node_pdf_to_md", "node_md_img")
        graph.add_edge("node_md_img", "node_document_split")
        graph.add_edge("node_document_split", "node_item_name_recognition")
        graph.add_edge("node_item_name_recognition", "node_bge_embedding")
        graph.add_edge("node_bge_embedding", "node_import_milvus")
        graph.add_edge("node_import_milvus", END)

        # 6. 编译工作流
        return graph.compile()

    @staticmethod
    def route_after_entry(state: ImportGraphState) -> str:
        """
        决定 入口节点之后 的走向
        :param state:
        :return:
        """
        if state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"

        elif state.get("is_md_read_enabled"):
            return "node_md_img"

        else:
            return END

    def run(self, state: ImportGraphState, stream: bool = False):
        # 设计出来 给别人调用的 你调用的时候  我就给你启动工作流
        if stream:
            #return self.graph.stream(state, stream_mode="values")
            return self.graph.stream(state)
        else:
            return self.graph.invoke(state)

# 工作流  导入阶段所有节点
if __name__ == "__main__":
    # 激活全局日志
    setup_logging()

    # 组件是不是都启动  minio （时间同步）

    init_state = {
        "import_file_path": r"D:\BaiduSyncdisk\13-企业级智能知识库系统项目\day01\资料\04-设备手册汇总\doc\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.pdf",
        "file_dir": r"D:\output"}
    workflow = KBImportWorkflow()
    res = workflow.run(init_state)

    logging.getLogger().info(res)

    # 打印编译后的图结构
    logging.getLogger().info(workflow.graph.get_graph().draw_ascii())
