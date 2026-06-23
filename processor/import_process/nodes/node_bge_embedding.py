from processor.import_process.base import BaseNode
from processor.import_process.state import ImportGraphState

class NodeBgeEmbedding(BaseNode):
    """
       混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """
    name = "node_bge_embedding"
    def process(self, state: ImportGraphState):
        return state