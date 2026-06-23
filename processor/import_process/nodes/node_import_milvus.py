from processor.import_process.base import BaseNode
from processor.import_process.state import ImportGraphState

class NodeImportMilvus(BaseNode):
    """
       导入向量库节点：数据持久化
    """
    name = "node_import_milvus"
    def process(self, state: ImportGraphState):
        return state