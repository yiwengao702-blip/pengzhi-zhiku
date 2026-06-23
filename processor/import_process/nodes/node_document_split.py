from processor.import_process.base import BaseNode
from processor.import_process.state import ImportGraphState

class NodeDocumentSplit(BaseNode):
    """
        文档切分节点：智能文档切片
    """
    name = "node_document_split"
    def process(self, state: ImportGraphState):
        return state