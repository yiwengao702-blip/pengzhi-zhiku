from processor.import_process.base import BaseNode
from processor.import_process.state import ImportGraphState

class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取
    """

    name = "node_item_name_recognition"
    def process(self, state: ImportGraphState):
        return state