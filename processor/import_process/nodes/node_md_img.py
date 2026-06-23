from processor.import_process.base import BaseNode
from processor.import_process.state import ImportGraphState

class NodeMdImg(BaseNode):
    """
       MarkDown图片处理节点：多模态图片理解
    """
    name = "node_md_img"
    def process(self, state: ImportGraphState):
        return state