from processor.import_process.base import BaseNode
from processor.import_process.state import ImportGraphState

class NodePdfToMd(BaseNode):
    """
       PDF 转 Markdown 节点：PDF结构化解析
    """
    name = "node_pdf_to_md"
    def process(self, state: ImportGraphState):
        return state