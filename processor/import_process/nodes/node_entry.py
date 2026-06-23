from processor.import_process.base import BaseNode
from processor.import_process.state import ImportGraphState


class NodeEntry(BaseNode):
    """
        入口节点：任务分发
    """
    name = "node_entry"

    def process(self, state: ImportGraphState):
        # 判断文件的扩展名是pdf还是md
        # 如果是pdf，则

        # return {
        #     "is_md_read_enabled": False,
        #     "is_pdf_read_enabled": True
        # }
        return {
            "is_md_read_enabled": True,
            "is_pdf_read_enabled": False
        }
        # return {
        #     "is_md_read_enabled": False,
        #     "is_pdf_read_enabled": False
        # }

        # return state
