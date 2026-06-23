import logging
from typing import Dict
from processor.import_process.base import BaseNode, T, setup_logging


class TestNode(BaseNode):
    name = "node_test"
    def process(self, state: Dict) -> Dict:
        self.logger.error(f"{self.name}正在执行")
        self.logger.debug(f"{self.name}正在debug")
        self.logger.warning(f"{self.name}warning")
        self.log_step("MCP查询","开始执行")
        return {}

if __name__ == "__main__":
    setup_logging(logging.INFO)
    test_node = TestNode()
    test_node({"abc": 456})