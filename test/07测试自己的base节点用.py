import logging
from typing import Dict
from processor.import_process.base import BaseNode, setup_logging, T
from processor.import_process.exceptions import ImportProcessError

class TestNode(BaseNode):
    name = 'test_node'

    def process(self, state: Dict) -> Dict:
        state["processed_by"] = self.name
        self.log_step("数据清洗", "移除空值完成")
        return state

if __name__ == '__main__':
    setup_logging(logging.INFO)
    node = TestNode()
    initial_state = {"data": [1, 2, 3], "status": "raw"}

    print("初始状态:", initial_state)

    try:
        final_state = node(initial_state)
        print("\n最终状态:", final_state)
    except ImportProcessError as e:
        print("捕获到流程错误:", e)