from abc import ABC, abstractmethod
from typing import TypeVar

from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.task_utils import add_running_task

# 定义泛型
T = TypeVar("T")


class NodeBase(ABC):
    # 节点名称：占位符
    # 子类中要覆盖这个名字
    name: str = "base_node"

    def __call__(self, state: T) -> T:

        try:
            logger.info(f"{self.name} 开始执行")
            # add_running_task 小本本上记录  当前task——id  当前哪个节点正在运行  是否流式
            add_running_task(state['session_id'], self.name, state.get("is_stream"))
            state = self.process(state)

            logger.info(f"{self.name} 结束执行")

        except Exception as e:
            logger.error(f"{self.name} 执行异常: {e}")
            raise

        return state

    # 将来 你的子类节点想要继承NodeBase 就必须实现 process方法
    @abstractmethod
    def process(self, state: T) -> T:
        pass
