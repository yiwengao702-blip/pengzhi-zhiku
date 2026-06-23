from dataclasses import dataclass, field
from typing import Optional
import datetime

@dataclass
class SimpleLogger:
    """日志记录器类"""
    log_level: str = "INFO"
    init_time: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    def log(self, message: str):
        print(f"[{self.init_time}] [{self.log_level}] {message}")

_logger_instance: Optional[SimpleLogger] = None

def get_logger() -> SimpleLogger:
    """
       获取日志记录器的单例对象。
       与 exceptions 中的 get_config() 逻辑完全一致。
       """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = SimpleLogger()
    return _logger_instance

if __name__ == "__main__":
    print("--- 第一次获取 Logger ---")
    logger1 = get_logger()
    logger1.log("这是第一条日志。")

    import time
    time.sleep(3)

    print("\n--- 第二次获取 Logger ---")
    logger2 = get_logger()
    logger2.log("这是第二条日志。")

    print("\n--- 验证唯一性 ---")
    is_same = logger1 is logger2
    print(f"logger1 和 logger2 是同一个实例吗？ {is_same}")