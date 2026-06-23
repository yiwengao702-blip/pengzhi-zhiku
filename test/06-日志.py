# import logging
#
# # 1. 创建 Logger（记录器），并起个名字叫 "my_project"
# logger = logging.getLogger("my_project")
# logger.setLevel(logging.INFO)  # 设置全局级别
#
# # 2. 创建 Handler（处理器），让日志输出到屏幕（控制台）
# console_handler = logging.StreamHandler()
#
# # 3. 创建 Formatter（格式器），定制日志长相
# # %(asctime)s: 时间 | %(name)s: 记录器名 | %(levelname)s: 级别 | %(message)s: 消息
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
# # 4. 组装：把格式器塞给处理器，把处理器塞给记录器
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)
#
# if __name__ == "__main__":
#     # 5. 快乐地用起来
#     logger.info("系统启动成功")
#     logger.error("数据库连接超时")


# import logging
# import colorlog
#
# # 1. 创建 Logger（记录器）
# logger = logging.getLogger("boss_project")
# logger.setLevel(logging.INFO)
#
# # 2. 创建 Handler（处理器：使用 colorlog 的 StreamHandler）
# console_handler = colorlog.StreamHandler()
#
# # 3. 创建 ColoredFormatter（彩色格式器）
# # 注意：开头的 %(log_color)s 是配置彩色渲染
# color_formatter = colorlog.ColoredFormatter(
#     '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     log_colors={
#         'DEBUG': 'cyan',
#         'INFO': 'green',       # INFO 显示为绿色
#         'WARNING': 'yellow',   # WARNING 显示为黄色
#         'ERROR': 'red',         # ERROR 显示为红色
#         'CRITICAL': 'bold_red', # CRITICAL 显示为加粗红
#     }
# )
#
# # 4. 组装：格式器塞给处理器，处理器塞给记录器
# console_handler.setFormatter(color_formatter)
# logger.addHandler(console_handler)
#
#
# if __name__ == "__main__":
#     # 5. 测试彩色日志输出
#     logger.info("系统初始化成功啦")
#     logger.warning("警告：检测到短内容合并")
#     logger.error("错误：Milvus 连接失败")



import logging

# 1. 创建 Logger（记录器）
logger = logging.getLogger("file_logger")
logger.setLevel(logging.INFO)

# 2. 创建 FileHandler（文件处理器），指定文件名和编码（防止中文乱码）
file_handler = logging.FileHandler("app.log", encoding="utf-8")

# 3. 创建 Formatter（干净的纯文本格式器，不带任何颜色代码）
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 4. 组装
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


if __name__ == "__main__":
    # 5. 运行后，控制台没有任何输出，数据直接进文件
    logger.info("系统启动成功，日志已安全落盘")
    logger.error("由于网络波动，Milvus 写入数据重试了一次")