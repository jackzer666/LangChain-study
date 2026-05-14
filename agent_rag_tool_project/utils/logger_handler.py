import logging
import os
from datetime import datetime
from agent_rag_tool_project.utils.path_tool import get_abs_path

# 日志保存的根目录
LOG_ROOT = get_abs_path("logs")

# 确保日志目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志格式配置
DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO, # 在控制台日志中，级别小于INFO的就不会显示
        file_level: int = logging.DEBUG,
        log_file = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加handler，因为下面代码logger = get_logger()会被多个文件导入使用，每个都会创建一遍，这里需要就需要判断
    if logger.handlers:
        return logger

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)

    # 文件handler
    if not log_file:
        # 日志文件的存放路径
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger


# 快捷获取日志器
logger = get_logger()

# if __name__ == "__main__":
#     logger.info('信息日志')
#     logger.error('信息日志2')
#     logger.warning('信息日志3')
#     logger.debug('信息日志4')
