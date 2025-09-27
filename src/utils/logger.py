# utils/logger.py
# 日志工具模块：提供日志记录功能
# 这个模块定义了一个Logger类，用于同时输出到控制台和文件
# 主要用于记录攻击过程中的详细信息，便于调试和结果分析

import sys
from pathlib import Path

class Logger:
    """
    自定义日志记录器类，同时输出到控制台和文件。

    该类重定向sys.stdout，使所有print语句同时输出到控制台和指定的日志文件。
    适用于需要同时查看实时输出和保存完整日志的场景。

    Attributes:
        terminal: 原始的sys.stdout，用于控制台输出
        log_file: 打开的日志文件对象，用于文件输出
    """

    def __init__(self, filepath: Path):
        """
        初始化Logger实例。

        Args:
            filepath: 日志文件的完整路径。文件会被创建或覆盖。
        """
        self.terminal = sys.stdout  # 保存原始的stdout
        self.log_file = open(filepath, "w", encoding="utf-8")  # 以写入模式打开日志文件

    def write(self, message: str):
        """
        写入消息到控制台和日志文件。

        该方法被sys.stdout重定向调用，实现同时输出。

        Args:
            message: 要写入的消息字符串
        """
        self.terminal.write(message)  # 输出到控制台
        self.log_file.write(message)  # 输出到文件

    def flush(self):
        """
        刷新输出缓冲区。

        确保消息立即写入控制台和文件，而不是等待缓冲区满。
        """
        self.terminal.flush()  # 刷新控制台缓冲区
        self.log_file.flush()  # 刷新文件缓冲区

    def close(self):
        """
        关闭日志文件。

        在程序结束时调用此方法，确保所有数据都被写入文件。
        """
        self.log_file.close()  # 关闭文件对象