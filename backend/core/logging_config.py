"""
日志配置模块

提供统一的日志初始化，支持控制台输出 + 文件持久化（带轮转）。
所有模块通过 logging.getLogger(__name__) 获取的 logger 均自动生效。
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """初始化日志系统（控制台 + 文件双输出）

    Args:
        log_dir: 日志文件目录
        log_level: 日志级别，如 DEBUG/INFO/WARNING/ERROR
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 检查是否已添加文件 handler（uvicorn 会添加控制台 handler，
    # 但不会添加文件 handler，所以只检查文件 handler 是否存在）
    has_file_handler = any(
        isinstance(h, RotatingFileHandler) for h in root_logger.handlers
    )
    if has_file_handler:
        return

    # 文件 handler（轮转）— 始终添加，不受 uvicorn handler 影响
    file_handler = RotatingFileHandler(
        filename=log_dir / "factorhub.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 降低第三方库日志级别，避免噪音
    _silence_noisy_loggers()


def _silence_noisy_loggers():
    """降低第三方库的日志级别"""
    for name in ("uvicorn.access", "uvicorn.error", "httpx"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
