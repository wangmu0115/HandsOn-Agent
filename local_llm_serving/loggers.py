import logging
import sys
import time
from functools import wraps
from typing import Literal, Optional

import colorlog

loggerLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def setup_logger(name: str = "vector_search", level: loggerLevel = "DEBUG") -> logging.Logger:
    """
    Set up a colorful and informative logger

    Args:
        name: Logger name
        level: Logger level(DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    level = getattr(logging, level)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers = []  # Clear existing handlers
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    log_format = "%(log_color)s%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s%(reset)s"
    formatter = colorlog.ColoredFormatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time

    Args:
        logger: Logger instance to use
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger("vector_search")
            logger.debug("Starting execution of %s", func.__name__)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                logger.info("✅ %s completed successfully in %.4f seconds", func.__name__, time.perf_counter() - start)
                return result
            except Exception as e:
                logger.exception("❌ %s failed after %.4f seconds: ", func.__name__, time.perf_counter() - start, e)
                raise

        return wrapper

    return decorator
