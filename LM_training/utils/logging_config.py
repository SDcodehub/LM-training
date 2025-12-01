"""
Logging configuration
"""
import logging
import os

def get_logger() -> logging.Logger:
    logger_instance = logging.getLogger(__name__)
    if logger_instance.handlers:
        return logger_instance

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s:%(name)s:%(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"  # short date+time
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger_instance.addHandler(handler)
    logger_instance.setLevel(level)
    logger_instance.propagate = False
    return logger_instance