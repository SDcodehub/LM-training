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


def attach_file_handler(logger: logging.Logger, log_file: str):
    """
    Attaches a file handler to the existing logger so logs are saved to disk.
    """
    abs_path = os.path.abspath(log_file)
    # Avoid attaching duplicate file handlers for the same file
    for existing_handler in logger.handlers:
        if isinstance(existing_handler, logging.FileHandler) and getattr(existing_handler, "baseFilename", None) == abs_path:
            return
    handler = logging.FileHandler(abs_path, encoding="utf-8")
    # Reuse the same formatting as the console
    fmt = "%(asctime)s %(levelname)s:%(name)s:%(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    
    logger.addHandler(handler)