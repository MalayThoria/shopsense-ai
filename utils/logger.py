"""
Shared logging utility for the ShopSense AI pipeline.
All pipeline modules import this logger instead of using print().
Logs are written to logs/pipeline.log with timestamps.
"""

import logging
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger instance.

    Args:
        name: Name for the logger, typically __name__ of the calling module.

    Returns:
        A logging.Logger instance that writes to both file and console.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Ensure logs directory exists
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "pipeline.log"

    # File handler — detailed logs
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    # Console handler — info and above only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(levelname)-7s | %(message)s")
    console_handler.setFormatter(console_fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger