"""
Logging configuration for pretrain-experiments.

This module provides a centralized logging configuration for the entire package.

Usage:
    from .logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Message")
"""

import logging
import sys


# ANSI color codes for terminal output
BOLD = "\033[1m"
RESET = "\033[0m"


def setup_logging(level: int = logging.INFO, force: bool = False) -> None:
    """
    Configure logging for the pretrain-experiments package.

    Args:
        level: Logging level (default: INFO)
        force: If True, reconfigure even if already configured
    """
    package_logger = logging.getLogger("pretrain_experiments")

    if package_logger.handlers and not force:
        return

    if force:
        package_logger.handlers.clear()

    package_logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    package_logger.addHandler(console_handler)
    package_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    setup_logging()

    if name.startswith("pretrain_experiments"):
        return logging.getLogger(name)
    else:
        return logging.getLogger(f"pretrain_experiments.{name}")
