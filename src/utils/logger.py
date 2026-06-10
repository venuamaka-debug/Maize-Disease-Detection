# =============================================================
# src/utils/logger.py
# Centralized logging setup for the entire pipeline
# =============================================================
# Every module imports from here. This guarantees consistent
# log format, output location, and severity levels throughout
# the entire codebase. No print() statements anywhere else.
# =============================================================

import logging
import os
from datetime import datetime
from pathlib import Path


def get_logger(name: str, config: dict = None) -> logging.Logger:
    """
    Create and return a configured logger instance.

    Every module calls this function to get its own logger.
    All loggers share the same format and output destinations
    but are identified by their module name.

    Args:
        name: The module name - use __name__ when calling
        config: Logging configuration from settings.
                If None, uses safe defaults.

    Returns:
        A fully configured Python logger instance

    Example:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Pipeline started")
    """

    # ── Default configuration if none provided ──────────────
    log_level = "INFO"
    output_dir = "logs/preprocessing"
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    log_to_console = True
    log_to_file = True

    # ── Override defaults with provided config ───────────────
    if config:
        log_level    = config.get("level", log_level)
        output_dir   = config.get("output_dir", output_dir)
        log_format   = config.get("format", log_format)
        date_format  = config.get("date_format", date_format)
        log_to_console = config.get("log_to_console", log_to_console)
        log_to_file    = config.get("log_to_file", log_to_file)

    # ── Create the logger ────────────────────────────────────
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # ── Shared formatter ─────────────────────────────────────
    formatter = logging.Formatter(
        fmt=log_format,
        datefmt=date_format
    )

    # ── Console handler — prints to terminal ─────────────────
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # ── File handler — writes to log file ────────────────────
    if log_to_file:
        # Create log directory if it doesn't exist
        log_dir = Path(output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Timestamped log file — one per pipeline run
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = log_dir / f"preprocessing_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_pipeline_logger(config: dict = None) -> logging.Logger:
    """
    Convenience function for the main pipeline logger.
    Use this in pipeline.py and run_preprocessing.py.
    """
    return get_logger("maize_pipeline", config)