"""
Centralized logging configuration.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Loading data...")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured = False


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    quiet: bool = False,
) -> None:
    """
    Set up root logger. Call once at startup (main.py).
    Safe to call multiple times — won't duplicate handlers.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if not quiet:
        # Use UTF-8 stream to avoid encoding errors on Windows (cp1252)
        import io
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        ch = logging.StreamHandler(stream)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "yfinance", "peewee", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger.
    Auto-configures basic logging if not already configured.
    """
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
