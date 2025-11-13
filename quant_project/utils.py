"""Utility helpers for the quantitative trading project."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from . import REPORTS_DIR


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging if it hasn't been configured yet."""
    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def ensure_reports_dir(path: Path | None = None) -> Path:
    """Ensure the reports directory exists."""
    directory = path or REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamped_filename(prefix: str, suffix: str = ".json", directory: Path | None = None) -> Path:
    """Generate a timestamped file path."""
    ensure_reports_dir(directory)
    directory = directory or REPORTS_DIR
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return directory / f"{prefix}-{timestamp}{suffix}"

