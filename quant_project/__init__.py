"""Utilities for building an automated quantitative trading pipeline with Backtrader."""

from pathlib import Path

from .backtest import build_report, create_cerebro, run_backtest
from .data import fetch_data, get_latest_data, load_dataframe
from .optimizer import run_optimization
from .trading import run_trade_session

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
REPORTS_DIR = PACKAGE_ROOT / "reports"

__all__ = [
    "PACKAGE_ROOT",
    "DATA_DIR",
    "REPORTS_DIR",
    "fetch_data",
    "get_latest_data",
    "load_dataframe",
    "run_backtest",
    "create_cerebro",
    "build_report",
    "run_optimization",
    "run_trade_session",
]
