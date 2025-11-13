"""Utilities for downloading and preparing market data."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "The yfinance package is required for data downloading. Install it with `pip install yfinance`."
    ) from exc

from .config import DataConfig
logger = logging.getLogger(__name__)


def ensure_directory(path: Path) -> Path:
    """Ensure the directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_yfinance(cfg: DataConfig) -> pd.DataFrame:
    """Download data via yfinance."""
    logger.info(
        "Downloading %s data from %s to %s with interval %s",
        cfg.symbol,
        cfg.resolved_start().date(),
        (cfg.resolved_end() or pd.Timestamp.utcnow()).date(),
        cfg.interval,
    )

    data = yf.download(
        tickers=cfg.symbol,
        start=cfg.resolved_start(),
        end=cfg.resolved_end(),
        interval=cfg.interval,
        auto_adjust=cfg.auto_adjust,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise RuntimeError(f"No data returned for {cfg.symbol} with interval {cfg.interval}.")

    # yfinance returns a MultiIndex for multi-ticker or if actions data included.
    if isinstance(data.columns, pd.MultiIndex):
        # Select the price component if available
        data = data.xs("Close", level=1, axis=1, drop_level=False)
    data.index.name = "datetime"
    return data


def fetch_data(cfg: Optional[DataConfig] = None, overwrite: bool = False) -> Path:
    """Fetch historical data based on the provided configuration.

    Parameters
    ----------
    cfg:
        DataConfig describing the data to download. If omitted, the default configuration is used.
    overwrite:
        Whether to re-download the data even if it already exists on disk.

    Returns
    -------
    Path
        The path to the CSV file containing the downloaded data.
    """

    cfg = cfg or DataConfig()
    destination = cfg.path()
    ensure_directory(destination.parent)

    if destination.exists() and not overwrite and cfg.cache:
        logger.info("Using cached data at %s", destination)
        return destination

    if cfg.source != "yfinance":
        raise NotImplementedError(f"Data source {cfg.source!r} is not implemented yet.")

    dataframe = _download_yfinance(cfg)
    dataframe.reset_index(inplace=True)
    dataframe.rename(
        columns={
            "Datetime": "datetime",
            "Date": "datetime",
        },
        inplace=True,
    )

    dataframe.to_csv(destination, index=False)
    logger.info("Saved data to %s", destination)
    return destination


def load_dataframe(path: Path, parse_dates: bool = True) -> pd.DataFrame:
    """Load the cached data into a DataFrame ready for Backtrader."""
    df = pd.read_csv(path, parse_dates=["datetime"] if parse_dates else None)
    df.sort_values("datetime", inplace=True)
    df.set_index("datetime", inplace=True)
    return df


def get_latest_data(cfg: Optional[DataConfig] = None) -> pd.DataFrame:
    """Convenience helper to fetch (and cache) then load the latest data as a DataFrame."""
    cfg = cfg or DataConfig()
    path = fetch_data(cfg)
    return load_dataframe(path)


@contextlib.contextmanager
def dataframe_to_bt_feed(df: pd.DataFrame, name: str = "data"):
    """Context manager that yields a Backtrader PandasData feed from a DataFrame."""
    from backtrader.feeds import PandasData

    yield PandasData(dataname=df, name=name)

