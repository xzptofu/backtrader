"""Utilities to build factor datasets for model training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd

from .config import DataConfig
from .data import fetch_data, load_dataframe
from .factors import FactorPool, FactorSpec, build_factor_pool
from .utils import configure_logging

logger = logging.getLogger(__name__)


def _resolve_price_column(df: pd.DataFrame, column: str = "close") -> str:
    if column in df.columns:
        return column
    lowered = column.lower()
    for candidate in df.columns:
        if candidate.lower() == lowered:
            return candidate
    raise KeyError(f"Column '{column}' not found when computing forward returns.")


def compute_factors_for_data(
    data_cfg: DataConfig,
    factor_specs: Sequence[FactorSpec] | Sequence[str],
    *,
    dropna: bool = True,
) -> pd.DataFrame:
    """Fetch data per configuration and append factor columns."""
    configure_logging()
    data_path = fetch_data(data_cfg)
    price_df = load_dataframe(data_path)
    pool = build_factor_pool(factor_specs)
    factor_df = pool.compute(price_df)
    dataset = price_df.join(factor_df)
    if dropna:
        dataset = dataset.dropna()
    logger.info("Computed %d factors over %d rows", len(pool.factors), len(dataset))
    return dataset


def append_forward_returns(
    dataset: pd.DataFrame,
    horizons: Sequence[int],
    *,
    price_column: str = "close",
) -> pd.DataFrame:
    """Append forward return columns used as supervised targets."""
    if not horizons:
        return dataset
    resolved_col = _resolve_price_column(dataset, price_column)
    price_series = dataset[resolved_col]

    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("Forward return horizons must be positive integers.")
        column_name = f"forward_return_{horizon}"
        dataset[column_name] = price_series.shift(-horizon) / price_series - 1
    return dataset


def build_factor_dataset(
    data_cfg: DataConfig,
    factor_specs: Sequence[FactorSpec] | Sequence[str],
    *,
    dropna: bool = True,
    forward_horizons: Sequence[int] | None = None,
    price_column: str = "close",
    output_path: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, Optional[Path]]:
    """Build (and optionally persist) a factor dataset."""
    dataset = compute_factors_for_data(data_cfg, factor_specs, dropna=False)
    if forward_horizons:
        dataset = append_forward_returns(dataset, forward_horizons, price_column=price_column)
    if dropna:
        dataset = dataset.dropna()

    path: Optional[Path] = None
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(path)
        logger.info("Saved factor dataset to %s", path)

    return dataset, path

