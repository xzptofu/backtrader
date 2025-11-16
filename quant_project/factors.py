"""Factor pool infrastructure for offline feature generation and online inference."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Type

import numpy as np
import pandas as pd


def _auto_column_lookup(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a DataFrame column with case-insensitive matching."""
    if column in df.columns:
        return df[column]
    lowered = column.lower()
    for candidate in df.columns:
        if candidate.lower() == lowered:
            return df[candidate]
    raise KeyError(f"Column '{column}' not found. Available columns: {df.columns.tolist()}")


@dataclass(frozen=True)
class FactorSpec:
    """Declarative specification for a factor."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    alias: str | None = None


class FactorFormula:
    """Base factor formula with unified interface."""

    description: str = ""
    required_columns: Tuple[str, ...] = ("close",)
    min_history: int = 1

    def __init__(self, *, alias: str | None = None, **params: Any):
        self.params = params
        self.alias = alias
        self.output_name = alias or self.default_output_name()

    @classmethod
    def key(cls) -> str:
        name = cls.__name__
        if name.endswith("Factor"):
            name = name[:-6]
        return name.replace("_", "").lower()

    def default_output_name(self) -> str:
        return self.key()

    def compute(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def compute_latest(self, df: pd.DataFrame) -> float:
        series = self.compute(df)
        if series.empty:
            return math.nan
        value = series.iloc[-1]
        return float(value) if pd.notna(value) else math.nan

    def _column(self, df: pd.DataFrame, name: str) -> pd.Series:
        return _auto_column_lookup(df, name)


class MomentumFactor(FactorFormula):
    """Simple price momentum over a lookback window."""

    description = "Price momentum calculated as pct change over the specified window."
    required_columns = ("close",)

    def __init__(self, window: int = 63, price_col: str = "close", annualize: bool = False, *, alias: str | None = None):
        self.window = int(window)
        if self.window <= 0:
            raise ValueError("MomentumFactor 'window' must be positive.")
        self.price_col = price_col
        self.annualize = bool(annualize)
        self.min_history = self.window + 1
        super().__init__(alias=alias, window=window, price_col=price_col, annualize=annualize)

    def default_output_name(self) -> str:
        return f"momentum_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        price = self._column(df, self.price_col)
        momentum = price / price.shift(self.window) - 1
        if self.annualize and self.window:
            scaling = max(1.0, 252 / self.window)
            momentum = (1 + momentum).pow(scaling) - 1
        return momentum.rename(self.output_name)


class MeanReversionZFactor(FactorFormula):
    """Rolling z-score for price relative to rolling mean."""

    description = "Mean-reversion z-score over a rolling window."
    required_columns = ("close",)

    def __init__(self, window: int = 20, price_col: str = "close", *, alias: str | None = None):
        self.window = int(window)
        if self.window <= 1:
            raise ValueError("MeanReversionZFactor 'window' must be greater than 1.")
        self.price_col = price_col
        self.min_history = self.window
        super().__init__(alias=alias, window=window, price_col=price_col)

    def default_output_name(self) -> str:
        return f"mean_reversion_z_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        price = self._column(df, self.price_col)
        mean = price.rolling(self.window).mean()
        std = price.rolling(self.window).std(ddof=0)
        zscore = (price - mean) / std.replace(0, np.nan)
        return zscore.rename(self.output_name)


class VolatilityFactor(FactorFormula):
    """Rolling realised volatility of returns."""

    description = "Rolling standard deviation of log returns (annualised)."
    required_columns = ("close",)

    def __init__(self, window: int = 20, price_col: str = "close", annualize: bool = True, *, alias: str | None = None):
        self.window = int(window)
        if self.window <= 1:
            raise ValueError("VolatilityFactor 'window' must be greater than 1.")
        self.price_col = price_col
        self.annualize = bool(annualize)
        self.min_history = self.window + 1
        super().__init__(alias=alias, window=window, price_col=price_col, annualize=annualize)

    def default_output_name(self) -> str:
        suffix = "ann" if self.annualize else "raw"
        return f"volatility_{self.window}_{suffix}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        price = self._column(df, self.price_col)
        returns = price.pct_change()
        vol = returns.rolling(self.window).std(ddof=0)
        if self.annualize:
            vol *= math.sqrt(252)
        return vol.rename(self.output_name)


class DailyReturnFactor(FactorFormula):
    """Single or multi-period return."""

    description = "Lagged percentage return over a configurable number of periods."
    required_columns = ("close",)

    def __init__(self, periods: int = 1, price_col: str = "close", *, alias: str | None = None):
        self.periods = int(periods)
        if self.periods <= 0:
            raise ValueError("DailyReturnFactor 'periods' must be positive.")
        self.price_col = price_col
        self.min_history = self.periods + 1
        super().__init__(alias=alias, periods=periods, price_col=price_col)

    def default_output_name(self) -> str:
        return f"return_{self.periods}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        price = self._column(df, self.price_col)
        ret = price.pct_change(periods=self.periods)
        return ret.rename(self.output_name)


class VolumeZScoreFactor(FactorFormula):
    """Normalized volume relative to rolling history."""

    description = "Volume z-score compared to rolling mean and std."
    required_columns = ("volume",)

    def __init__(self, window: int = 20, volume_col: str = "volume", *, alias: str | None = None):
        self.window = int(window)
        if self.window <= 1:
            raise ValueError("VolumeZScoreFactor 'window' must be greater than 1.")
        self.volume_col = volume_col
        self.min_history = self.window
        super().__init__(alias=alias, window=window, volume_col=volume_col)

    def default_output_name(self) -> str:
        return f"volume_z_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        volume = self._column(df, self.volume_col)
        mean = volume.rolling(self.window).mean()
        std = volume.rolling(self.window).std(ddof=0)
        zscore = (volume - mean) / std.replace(0, np.nan)
        return zscore.rename(self.output_name)


class FactorRegistry:
    """Registry mapping factor keys to their implementations."""

    def __init__(self):
        self._registry: Dict[str, Type[FactorFormula]] = {}

    def register(self, factor_cls: Type[FactorFormula]) -> None:
        key = factor_cls.key()
        self._registry[key] = factor_cls

    def get(self, name: str) -> Type[FactorFormula]:
        key = name.lower().replace("-", "").replace(" ", "").replace("_", "")
        if key not in self._registry:
            raise KeyError(f"Unknown factor '{name}'. Available: {sorted(self._registry)}")
        return self._registry[key]

    def create(self, spec: FactorSpec) -> FactorFormula:
        factor_cls = self.get(spec.name)
        params = dict(spec.params)
        return factor_cls(alias=spec.alias, **params)

    def available(self) -> Dict[str, Type[FactorFormula]]:
        return dict(self._registry)


GLOBAL_FACTOR_REGISTRY = FactorRegistry()


def _register_defaults() -> None:
    for factor_cls in (MomentumFactor, MeanReversionZFactor, VolatilityFactor, DailyReturnFactor, VolumeZScoreFactor):
        GLOBAL_FACTOR_REGISTRY.register(factor_cls)


_register_defaults()


DEFAULT_FACTOR_SPECS: Tuple[FactorSpec, ...] = (
    FactorSpec("momentum", params={"window": 63}),
    FactorSpec("mean_reversion_z", params={"window": 20}),
    FactorSpec("volatility", params={"window": 20}),
    FactorSpec("daily_return", params={"periods": 1}),
    FactorSpec("volume_z_score", params={"window": 20}),
)

DEFAULT_FACTOR_SET: Tuple[str, ...] = tuple(spec.name for spec in DEFAULT_FACTOR_SPECS)


def build_factor_pool(specs: Sequence[FactorSpec] | Sequence[str], registry: FactorRegistry | None = None) -> "FactorPool":
    registry = registry or GLOBAL_FACTOR_REGISTRY
    normalized_specs: List[FactorSpec] = []
    if not specs:
        specs = DEFAULT_FACTOR_SPECS
    for spec in specs:
        if isinstance(spec, FactorSpec):
            normalized_specs.append(spec)
        else:
            normalized_specs.append(FactorSpec(name=str(spec)))
    factors = [registry.create(spec) for spec in normalized_specs]
    return FactorPool(factors)


def specs_from_names(
    names: Sequence[str],
    params: Mapping[str, Mapping[str, Any]] | None = None,
) -> List[FactorSpec]:
    """Build FactorSpec objects from a list of names and optional param overrides."""
    params = params or {}
    specs: List[FactorSpec] = []
    for name in names:
        overrides = params.get(name, {})
        specs.append(FactorSpec(name=name, params=dict(overrides)))
    return specs


class FactorPool:
    """Container that evaluates a list of factor formulas."""

    def __init__(self, factors: Sequence[FactorFormula]):
        if not factors:
            raise ValueError("FactorPool requires at least one factor.")
        self.factors = list(factors)
        self._required_columns = tuple(sorted({col for factor in self.factors for col in factor.required_columns}))
        self._max_history = max(factor.min_history for factor in self.factors)

    @property
    def required_columns(self) -> Tuple[str, ...]:
        return self._required_columns

    @property
    def max_history(self) -> int:
        return self._max_history

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for factor in self.factors:
            series = factor.compute(df)
            frames.append(series)
        features = pd.concat(frames, axis=1)
        return features

    def compute_latest(self, history: pd.DataFrame) -> Dict[str, float]:
        outputs: Dict[str, float] = {}
        for factor in self.factors:
            outputs[factor.output_name] = factor.compute_latest(history)
        return outputs


class FactorRuntimeBuffer:
    """Rolling buffer that stores required columns for online factor evaluation."""

    def __init__(self, columns: Sequence[str], window: int):
        if window <= 0:
            raise ValueError("FactorRuntimeBuffer 'window' must be positive.")
        self.columns = tuple(columns)
        self.window = window
        self._buffers: Dict[str, deque] = {column: deque(maxlen=window) for column in self.columns}

    def push(self, row: Mapping[str, Any]) -> None:
        for column in self.columns:
            value = row.get(column, math.nan)
            self._buffers[column].append(value)

    def ready(self, min_history: int) -> bool:
        if not self.columns:
            return True
        return all(len(buf) >= min_history for buf in self._buffers.values())

    def to_dataframe(self) -> pd.DataFrame:
        if not self.columns:
            return pd.DataFrame()
        data = {column: list(buffer) for column, buffer in self._buffers.items()}
        return pd.DataFrame(data)

    def __len__(self) -> int:
        if not self.columns:
            return 0
        return min(len(buffer) for buffer in self._buffers.values())

