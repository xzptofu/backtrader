"""Configuration objects for the Backtrader quantitative project."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
REPORTS_DIR = PACKAGE_ROOT / "reports"


def _to_datetime(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


@dataclass(slots=True)
class DataConfig:
    """Configuration for fetching market data."""

    symbol: str = "AAPL"
    start: datetime | str = "2020-01-01"
    end: Optional[datetime | str] = None
    interval: str = "1d"
    source: str = "yfinance"
    cache: bool = True
    auto_adjust: bool = True
    data_dir: Path = DATA_DIR

    def resolved_start(self) -> datetime:
        return _to_datetime(self.start)  # type: ignore[arg-type]

    def resolved_end(self) -> Optional[datetime]:
        return _to_datetime(self.end)  # type: ignore[arg-type]

    def path(self) -> Path:
        suffix = self.interval.replace(" ", "_")
        start = self.resolved_start().strftime("%Y%m%d")
        end = self.resolved_end().strftime("%Y%m%d") if self.resolved_end() else "latest"
        filename = f"{self.symbol}-{suffix}-{start}-{end}.csv"
        return self.data_dir / filename


@dataclass(slots=True)
class StrategyConfig:
    """Configuration for supported trading strategies."""

    name: str = "moving_average_cross"
    short_window: int = 20
    long_window: int = 50
    stop_loss: float = 0.03  # 3%
    take_profit: float = 0.05  # 5%
    position_size: float = 0.95  # invest 95% of available cash
    momentum_window: int = 63
    mean_reversion_window: int = 20
    volatility_window: int = 20
    momentum_weight: float = 0.6
    mean_reversion_weight: float = 0.3
    volatility_weight: float = 0.1
    signal_threshold: float = 0.05
    rebalance_interval: int = 5
    volatility_target: Optional[float] = None
    mean_reversion_entry_z: float = 1.0
    mean_reversion_exit_z: float = 0.25
    allow_short: bool = False
    ml_model_path: Optional[str] = None
    ml_positive_threshold: float = 0.55
    ml_negative_threshold: float = 0.45


@dataclass(slots=True)
class BacktestConfig:
    """Configuration controlling the backtest setup."""

    initial_cash: float = 100_000
    commission: float = 0.001
    slippage_perc: float = 0.0005
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    report_dir: Path = REPORTS_DIR


@dataclass(slots=True)
class OptimizationConfig:
    """Configuration for strategy parameter optimization."""

    short_window_range: Sequence[int] = (10, 20, 30)
    long_window_range: Sequence[int] = (50, 100, 150)
    stop_loss_range: Sequence[float] = (0.02, 0.03, 0.05)
    take_profit_range: Sequence[float] = (0.04, 0.06, 0.08)
    max_concurrent: int = 0  # 0 == use serial execution


@dataclass(slots=True)
class TradeConfig:
    """Configuration for paper/live trading."""

    broker: str = "paper"  # supported: "paper", "ccxt"
    cash: float = 25_000
    commission: float = 0.001
    data_refresh_interval: str = "1m"
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    exchange: str = "binance"
    ccxt_symbol: Optional[str] = None
    timeframe: str = "minute"
    compression: int = 1
    store_kwargs: dict = field(default_factory=dict)
    ohlcv_limit: int = 200
    poll_interval_seconds: int = 60


