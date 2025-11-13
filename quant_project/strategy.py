"""Backtrader strategy definitions used by the quant project."""

from __future__ import annotations

import logging
import math
import statistics
from typing import Dict, Tuple

import backtrader as bt

from .config import StrategyConfig

logger = logging.getLogger(__name__)
DEFAULT_CFG = StrategyConfig()


class BaseStrategy(bt.Strategy):
    """Base strategy implementing logging helpers shared across strategies."""

    def __init__(self):
        super().__init__()
        self.order = None
        self.entry_price = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            action = "BUY" if order.isbuy() else "SELL"
            self._log(
                "%s EXECUTED, Price: %.2f, Size: %.2f, Cost: %.2f, Comm %.2f",
                action,
                order.executed.price,
                order.executed.size,
                order.executed.value,
                order.executed.comm,
            )
            if order.isbuy():
                self.entry_price = order.executed.price
            else:
                self.entry_price = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self._log("Order %s", order.getstatusname())

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self._log(
            "OPERATION PROFIT, GROSS %.2f, NET %.2f, TOTAL COMM %.2f",
            trade.pnl,
            trade.pnlcomm,
            trade.commission,
        )

    def _log(self, txt, *args):
        dt = self.datas[0].datetime.datetime(0)
        timestamp = dt.isoformat() if dt else "NA"
        message = txt % args if args else txt
        logger.info("%s - %s", timestamp, message)

    def _get_position_size(self, price: float, fraction: float | None = None) -> float:
        """Calculate position size based on available cash."""
        cash = self.broker.get_cash()
        pct = fraction if fraction is not None else getattr(self.p, "position_size", 1.0)
        target_cash = cash * pct
        if price <= 0:
            return 0.0
        return max(0.0, target_cash / price)

    def _current_exposure(self, price: float) -> float:
        """Return current exposure as a percent of portfolio value."""
        portfolio_value = self.broker.getvalue()
        if not portfolio_value or price <= 0:
            return 0.0
        position_value = self.position.size * price
        return position_value / portfolio_value


class MovingAverageCrossStrategy(BaseStrategy):
    """Simple moving-average crossover strategy with basic risk management."""

    params = (
        ("short_window", DEFAULT_CFG.short_window),
        ("long_window", DEFAULT_CFG.long_window),
        ("stop_loss", DEFAULT_CFG.stop_loss),
        ("take_profit", DEFAULT_CFG.take_profit),
        ("position_size", DEFAULT_CFG.position_size),
    )

    def __init__(self):
        super().__init__()
        if self.p.short_window >= self.p.long_window:
            raise ValueError("short_window must be smaller than long_window for crossover strategies.")

        self.short_ma = bt.indicators.SMA(self.datas[0], period=self.p.short_window)
        self.long_ma = bt.indicators.SMA(self.datas[0], period=self.p.long_window)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)

    def next(self):
        if self.order:
            return

        price = self.data.close[0]

        if not self.position:
            if self.crossover > 0:
                size = self._get_position_size(price)
                if size > 0:
                    self._log("BUY CREATE, %.2f (size=%.2f)", price, size)
                    self.order = self.buy(size=size)
        else:
            stop_price = self.entry_price * (1 - self.p.stop_loss) if self.entry_price else None
            take_price = self.entry_price * (1 + self.p.take_profit) if self.entry_price else None

            if stop_price and price <= stop_price:
                self._log("STOP LOSS HIT, %.2f <= %.2f", price, stop_price)
                self.order = self.close()
            elif take_price and price >= take_price:
                self._log("TAKE PROFIT HIT, %.2f >= %.2f", price, take_price)
                self.order = self.close()
            elif self.crossover < 0:
                self._log("SELL CREATE (crossover), %.2f", price)
                self.order = self.close()


class MeanReversionStrategy(BaseStrategy):
    """Z-score mean reversion strategy with optional shorting."""

    params = (
        ("lookback", DEFAULT_CFG.mean_reversion_window),
        ("entry_z", DEFAULT_CFG.mean_reversion_entry_z),
        ("exit_z", DEFAULT_CFG.mean_reversion_exit_z),
        ("position_size", DEFAULT_CFG.position_size),
        ("allow_short", DEFAULT_CFG.allow_short),
    )

    def __init__(self):
        super().__init__()
        if self.p.lookback <= 1:
            raise ValueError("lookback window must be greater than 1.")
        if self.p.exit_z < 0:
            raise ValueError("exit_z must be non-negative.")
        if self.p.entry_z <= 0:
            raise ValueError("entry_z must be positive.")

        self.sma = bt.indicators.SMA(self.datas[0], period=self.p.lookback)
        self.std = bt.indicators.StdDev(self.datas[0], period=self.p.lookback)

    def next(self):
        if self.order:
            return

        price = self.data.close[0]
        std = self.std[0]
        if std is None or std == 0:
            return

        mean = self.sma[0]
        zscore = (price - mean) / std

        if not self.position:
            if zscore <= -self.p.entry_z:
                size = self._get_position_size(price)
                if size > 0:
                    self._log("MEAN REVERSION LONG, price=%.2f z=%.2f size=%.2f", price, zscore, size)
                    self.order = self.buy(size=size)
            elif self.p.allow_short and zscore >= self.p.entry_z:
                size = self._get_position_size(price)
                if size > 0:
                    self._log("MEAN REVERSION SHORT, price=%.2f z=%.2f size=%.2f", price, zscore, size)
                    self.order = self.sell(size=size)
        else:
            if self.position.size > 0:
                if zscore >= -self.p.exit_z:
                    self._log("MEAN REVERSION EXIT LONG, price=%.2f z=%.2f", price, zscore)
                    self.order = self.close()
            else:
                if zscore <= self.p.exit_z:
                    self._log("MEAN REVERSION EXIT SHORT, price=%.2f z=%.2f", price, zscore)
                    self.order = self.close()


class MultiFactorAlphaStrategy(BaseStrategy):
    """Multi-factor alpha strategy combining momentum, mean reversion, and volatility signals."""

    params = (
        ("momentum_window", DEFAULT_CFG.momentum_window),
        ("mean_reversion_window", DEFAULT_CFG.mean_reversion_window),
        ("volatility_window", DEFAULT_CFG.volatility_window),
        ("momentum_weight", DEFAULT_CFG.momentum_weight),
        ("mean_reversion_weight", DEFAULT_CFG.mean_reversion_weight),
        ("volatility_weight", DEFAULT_CFG.volatility_weight),
        ("signal_threshold", DEFAULT_CFG.signal_threshold),
        ("rebalance_interval", DEFAULT_CFG.rebalance_interval),
        ("position_size", DEFAULT_CFG.position_size),
        ("volatility_target", DEFAULT_CFG.volatility_target),
        ("allow_short", DEFAULT_CFG.allow_short),
    )

    def __init__(self):
        super().__init__()
        if self.p.momentum_window <= 1:
            raise ValueError("momentum_window must be greater than 1.")
        if self.p.mean_reversion_window <= 1:
            raise ValueError("mean_reversion_window must be greater than 1.")
        if self.p.volatility_window <= 1:
            raise ValueError("volatility_window must be greater than 1.")
        if self.p.rebalance_interval <= 0:
            raise ValueError("rebalance_interval must be positive.")

        self.mean_sma = bt.indicators.SMA(self.datas[0], period=self.p.mean_reversion_window)
        self.mean_std = bt.indicators.StdDev(self.datas[0], period=self.p.mean_reversion_window)
        self.last_rebalance_bar = -self.p.rebalance_interval

    def _normalized_weights(self) -> Dict[str, float]:
        weights = {
            "momentum": self.p.momentum_weight,
            "mean_reversion": self.p.mean_reversion_weight,
            "volatility": self.p.volatility_weight,
        }
        total = sum(abs(value) for value in weights.values())
        if total == 0:
            return {key: 0.0 for key in weights}
        return {key: value / total for key, value in weights.items()}

    def _volatility(self) -> float:
        returns = []
        for i in range(1, self.p.volatility_window + 1):
            try:
                prev_price = self.data.close[-i - 1]
                curr_price = self.data.close[-i]
            except IndexError:
                return 0.0
            if prev_price:
                returns.append((curr_price / prev_price) - 1)
        if not returns:
            return 0.0
        if len(returns) == 1:
            return abs(returns[0])
        return statistics.pstdev(returns)

    def next(self):
        if self.order:
            return

        price = self.data.close[0]
        try:
            past_price = self.data.close[-self.p.momentum_window]
        except IndexError:
            return

        std = self.mean_std[0]
        if std is None or std == 0:
            return

        momentum = (price / past_price - 1) if past_price else 0.0
        mean = self.mean_sma[0]
        mean_reversion = 0.0 if mean is None else -(price - mean) / std
        volatility = self._volatility()

        weights = self._normalized_weights()
        composite = (
            weights["momentum"] * momentum
            + weights["mean_reversion"] * mean_reversion
            + weights["volatility"] * (-volatility)
        )

        if abs(composite) < self.p.signal_threshold:
            target_percent = 0.0
        else:
            target_percent = math.tanh(composite) * self.p.position_size

        if not self.p.allow_short and target_percent < 0:
            target_percent = 0.0

        if self.p.volatility_target and volatility > 0:
            scale = self.p.volatility_target / volatility
            target_percent *= scale

        target_percent = max(-self.p.position_size, min(self.p.position_size, target_percent))

        current_exposure = self._current_exposure(price)
        exposure_gap = abs(target_percent - current_exposure)

        bars_since = len(self) - self.last_rebalance_bar
        should_rebalance = (
            bars_since >= self.p.rebalance_interval
            or (self.position and target_percent == 0.0)
            or exposure_gap >= 0.05
        )

        if not should_rebalance:
            return

        if self.order:
            self.cancel(self.order)

        self._log(
            "ALPHA REBALANCE -> target=%.2f%% (comp=%.4f, mom=%.4f, mean=%.4f, vol=%.4f)",
            target_percent * 100,
            composite,
            momentum,
            mean_reversion,
            volatility,
        )
        self.order = self.order_target_percent(target=target_percent)
        self.last_rebalance_bar = len(self)


def _normalise_name(name: str) -> str:
    return name.replace("-", "_").lower()


def strategy_from_config(cfg: StrategyConfig) -> Tuple[type[bt.Strategy], dict]:
    """Return the strategy class and parameter dictionary for Cerebro."""
    name = _normalise_name(cfg.name or "moving_average_cross")

    if name in {"moving_average_cross", "moving_average", "ma_cross"}:
        params = dict(
            short_window=cfg.short_window,
            long_window=cfg.long_window,
            stop_loss=cfg.stop_loss,
            take_profit=cfg.take_profit,
            position_size=cfg.position_size,
        )
        return MovingAverageCrossStrategy, params

    if name in {"mean_reversion", "mean_reversion_zscore", "zscore"}:
        params = dict(
            lookback=cfg.mean_reversion_window,
            entry_z=cfg.mean_reversion_entry_z,
            exit_z=cfg.mean_reversion_exit_z,
            position_size=cfg.position_size,
            allow_short=cfg.allow_short,
        )
        return MeanReversionStrategy, params

    if name in {"multi_factor_alpha", "multifactor_alpha", "multi_factor"}:
        params = dict(
            momentum_window=cfg.momentum_window,
            mean_reversion_window=cfg.mean_reversion_window,
            volatility_window=cfg.volatility_window,
            momentum_weight=cfg.momentum_weight,
            mean_reversion_weight=cfg.mean_reversion_weight,
            volatility_weight=cfg.volatility_weight,
            signal_threshold=cfg.signal_threshold,
            rebalance_interval=cfg.rebalance_interval,
            position_size=cfg.position_size,
            volatility_target=cfg.volatility_target,
            allow_short=cfg.allow_short,
        )
        return MultiFactorAlphaStrategy, params

    raise ValueError(f"Unsupported strategy '{cfg.name}'.")

