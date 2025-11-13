"""Default Backtrader strategy definitions."""

from __future__ import annotations

import logging

import backtrader as bt

from .config import StrategyConfig

logger = logging.getLogger(__name__)
DEFAULT_CFG = StrategyConfig()


class MovingAverageCrossStrategy(bt.Strategy):
    """Simple moving-average crossover strategy with basic risk management."""

    params = (
        ("short_window", DEFAULT_CFG.short_window),
        ("long_window", DEFAULT_CFG.long_window),
        ("stop_loss", DEFAULT_CFG.stop_loss),
        ("take_profit", DEFAULT_CFG.take_profit),
        ("position_size", DEFAULT_CFG.position_size),
    )

    def __init__(self):
        if self.p.short_window >= self.p.long_window:
            raise ValueError("short_window must be smaller than long_window for crossover strategies.")

        self.short_ma = bt.indicators.SMA(self.datas[0], period=self.p.short_window)
        self.long_ma = bt.indicators.SMA(self.datas[0], period=self.p.long_window)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)

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

    def _get_position_size(self, price: float) -> float:
        """Calculate position size based on available cash."""
        cash = self.broker.get_cash()
        target_cash = cash * self.p.position_size
        size = target_cash / price
        return max(0, size)


def strategy_from_config(cfg: StrategyConfig) -> tuple[type[bt.Strategy], dict]:
    """Return the strategy class and parameter dictionary for Cerebro."""
    params = dict(
        short_window=cfg.short_window,
        long_window=cfg.long_window,
        stop_loss=cfg.stop_loss,
        take_profit=cfg.take_profit,
        position_size=cfg.position_size,
    )
    return MovingAverageCrossStrategy, params

