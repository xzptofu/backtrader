"""Trading execution interfaces."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Dict, Optional

import backtrader as bt

from .backtest import build_report, create_cerebro
from .config import BacktestConfig, TradeConfig
from .data import fetch_data, load_dataframe
from .strategy import strategy_from_config
from .utils import configure_logging

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """Run a looping paper-trading session by periodically refreshing the data."""

    def __init__(self, cfg: TradeConfig):
        self.cfg = cfg

    def run_once(self) -> Dict:
        data_cfg = replace(self.cfg.data, interval=self.cfg.data_refresh_interval, end=None, cache=False)
        data_path = fetch_data(data_cfg, overwrite=True)
        df = load_dataframe(data_path)

        bt_cfg = BacktestConfig(
            initial_cash=self.cfg.cash,
            commission=self.cfg.commission,
            slippage_perc=0.0,
            data=data_cfg,
            strategy=self.cfg.strategy,
        )

        cerebro = create_cerebro(bt_cfg)
        data_feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data_feed)

        strategy_cls, params = strategy_from_config(self.cfg.strategy)
        cerebro.addstrategy(strategy_cls, **params)

        strategies = cerebro.run()
        report = build_report(strategies[0], bt_cfg, cerebro)
        logger.info(
            "Paper trading run finished. Final value: %.2f (PnL: %.2f)",
            report["final_value"],
            report["profit_loss"],
        )
        return report

    def run(self, loop: bool = False) -> Dict:
        if not loop:
            return self.run_once()

        report: Optional[Dict] = None
        while True:
            report = self.run_once()
            logger.info("Sleeping for %s seconds before next refresh", self.cfg.poll_interval_seconds)
            time.sleep(self.cfg.poll_interval_seconds)
        return report or {}


def run_trade_session(cfg: Optional[TradeConfig] = None, *, loop: bool = False) -> Dict:
    """Run a trade session (paper by default)."""
    configure_logging()
    cfg = cfg or TradeConfig()

    if cfg.broker == "paper":
        engine = PaperTradingEngine(cfg)
        return engine.run(loop=loop)

    raise NotImplementedError(
        f"Broker {cfg.broker!r} is not implemented yet. Extend run_trade_session with your broker integration."
    )


def main() -> None:
    run_trade_session()


if __name__ == "__main__":  # pragma: no cover
    main()

