"""Backtesting workflow built on Backtrader."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import backtrader as bt

from .config import BacktestConfig
from .data import fetch_data, load_dataframe
from .strategy import strategy_from_config
from .utils import configure_logging, ensure_reports_dir, timestamped_filename

logger = logging.getLogger(__name__)


def create_cerebro(cfg: BacktestConfig) -> bt.Cerebro:
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cfg.initial_cash)
    cerebro.broker.setcommission(commission=cfg.commission)
    cerebro.broker.set_slippage_perc(cfg.slippage_perc)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Trades)
    return cerebro


def build_report(strategy: bt.Strategy, cfg: BacktestConfig, cerebro: bt.Cerebro) -> Dict[str, Any]:
    analyzers = strategy.analyzers
    sharpe = analyzers.sharpe.get_analysis().get("sharperatio")
    drawdown = analyzers.drawdown.get_analysis()
    returns = analyzers.returns.get_analysis()
    trades = analyzers.trades.get_analysis()

    initial_cash = cfg.initial_cash
    final_value = cerebro.broker.getvalue()
    pnl = final_value - initial_cash

    report = {
        "symbol": cfg.data.symbol,
        "interval": cfg.data.interval,
        "initial_cash": initial_cash,
        "final_value": final_value,
        "profit_loss": pnl,
        "profit_loss_pct": pnl / initial_cash if initial_cash else None,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown.get("max", {}).get("moneydown"),
        "max_drawdown_pct": drawdown.get("max", {}).get("drawdown"),
        "max_drawdown_len": drawdown.get("max", {}).get("len"),
        "return_cumulative": returns.get("rtot"),
        "return_average": returns.get("ravg"),
        "return_compounded": returns.get("rnorm"),
        "trades_total": trades.get("total", {}).get("total"),
        "trades_won": trades.get("won", {}).get("total"),
        "trades_lost": trades.get("lost", {}).get("total"),
    }

    # Calculate win rate if possible
    won = report["trades_won"] or 0
    total = report["trades_total"] or 0
    report["win_rate"] = (won / total) if total else None
    return report


def run_backtest(cfg: Optional[BacktestConfig] = None, *, plot: bool = False, data_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run the backtest and return a report dictionary."""
    configure_logging()

    cfg = cfg or BacktestConfig()
    ensure_reports_dir(cfg.report_dir)

    if data_path is None:
        data_path = fetch_data(cfg.data)
    df = load_dataframe(data_path)

    cerebro = create_cerebro(cfg)
    data_feed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data_feed)

    strategy_cls, params = strategy_from_config(cfg.strategy)
    cerebro.addstrategy(strategy_cls, **params)

    logger.info("Starting portfolio value: %.2f", cerebro.broker.getvalue())
    results = cerebro.run()
    logger.info("Final portfolio value: %.2f", cerebro.broker.getvalue())

    strategy = results[0]
    report = build_report(strategy, cfg, cerebro)

    report_path = timestamped_filename(f"backtest-{cfg.data.symbol}", directory=cfg.report_dir)
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    logger.info("Backtest report saved to %s", report_path)

    if plot:
        cerebro.plot(style="candlestick")

    return report


def main() -> None:
    run_backtest()


if __name__ == "__main__":  # pragma: no cover
    main()

