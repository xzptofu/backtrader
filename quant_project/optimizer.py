"""Strategy parameter optimization utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional

import backtrader as bt

from .backtest import build_report, create_cerebro
from .config import BacktestConfig, OptimizationConfig
from .data import fetch_data, load_dataframe
from .strategy import MovingAverageCrossStrategy
from .utils import configure_logging, timestamped_filename

logger = logging.getLogger(__name__)


def _valid_parameter_sets(opt_cfg: OptimizationConfig) -> Iterable[Dict[str, Any]]:
    for short_window in opt_cfg.short_window_range:
        for long_window in opt_cfg.long_window_range:
            if short_window >= long_window:
                continue
            for stop_loss in opt_cfg.stop_loss_range:
                for take_profit in opt_cfg.take_profit_range:
                    yield {
                        "short_window": short_window,
                        "long_window": long_window,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                    }


def run_optimization(
    backtest_cfg: Optional[BacktestConfig] = None,
    opt_cfg: Optional[OptimizationConfig] = None,
    *,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Run an optimization sweep and return the top parameter combinations."""
    configure_logging()

    backtest_cfg = backtest_cfg or BacktestConfig()
    opt_cfg = opt_cfg or OptimizationConfig()

    # Backtrader's optimization interface doesn't allow easily filtering invalid parameter combinations,
    # so we perform a manual loop that reuses the backtesting routine per parameter set.
    reports: List[Dict[str, Any]] = []
    data_path = fetch_data(backtest_cfg.data)
    df = load_dataframe(data_path)

    for params in _valid_parameter_sets(opt_cfg):
        strategy_cfg = replace(backtest_cfg.strategy, **params)
        cfg_copy = BacktestConfig(
            initial_cash=backtest_cfg.initial_cash,
            commission=backtest_cfg.commission,
            slippage_perc=backtest_cfg.slippage_perc,
            data=backtest_cfg.data,
            strategy=strategy_cfg,
            report_dir=backtest_cfg.report_dir,
        )

        cerebro = create_cerebro(cfg_copy)
        data_feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data_feed)
        cerebro.addstrategy(MovingAverageCrossStrategy, **params, position_size=cfg_copy.strategy.position_size)

        logger.info(
            "Optimization run for params: short=%s long=%s stop=%.3f take=%.3f",
            params["short_window"],
            params["long_window"],
            params["stop_loss"],
            params["take_profit"],
        )

        strategies = cerebro.run()
        strategy = strategies[0]
        report = build_report(strategy, cfg_copy, cerebro)
        report["params"] = params
        reports.append(report)

    reports.sort(key=lambda x: x.get("profit_loss", 0), reverse=True)
    top_reports = reports[:top_n] if top_n else reports

    report_path = timestamped_filename(f"optimization-{backtest_cfg.data.symbol}")
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(top_reports, fp, indent=2, ensure_ascii=False)

    logger.info("Optimization results saved to %s", report_path)
    return top_reports


def main() -> None:
    run_optimization()


if __name__ == "__main__":  # pragma: no cover
    main()

