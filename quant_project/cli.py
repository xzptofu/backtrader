"""Command-line interface for the quantitative trading project."""

from __future__ import annotations

import argparse
from typing import Sequence

from .backtest import run_backtest
from .config import BacktestConfig, DataConfig, OptimizationConfig, StrategyConfig, TradeConfig
from .data import fetch_data
from .optimizer import run_optimization
from .trading import run_trade_session


def _parse_range(values: Sequence[str], cast):
    if not values:
        return []
    return [cast(value) for value in values]


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol to download.")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Defaults to latest.")
    parser.add_argument("--interval", default="1d", help="Data interval (e.g., 1d, 1h, 1m).")


def _build_data_config(args: argparse.Namespace) -> DataConfig:
    return DataConfig(symbol=args.symbol, start=args.start, end=args.end, interval=args.interval)


def fetch_command(args: argparse.Namespace) -> None:
    cfg = _build_data_config(args)
    path = fetch_data(cfg, overwrite=args.overwrite)
    print(f"Data saved to {path}")


def backtest_command(args: argparse.Namespace) -> None:
    data_cfg = _build_data_config(args)
    strategy_cfg = StrategyConfig(
        short_window=args.short_window,
        long_window=args.long_window,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        position_size=args.position_size,
    )
    backtest_cfg = BacktestConfig(
        initial_cash=args.cash,
        commission=args.commission,
        slippage_perc=args.slippage,
        data=data_cfg,
        strategy=strategy_cfg,
    )
    report = run_backtest(backtest_cfg, plot=args.plot)
    print(report)


def optimize_command(args: argparse.Namespace) -> None:
    data_cfg = _build_data_config(args)
    strategy_cfg = StrategyConfig(position_size=args.position_size)
    backtest_cfg = BacktestConfig(
        initial_cash=args.cash,
        commission=args.commission,
        slippage_perc=args.slippage,
        data=data_cfg,
        strategy=strategy_cfg,
    )
    opt_cfg = OptimizationConfig(
        short_window_range=_parse_range(args.short_range, int),
        long_window_range=_parse_range(args.long_range, int),
        stop_loss_range=_parse_range(args.stop_range, float),
        take_profit_range=_parse_range(args.take_range, float),
        max_concurrent=args.max_concurrent,
    )
    reports = run_optimization(backtest_cfg, opt_cfg, top_n=args.top_n)
    for idx, report in enumerate(reports, start=1):
        print(f"Rank {idx}: {report['params']} -> PnL {report['profit_loss']:.2f}")


def trade_command(args: argparse.Namespace) -> None:
    data_cfg = _build_data_config(args)
    strategy_cfg = StrategyConfig(
        short_window=args.short_window,
        long_window=args.long_window,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        position_size=args.position_size,
    )
    trade_cfg = TradeConfig(
        broker=args.broker,
        cash=args.cash,
        commission=args.commission,
        data_refresh_interval=args.interval,
        data=data_cfg,
        strategy=strategy_cfg,
        poll_interval_seconds=args.poll_interval,
    )
    report = run_trade_session(trade_cfg, loop=args.loop)
    if report:
        print(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtrader quantitative project CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Download market data.")
    _add_data_arguments(fetch_parser)
    fetch_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cached data.")
    fetch_parser.set_defaults(func=fetch_command)

    backtest_parser = subparsers.add_parser("backtest", help="Run a backtest.")
    _add_data_arguments(backtest_parser)
    backtest_parser.add_argument("--cash", type=float, default=100000)
    backtest_parser.add_argument("--commission", type=float, default=0.001)
    backtest_parser.add_argument("--slippage", type=float, default=0.0005)
    backtest_parser.add_argument("--short-window", type=int, default=20)
    backtest_parser.add_argument("--long-window", type=int, default=50)
    backtest_parser.add_argument("--stop-loss", type=float, default=0.03)
    backtest_parser.add_argument("--take-profit", type=float, default=0.05)
    backtest_parser.add_argument("--position-size", type=float, default=0.95)
    backtest_parser.add_argument("--plot", action="store_true", help="Display the Backtrader plot.")
    backtest_parser.set_defaults(func=backtest_command)

    opt_parser = subparsers.add_parser("optimize", help="Optimize strategy parameters.")
    _add_data_arguments(opt_parser)
    opt_parser.add_argument("--cash", type=float, default=100000)
    opt_parser.add_argument("--commission", type=float, default=0.001)
    opt_parser.add_argument("--slippage", type=float, default=0.0005)
    opt_parser.add_argument("--position-size", type=float, default=0.95)
    opt_parser.add_argument("--short-range", nargs="+", default=["10", "20", "30"])
    opt_parser.add_argument("--long-range", nargs="+", default=["50", "100", "150"])
    opt_parser.add_argument("--stop-range", nargs="+", default=["0.02", "0.03", "0.05"])
    opt_parser.add_argument("--take-range", nargs="+", default=["0.04", "0.06", "0.08"])
    opt_parser.add_argument("--max-concurrent", type=int, default=0, help="Number of parallel workers (0=serial).")
    opt_parser.add_argument("--top-n", type=int, default=5, help="Number of top results to keep.")
    opt_parser.set_defaults(func=optimize_command)

    trade_parser = subparsers.add_parser("trade", help="Run trading session (paper by default).")
    _add_data_arguments(trade_parser)
    trade_parser.add_argument("--broker", default="paper", help="Broker to use (paper).")
    trade_parser.add_argument("--cash", type=float, default=25000)
    trade_parser.add_argument("--commission", type=float, default=0.001)
    trade_parser.add_argument("--short-window", type=int, default=20)
    trade_parser.add_argument("--long-window", type=int, default=50)
    trade_parser.add_argument("--stop-loss", type=float, default=0.03)
    trade_parser.add_argument("--take-profit", type=float, default=0.05)
    trade_parser.add_argument("--position-size", type=float, default=0.95)
    trade_parser.add_argument("--poll-interval", type=int, default=60, help="Refresh interval in seconds when looping.")
    trade_parser.add_argument("--loop", action="store_true", help="Loop indefinitely for paper trading.")
    trade_parser.set_defaults(func=trade_command)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()

