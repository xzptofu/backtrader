"""Command-line interface for the quantitative trading project."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Sequence

from .backtest import run_backtest
from .config import BacktestConfig, DataConfig, OptimizationConfig, StrategyConfig, TradeConfig
from .data import fetch_data
from .factor_pipeline import build_factor_dataset
from .factors import DEFAULT_FACTOR_SET, FactorSpec, GLOBAL_FACTOR_REGISTRY
from .optimizer import run_optimization
from .trading import run_trade_session


STRATEGY_CHOICES = ("moving_average_cross", "mean_reversion", "multi_factor_alpha", "machine_learning_alpha")


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


def _build_strategy_config(args: argparse.Namespace) -> StrategyConfig:
    defaults = StrategyConfig()
    return StrategyConfig(
        name=getattr(args, "strategy", defaults.name),
        short_window=getattr(args, "short_window", defaults.short_window),
        long_window=getattr(args, "long_window", defaults.long_window),
        stop_loss=getattr(args, "stop_loss", defaults.stop_loss),
        take_profit=getattr(args, "take_profit", defaults.take_profit),
        position_size=getattr(args, "position_size", defaults.position_size),
        momentum_window=getattr(args, "momentum_window", defaults.momentum_window),
        mean_reversion_window=getattr(args, "mean_reversion_window", defaults.mean_reversion_window),
        volatility_window=getattr(args, "volatility_window", defaults.volatility_window),
        momentum_weight=getattr(args, "momentum_weight", defaults.momentum_weight),
        mean_reversion_weight=getattr(args, "mean_reversion_weight", defaults.mean_reversion_weight),
        volatility_weight=getattr(args, "volatility_weight", defaults.volatility_weight),
        signal_threshold=getattr(args, "signal_threshold", defaults.signal_threshold),
        rebalance_interval=getattr(args, "rebalance_interval", defaults.rebalance_interval),
        volatility_target=getattr(args, "volatility_target", defaults.volatility_target),
        mean_reversion_entry_z=getattr(args, "mr_entry_z", defaults.mean_reversion_entry_z),
        mean_reversion_exit_z=getattr(args, "mr_exit_z", defaults.mean_reversion_exit_z),
        allow_short=getattr(args, "allow_short", defaults.allow_short),
        ml_model_path=getattr(args, "model_path", defaults.ml_model_path),
        ml_positive_threshold=getattr(args, "ml_positive_threshold", defaults.ml_positive_threshold),
        ml_negative_threshold=getattr(args, "ml_negative_threshold", defaults.ml_negative_threshold),
    )


def _normalize_factor_key(name: str) -> str:
    return name.replace("-", "").replace("_", "").replace(" ", "").lower()


def _coerce_cli_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _parse_factor_param_overrides(values: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    overrides: Dict[str, Dict[str, Any]] = {}
    for raw in values:
        if "=" not in raw or "." not in raw:
            raise ValueError(f"Invalid --factor-param format '{raw}'. Expected 'factor.param=value'.")
        target, raw_value = raw.split("=", 1)
        identifier, param_name = target.split(".", 1)
        key = _normalize_factor_key(identifier.strip())
        overrides.setdefault(key, {})[param_name.strip()] = _coerce_cli_value(raw_value.strip())
    return overrides


def _build_factor_specs_from_args(factors: Sequence[str], param_args: Sequence[str]) -> Sequence[FactorSpec]:
    overrides = _parse_factor_param_overrides(param_args)
    requested = factors or DEFAULT_FACTOR_SET

    specs: list[FactorSpec] = []
    for raw in requested:
        if ":" in raw:
            name, alias = raw.split(":", 1)
        else:
            name, alias = raw, None
        clean_name = name.strip()
        alias_name = alias.strip() if alias else None

        params = dict(overrides.get(_normalize_factor_key(clean_name), {}))
        if alias_name:
            params.update(overrides.get(_normalize_factor_key(alias_name), {}))

        specs.append(FactorSpec(name=clean_name, alias=alias_name, params=params))
    return specs


def fetch_command(args: argparse.Namespace) -> None:
    cfg = _build_data_config(args)
    path = fetch_data(cfg, overwrite=args.overwrite)
    print(f"Data saved to {path}")


def backtest_command(args: argparse.Namespace) -> None:
    data_cfg = _build_data_config(args)
    strategy_cfg = _build_strategy_config(args)
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
    if args.strategy != "moving_average_cross":
        raise ValueError("Optimization currently supports the moving_average_cross strategy.")
    data_cfg = _build_data_config(args)
    strategy_cfg = _build_strategy_config(args)
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
    strategy_cfg = _build_strategy_config(args)
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


def factors_list_command(_: argparse.Namespace) -> None:
    registry = GLOBAL_FACTOR_REGISTRY.available()
    print("Available factors:")
    for name in sorted(registry):
        desc = registry[name].description or ""
        print(f"- {name}: {desc}")


def factors_build_command(args: argparse.Namespace) -> None:
    data_cfg = _build_data_config(args)
    specs = _build_factor_specs_from_args(args.factor or [], args.factor_param or [])
    horizons = args.forward or []
    dataset, output_path = build_factor_dataset(
        data_cfg,
        specs,
        dropna=not args.keep_na,
        forward_horizons=horizons,
        output_path=args.output,
    )
    print(f"Generated factor dataset with shape {dataset.shape}.")
    if output_path:
        print(f"Saved dataset to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtrader quantitative project CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Download market data.")
    _add_data_arguments(fetch_parser)
    fetch_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cached data.")
    fetch_parser.set_defaults(func=fetch_command)

    backtest_parser = subparsers.add_parser("backtest", help="Run a backtest.")
    _add_data_arguments(backtest_parser)
    backtest_parser.add_argument(
        "--strategy",
        choices=STRATEGY_CHOICES,
        default="moving_average_cross",
        help="Strategy to execute (moving_average_cross, mean_reversion, multi_factor_alpha).",
    )
    backtest_parser.add_argument("--cash", type=float, default=100000)
    backtest_parser.add_argument("--commission", type=float, default=0.001)
    backtest_parser.add_argument("--slippage", type=float, default=0.0005)
    backtest_parser.add_argument("--short-window", type=int, default=20)
    backtest_parser.add_argument("--long-window", type=int, default=50)
    backtest_parser.add_argument("--stop-loss", type=float, default=0.03)
    backtest_parser.add_argument("--take-profit", type=float, default=0.05)
    backtest_parser.add_argument("--position-size", type=float, default=0.95)
    backtest_parser.add_argument("--momentum-window", type=int, default=63, help="Lookback for the momentum factor.")
    backtest_parser.add_argument(
        "--mean-reversion-window",
        type=int,
        default=20,
        help="Lookback window for mean reversion calculations.",
    )
    backtest_parser.add_argument("--volatility-window", type=int, default=20, help="Lookback for volatility estimation.")
    backtest_parser.add_argument("--momentum-weight", type=float, default=0.6, help="Weight for the momentum factor.")
    backtest_parser.add_argument(
        "--mean-reversion-weight",
        type=float,
        default=0.3,
        help="Weight for the mean reversion factor.",
    )
    backtest_parser.add_argument("--volatility-weight", type=float, default=0.1, help="Weight for the volatility factor.")
    backtest_parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.05,
        help="Absolute composite score required before entering a position.",
    )
    backtest_parser.add_argument(
        "--rebalance-interval",
        type=int,
        default=5,
        help="Bars between rebalances for the multi-factor strategy.",
    )
    backtest_parser.add_argument(
        "--volatility-target",
        type=float,
        default=None,
        help="Optional volatility target; omit for no targeting.",
    )
    backtest_parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a trained LightGBM model file (required for machine_learning_alpha).",
    )
    backtest_parser.add_argument(
        "--ml-positive-threshold",
        type=float,
        default=0.55,
        help="Probability threshold to trigger long positions for ML strategies.",
    )
    backtest_parser.add_argument(
        "--ml-negative-threshold",
        type=float,
        default=0.45,
        help="Probability threshold to trigger short positions for ML strategies.",
    )
    backtest_parser.add_argument(
        "--mr-entry-z",
        type=float,
        default=1.0,
        help="Z-score entry threshold for the mean reversion strategy.",
    )
    backtest_parser.add_argument(
        "--mr-exit-z",
        type=float,
        default=0.25,
        help="Z-score exit threshold for the mean reversion strategy.",
    )
    backtest_parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Allow strategies to take short exposure where supported.",
    )
    backtest_parser.add_argument("--plot", action="store_true", help="Display the Backtrader plot.")
    backtest_parser.set_defaults(func=backtest_command)

    opt_parser = subparsers.add_parser("optimize", help="Optimize strategy parameters.")
    _add_data_arguments(opt_parser)
    opt_parser.add_argument(
        "--strategy",
        choices=STRATEGY_CHOICES,
        default="moving_average_cross",
        help="Strategy to optimize (currently only moving_average_cross supported).",
    )
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
    trade_parser.add_argument(
        "--strategy",
        choices=STRATEGY_CHOICES,
        default="moving_average_cross",
        help="Strategy to execute during trading.",
    )
    trade_parser.add_argument("--broker", default="paper", help="Broker to use (paper).")
    trade_parser.add_argument("--cash", type=float, default=25000)
    trade_parser.add_argument("--commission", type=float, default=0.001)
    trade_parser.add_argument("--short-window", type=int, default=20)
    trade_parser.add_argument("--long-window", type=int, default=50)
    trade_parser.add_argument("--stop-loss", type=float, default=0.03)
    trade_parser.add_argument("--take-profit", type=float, default=0.05)
    trade_parser.add_argument("--position-size", type=float, default=0.95)
    trade_parser.add_argument("--momentum-window", type=int, default=63)
    trade_parser.add_argument("--mean-reversion-window", type=int, default=20)
    trade_parser.add_argument("--volatility-window", type=int, default=20)
    trade_parser.add_argument("--momentum-weight", type=float, default=0.6)
    trade_parser.add_argument("--mean-reversion-weight", type=float, default=0.3)
    trade_parser.add_argument("--volatility-weight", type=float, default=0.1)
    trade_parser.add_argument("--signal-threshold", type=float, default=0.05)
    trade_parser.add_argument("--rebalance-interval", type=int, default=5)
    trade_parser.add_argument("--volatility-target", type=float, default=None)
    trade_parser.add_argument("--model-path", default=None)
    trade_parser.add_argument("--ml-positive-threshold", type=float, default=0.55)
    trade_parser.add_argument("--ml-negative-threshold", type=float, default=0.45)
    trade_parser.add_argument("--mr-entry-z", type=float, default=1.0)
    trade_parser.add_argument("--mr-exit-z", type=float, default=0.25)
    trade_parser.add_argument("--allow-short", action="store_true")
    trade_parser.add_argument("--poll-interval", type=int, default=60, help="Refresh interval in seconds when looping.")
    trade_parser.add_argument("--loop", action="store_true", help="Loop indefinitely for paper trading.")
    trade_parser.set_defaults(func=trade_command)

    factors_parser = subparsers.add_parser("factors", help="Factor pool utilities.")
    factor_subparsers = factors_parser.add_subparsers(dest="factor_command", required=True)

    factors_list_parser = factor_subparsers.add_parser("list", help="List the registered factors.")
    factors_list_parser.set_defaults(func=factors_list_command)

    factors_build_parser = factor_subparsers.add_parser("build", help="Generate factor values for training.")
    _add_data_arguments(factors_build_parser)
    factors_build_parser.add_argument(
        "--factor",
        action="append",
        default=[],
        help="Factor to include (optionally 'name:alias' for duplicates). Defaults to built-in set.",
    )
    factors_build_parser.add_argument(
        "--factor-param",
        action="append",
        default=[],
        help="Override factor parameters via 'identifier.param=value'. Identifiers can be names or aliases.",
    )
    factors_build_parser.add_argument(
        "--forward",
        nargs="*",
        type=int,
        default=[1, 5, 20],
        help="Forward return horizons to append as targets.",
    )
    factors_build_parser.add_argument(
        "--keep-na",
        action="store_true",
        help="Keep rows with NaNs (otherwise rows with missing factors are dropped).",
    )
    factors_build_parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path to persist the factor dataset.",
    )
    factors_build_parser.set_defaults(func=factors_build_command)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()

