# Quantitative Trading Workflow with Backtrader

This project adds an end-to-end quantitative trading workflow on top of Backtrader. It automates data ingestion, backtesting, parameter optimisation, and paper trading using configurable modules.

## Features

- **Automated data downloads** via `yfinance` with caching to `quant_project/data/`.
- **Configurable strategy**: Moving-average crossover strategy with stop-loss/take-profit controls and position sizing.
- **Backtesting pipeline** that generates JSON performance reports in `quant_project/reports/`.
- **Parameter optimisation** sweeping strategy windows and risk controls.
- **Paper-trading loop** that periodically refreshes the latest data and re-evaluates positions.
- **Unified CLI** (`python -m quant_project`) to run each step.

## Requirements

Install dependencies (Backtrader is part of this repository):

```bash
pip install yfinance pandas matplotlib
```

If you want plots during backtests, make sure `matplotlib` is available. Optimisation can be CPU-intensive—adjust ranges to fit your machine.

## Quick Start

All commands run from the repository root (`/workspace`):

### 1. Fetch Data

```bash
python -m quant_project fetch --symbol AAPL --start 2022-01-01 --interval 1d
```

Data is cached under `quant_project/data/` and reused unless `--overwrite` is passed.

### 2. Backtest the Strategy

```bash
python -m quant_project backtest --symbol AAPL --start 2022-01-01 \
  --short-window 20 --long-window 50 --stop-loss 0.03 --take-profit 0.05
```

A JSON report is written to `quant_project/reports/` with performance metrics including Sharpe ratio, drawdowns, and trade stats.

### 3. Optimise Strategy Parameters

```bash
python -m quant_project optimize --symbol AAPL --start 2021-01-01 \
  --short-range 10 15 20 --long-range 50 75 100 --stop-range 0.02 0.03 \
  --take-range 0.04 0.05 0.06 --top-n 3
```

The optimiser iterates through valid parameter combinations (short < long) and stores the top results in the reports directory.

### 4. Run Paper Trading

```bash
python -m quant_project trade --symbol AAPL --interval 1h --loop
```

This launches a paper-trading session. It downloads the most recent data at the requested interval, runs the strategy once, logs the results, sleeps for the configured polling interval (default 60 seconds), and repeats. Omit `--loop` to execute a single evaluation.

> **Note:** Live brokerage integrations are not enabled by default. Extend `run_trade_session` in `quant_project/trading.py` with your broker’s store (e.g., IB or Oanda) as needed.

## Module Overview

- `config.py` — dataclasses describing configuration for data, strategy, backtests, optimisation, and trading.
- `data.py` — download and load data, returning Pandas DataFrames or Backtrader feeds.
- `strategy.py` — moving-average crossover Backtrader strategy with risk controls.
- `backtest.py` — sets up Cerebro, runs backtests, and emits JSON reports.
- `optimizer.py` — brute-force parameter search that reuses the backtest pipeline.
- `trading.py` — paper trading loop with extensible broker integration placeholder.
- `cli.py` — command-line interface binding the modules behind a single entry point.

## Extending for Live Trading

To connect to a live broker:

1. Expand `TradeConfig` with the credentials and parameters required by your broker.
2. Implement a new branch inside `run_trade_session` that creates the relevant Backtrader store and broker (e.g., `IBStore`, `OandaStore`).
3. Reuse `strategy_from_config` and the Cerebro setup to add your live data feed.

Because live credentials are sensitive, keep them in environment variables or an external config file.

## Project Layout

```
quant_project/
├── __init__.py
├── __main__.py
├── backtest.py
├── cli.py
├── config.py
├── data.py
├── optimizer.py
├── reports/        # auto-created for JSON outputs
├── strategy.py
├── trading.py
└── utils.py
```

## Support

For further development ideas:

- Add richer analyzers (e.g., `PyFolio`) or visual reports.
- Persist optimisation results in a database.
- Implement a unit-test suite around the strategy logic.
- Integrate a live broker when ready and test with paper credentials first.

