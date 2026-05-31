"""OpenBB-driven backtesting harness.

The backtest:
  1) Pulls SPX (or single-name) option chain snapshots from OpenBB for each
     trading day in the requested window.
  2) Filters quotes for liquidity (OI, tight spread).
  3) Recalibrates Heston and rBergomi to each day's surface (warm-started
     from the previous day's parameters where available).
  4) Tracks per-day RMSE, parameter drift, ATM-skew fit, and the model-risk
     spread on a held-out portfolio of vanillas.

Layout
------
- data_loader.py    : OpenBB pulls + filtering + on-disk caching.
- backtest_engine.py: rolling-window recalibration + result aggregation.
- run_backtest.py   : CLI runner. `python -m volengine.backtesting.run_backtest`.
"""

from volengine.backtesting.backtest_engine import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
)
from volengine.backtesting.data_loader import (
    OptionChainSnapshot,
    filter_for_calibration,
    load_option_chain,
)

__all__ = [
    "OptionChainSnapshot",
    "load_option_chain",
    "filter_for_calibration",
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
]
