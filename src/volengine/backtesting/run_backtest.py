"""CLI runner for the backtest engine.

Usage
-----
    python -m volengine.backtesting.run_backtest \
        --symbol SPX --start 2024-01-02 --end 2024-06-30 \
        --provider yfinance

Outputs `results/backtest/daily_results.parquet` and prints a summary table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

from volengine.backtesting.backtest_engine import BacktestConfig, run_backtest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenBB-driven Heston/rBergomi backtest.")
    parser.add_argument("--symbol", default="SPX", help="Underlying ticker.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD.")
    parser.add_argument("--provider", default="yfinance", help="OpenBB provider.")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate.")
    parser.add_argument("--q", type=float, default=0.015, help="Dividend yield.")
    parser.add_argument("--cache-dir", default="data/cache", help="Quote cache directory.")
    parser.add_argument("--save-dir", default="results/backtest", help="Output directory.")
    parser.add_argument("--no-warm-start", action="store_true",
                        help="Disable param warm-starting across days (slower, more independent).")
    parser.add_argument("--rbergomi-paths", type=int, default=15_000,
                        help="MC paths for rBergomi calibration.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    cfg = BacktestConfig(
        symbol=args.symbol,
        start=dt.date.fromisoformat(args.start),
        end=dt.date.fromisoformat(args.end),
        r=args.r, q=args.q,
        provider=args.provider,
        cache_dir=Path(args.cache_dir),
        save_dir=Path(args.save_dir),
        warm_start=not args.no_warm_start,
        rbergomi_n_paths=args.rbergomi_paths,
    )

    result = run_backtest(cfg)
    df = result.to_dataframe()
    if df.empty:
        print("No days completed.")
        return 1

    summary = df.agg({
        "n_quotes": "mean",
        "heston_rmse": "median",
        "rbergomi_rmse": "median",
        "model_risk_spread_bps": "median",
    })
    print("\n=== Backtest summary ===")
    print(summary.to_string())
    print(f"\nPer-day results written to {cfg.save_dir}/daily_results.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
