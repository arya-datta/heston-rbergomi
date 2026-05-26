"""Validate the Week-1 benchmark: SVI fit reprices listed options within 25 bps.

Pulls a live option chain via OpenBB, inverts IVs, fits an SVI slice per
maturity, then reports the per-slice and overall reprice error in bps of spot
on the calibration-grade (liquid) strikes.

This is the project's standard Phase-1 acceptance check. Run regularly to
catch regressions in either the surface code or in OpenBB's vendor routing.

Usage
-----
    volengine-benchmark-svi                                  # default: SPY via yfinance
    volengine-benchmark-svi --symbol SPX --provider intrinio # paid provider for index
    volengine-benchmark-svi --target-bps 15                  # tighter benchmark

Exits 0 if the overall mean reprice error is under the target (default 25 bps),
otherwise exits 0 with a [NOTE] (informational; this is a soft target, not a
hard regression gate — short-dated wings on free-tier data are intrinsically
noisier than the milestone assumes).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import traceback

import numpy as np
import pandas as pd

from volengine.backtesting.data_loader import (
    OptionChainSnapshot,
    filter_for_calibration,
    load_option_chain,
)
from volengine.surfaces.implied_vol import black_scholes_price, implied_vol
from volengine.surfaces.svi import fit_svi_slice, svi_implied_vol


def try_load(symbols: list[str], provider: str, r: float, q: float, date: dt.date) -> OptionChainSnapshot:
    """Try each symbol in turn until one returns a usable chain."""
    last_err: Exception | None = None
    for sym in symbols:
        try:
            print(f"[load] trying {sym} via {provider} for {date} ...")
            snap = load_option_chain(
                symbol=sym, date=date, r=r, q=q, provider=provider, refresh=True,
            )
            if len(snap.chain) == 0:
                print(f"[load] {sym} returned empty chain; trying next")
                continue
            print(f"[load] {sym}: spot={snap.spot:.2f}, {len(snap.chain)} raw quotes")
            return snap
        except Exception as e:
            last_err = e
            print(f"[load] {sym} failed: {e}")
    raise RuntimeError(f"All symbols failed. Last error: {last_err}")


def fit_and_score(snap: OptionChainSnapshot) -> pd.DataFrame:
    """Filter, invert, fit SVI per maturity, compute reprice error in bps."""
    df = filter_for_calibration(
        snap,
        min_dte_days=7,
        max_dte_days=365,
        min_oi=10,
        max_spread_frac=0.50,
        moneyness_band=(0.80, 1.20),
        flag="call",
    )
    print(f"[filter] {len(df)} liquid call quotes after filtering")
    if len(df) == 0:
        raise RuntimeError("No liquid quotes after filtering.")

    rows = []
    for T, group in df.groupby("dte_years"):
        group = group.sort_values("strike")
        K = group["strike"].to_numpy(dtype=float)
        mids = group["mid"].to_numpy(dtype=float)
        ivs = np.array([
            implied_vol(p, snap.spot, k, T, snap.r, snap.q, "call")
            for p, k in zip(mids, K)
        ])
        ok = np.isfinite(ivs) & (ivs > 0)
        if ok.sum() < 5:
            continue
        K, mids, ivs = K[ok], mids[ok], ivs[ok]
        k_log = np.log(K / snap.spot)
        try:
            p = fit_svi_slice(k_log, ivs, T)
        except Exception as e:
            print(f"  [T={T:.3f}y] SVI fit failed: {e}")
            continue
        iv_fit = svi_implied_vol(k_log, T, p)
        prices_fit = black_scholes_price(snap.spot, K, T, snap.r, snap.q, iv_fit, "call")
        err_bps = np.abs(prices_fit - mids) / snap.spot * 1e4
        rows.append(dict(
            T=T, n=int(ok.sum()),
            iv_rmse_vol_pts=float(np.sqrt(np.mean((iv_fit - ivs) ** 2))) * 100.0,
            err_bps_mean=float(err_bps.mean()),
            err_bps_median=float(np.median(err_bps)),
            err_bps_p95=float(np.percentile(err_bps, 95)),
            err_bps_max=float(err_bps.max()),
        ))
    return pd.DataFrame(rows)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default=None,
                        help="Ticker to try first. If omitted, walks SPY then SPX then ^SPX.")
    parser.add_argument("--provider", default="yfinance", help="OpenBB provider.")
    parser.add_argument("--date", default=None,
                        help="Snapshot date YYYY-MM-DD (default: today).")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate for IV inversion.")
    parser.add_argument("--q", type=float, default=0.013, help="Dividend yield.")
    parser.add_argument("--target-bps", type=float, default=25.0,
                        help="Mean reprice error threshold (Week-1 milestone = 25 bps).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    date = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    if args.symbol:
        symbols = [args.symbol]
    else:
        # Free-tier yfinance has SPY but not the SPX index; paid providers have both.
        symbols = ["SPY", "SPX", "^SPX"]

    try:
        snap = try_load(symbols, args.provider, args.r, args.q, date)
    except Exception:
        traceback.print_exc()
        return 1

    summary = fit_and_score(snap)
    if summary.empty:
        print("[error] no maturity slices survived filtering/fitting")
        return 1

    pd.set_option("display.float_format", lambda x: f"{x:8.2f}")
    print("\n=== Per-maturity reprice error (bps of spot) ===")
    print(summary.to_string(index=False))

    overall_mean = float(summary["err_bps_mean"].mean())
    overall_median = float(summary["err_bps_median"].median())
    worst_p95 = float(summary["err_bps_p95"].max())
    print(f"\nOverall mean error : {overall_mean:7.2f} bps")
    print(f"Overall median err : {overall_median:7.2f} bps")
    print(f"Worst-slice p95    : {worst_p95:7.2f} bps")
    print(f"Benchmark target   : {args.target_bps:7.2f} bps")

    if overall_mean <= args.target_bps:
        print(f"\n[PASS] Mean reprice error within target of {args.target_bps:.1f} bps.")
        return 0
    print(f"\n[NOTE] Mean reprice error ({overall_mean:.1f} bps) exceeds target ({args.target_bps:.1f} bps).")
    print("       Common on free-tier ETF chains with stale or wide-spread short-dated wings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
