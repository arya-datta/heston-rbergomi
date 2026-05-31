"""Generate the repo's headline plot: ATM-skew term structure on real data.

End-to-end pipeline:
    chain ─► SVI surface ─► Heston + rBergomi calibration
                       │
                       └─► ATM skew (market / Heston / rBergomi) ─► log-log plot

Produces `results/figures/atm_skew_term_structure.png` and prints the
fitted power-law exponent (≈ H − 1/2) for each curve.

Runtime: ~6–10 minutes on a laptop (rBergomi calibration is the bottleneck).

Usage:
    volengine-generate-headline                            # default: SPY via yfinance
    volengine-generate-headline --symbol SPX --provider intrinio
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volengine.analysis.atm_skew import (
    atm_skew_from_surface,
    fit_skew_power_law,
    heston_atm_skew,
    rbergomi_atm_skew,
)
from volengine.backtesting.data_loader import (
    OptionChainSnapshot,
    filter_for_calibration,
    load_option_chain,
)
from volengine.calibration import IVQuote, calibrate_heston, calibrate_rbergomi
from volengine.surfaces.implied_vol import implied_vol
from volengine.surfaces.svi import SVISurface, fit_svi_slice


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}")


def try_load(symbols: list[str], provider: str, date: dt.date,
             r: float, q: float) -> OptionChainSnapshot:
    """Load a chain, walking symbol fallbacks; reuses on-disk cache if present."""
    last_err: Exception | None = None
    for sym in symbols:
        try:
            return load_option_chain(symbol=sym, date=date, r=r, q=q,
                                     provider=provider, refresh=False)
        except Exception as e:
            last_err = e
            print(f"[load] {sym} failed: {e}")
    raise RuntimeError(f"All symbols failed. Last error: {last_err}")


def build_quotes_and_surface(
    snap: OptionChainSnapshot,
) -> tuple[list[IVQuote], SVISurface, dict[float, float]]:
    """Filter the chain, invert IVs, fit SVI per maturity.

    Returns (IVQuote list, SVI surface, dict of maturity-> ATM iv) so downstream
    code can plot whatever it needs without re-deriving these.
    """
    df = filter_for_calibration(
        snap, min_dte_days=14, max_dte_days=365, min_oi=10,
        max_spread_frac=0.50, moneyness_band=(0.85, 1.15), flag="call",
    )
    quotes: list[IVQuote] = []
    slices: dict[float, object] = {}
    atm_iv: dict[float, float] = {}
    for T, group in df.groupby("dte_years"):
        if T < 14 / 365.25:                  # skip ultra-short which is noisy
            continue
        group = group.sort_values("strike")
        K = group["strike"].to_numpy(dtype=float)
        mids = group["mid"].to_numpy(dtype=float)
        ivs = np.array([
            implied_vol(p, snap.spot, k, T, snap.r, snap.q, "call")
            for p, k in zip(mids, K, strict=True)
        ])
        ok = np.isfinite(ivs) & (ivs > 0)
        if ok.sum() < 5:
            continue
        K, ivs = K[ok], ivs[ok]
        k_log = np.log(K / snap.spot)
        try:
            p = fit_svi_slice(k_log, ivs, T)
        except Exception:
            continue
        slices[float(T)] = p
        # ATM iv via SVI at k = 0
        atm_iv[float(T)] = float(np.sqrt(max(
            np.interp(0.0, k_log, ivs ** 2), 1e-12)))
        for kk, iv in zip(K, ivs, strict=True):
            # Weight inversely with distance from ATM — liquid quotes dominate.
            w = 1.0 / max(abs(np.log(kk / snap.spot)) + 0.05, 0.05)
            quotes.append(IVQuote(K=float(kk), T=T, iv_mkt=float(iv), weight=w))
    surface = SVISurface(maturities=np.array(sorted(slices.keys())), slices=slices)
    return quotes, surface, atm_iv


def compute_skew_curves(
    surface: SVISurface,
    heston_params,
    rbergomi_params,
    S0: float, r: float, q: float,
    maturities: np.ndarray,
    rb_n_paths: int = 60_000,
    rb_n_steps_per_year: int = 100,
    seed: int = 7,
) -> pd.DataFrame:
    """Compute ATM skew under each model across `maturities`."""
    rows = []
    for T in maturities:
        # Market skew from the (possibly interpolated) SVI surface.
        # Find the closest fitted slice — atm_skew_from_surface needs a slice, not surface.
        Ts = sorted(surface.slices.keys())
        T_near = Ts[int(np.argmin(np.abs(np.array(Ts) - T)))]
        mkt = atm_skew_from_surface(surface.slices[T_near], T_near)

        h = heston_atm_skew(heston_params, T=T, S0=S0, r=r, q=q, dk=5e-3)
        rb = rbergomi_atm_skew(
            rbergomi_params, T=T, S0=S0, r=r, q=q, dk=1.5e-2,
            n_paths=rb_n_paths,
            n_steps=max(20, int(rb_n_steps_per_year * T)),
            seed=seed,
        )
        rows.append(dict(T=T, market=mkt, heston=h, rbergomi=rb))
    return pd.DataFrame(rows)


def make_figure(skew_df: pd.DataFrame, out_path: Path, symbol: str,
                snapshot_date: dt.date) -> dict[str, tuple[float, float]]:
    """Plot log-log ATM skew with three overlays. Returns power-law fits."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6))

    fits: dict[str, tuple[float, float]] = {}
    styles = {
        "market":  dict(color="black",  marker="o", lw=0,   ms=7, label=f"{symbol} market (SVI)"),
        "heston":  dict(color="C0",     marker="s", lw=1.5, ms=5, label="Heston (calibrated)",   linestyle="--"),
        "rbergomi": dict(color="C3",    marker="^", lw=2.0, ms=5, label="rBergomi (calibrated)"),
    }
    Ts = skew_df["T"].to_numpy()
    for name, style in styles.items():
        y = skew_df[name].to_numpy()
        ax.loglog(Ts, y, **style)
        alpha, c = fit_skew_power_law(Ts, y)
        fits[name] = (alpha, c)
        if np.isfinite(alpha):
            Tline = np.geomspace(Ts.min(), Ts.max(), 50)
            ax.loglog(Tline, np.exp(c) * Tline ** alpha,
                      color=style["color"], lw=0.8, alpha=0.4)

    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("ATM skew  $|\\partial \\sigma_{\\rm imp}/\\partial k|_{k=0}$")
    ax.set_title(
        f"ATM-skew term structure: {symbol} on {snapshot_date}\n"
        f"market slope = {fits['market'][0]:+.2f}  →  H ≈ {fits['market'][0] + 0.5:+.2f}\n"
        f"Heston slope = {fits['heston'][0]:+.2f}    rBergomi slope = {fits['rbergomi'][0]:+.2f}"
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"\n[fig] saved to {out_path}")
    return fits


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default=None,
                        help="Ticker (default: walk SPY then SPX then ^SPX).")
    parser.add_argument("--provider", default="yfinance")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today).")
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--q", type=float, default=0.013)
    parser.add_argument("--rb-paths", type=int, default=12_000,
                        help="rBergomi paths during calibration (default: 12k for speed).")
    parser.add_argument("--skew-paths", type=int, default=60_000,
                        help="rBergomi paths for the skew-MC step (separate; default: 60k).")
    parser.add_argument("--out", default="results/figures/atm_skew_term_structure.png")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Use default parameters instead of calibrating (debug).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    t0 = time.time()
    date = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    symbols = [args.symbol] if args.symbol else ["SPY", "SPX", "^SPX"]

    try:
        snap = try_load(symbols, args.provider, date, args.r, args.q)
        _log(f"loaded {snap.symbol}: spot={snap.spot:.2f}, "
             f"{len(snap.chain)} raw quotes", t0)
    except Exception:
        traceback.print_exc()
        return 1

    quotes, surface, atm_iv = build_quotes_and_surface(snap)
    _log(f"built {len(quotes)} quotes across "
         f"{len(surface.slices)} maturities", t0)
    if len(quotes) < 30 or len(surface.slices) < 4:
        print("[error] too few quotes or maturities for a meaningful skew plot")
        return 1

    from volengine.models.heston import HestonParameters
    from volengine.models.rbergomi import RBergomiParameters

    if args.no_calibration:
        h_params = HestonParameters(kappa=1.5, theta=0.04, xi=0.5, rho=-0.75, v0=0.04)
        rb_params = RBergomiParameters(H=0.10, eta=1.9, rho=-0.85, xi0=0.04)
        _log("skipping calibration (--no-calibration)", t0)
    else:
        h_res = calibrate_heston(quotes, S0=snap.spot, r=snap.r, q=snap.q,
                                  de_maxiter=40, de_popsize=15, seed=0)
        h_params = h_res.params
        _log(f"Heston calibrated: RMSE = {h_res.rmse_vol_points*100:.2f} vol pts, "
             f"params = {h_params}", t0)
        rb_res = calibrate_rbergomi(
            quotes, S0=snap.spot, r=snap.r, q=snap.q,
            n_paths=args.rb_paths, n_steps_per_year=80,
            de_maxiter=12, de_popsize=10, seed=0,
        )
        rb_params = rb_res.params
        _log(f"rBergomi calibrated: RMSE = {rb_res.rmse_vol_points*100:.2f} vol pts, "
             f"params = {rb_params}", t0)

    # Skew grid: log-spaced from shortest fitted slice to longest, capped at 1y
    Ts_fit = np.array(sorted(surface.slices.keys()))
    T_min = max(Ts_fit.min(), 14 / 365.25)
    T_max = min(Ts_fit.max(), 1.0)
    skew_Ts = np.geomspace(T_min, T_max, 7)
    _log(f"computing skew at {len(skew_Ts)} maturities: "
         f"{[f'{T:.2f}y' for T in skew_Ts]}", t0)

    skew_df = compute_skew_curves(
        surface, h_params, rb_params,
        S0=snap.spot, r=snap.r, q=snap.q,
        maturities=skew_Ts,
        rb_n_paths=args.skew_paths,
    )
    _log("skew computation done", t0)
    print("\n=== Skew table ===")
    print(skew_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    fits = make_figure(skew_df, Path(args.out), snap.symbol, date)
    _log("done", t0)

    print("\n=== Power-law slopes (slope α; H = α + 0.5) ===")
    for name, (alpha, _c) in fits.items():
        H_imp = alpha + 0.5
        print(f"  {name:9s}: α = {alpha:+.3f}, implied H = {H_imp:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
