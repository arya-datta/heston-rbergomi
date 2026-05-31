"""Calibration objective: weighted IV RMSE versus market.

A quote bundles together (strike, maturity, observed mid IV, weight). Model-
agnostic: any pricer that maps (K, T, params) -> IV plugs in. The weight is
typically 1/vega_BS or 1/spread_BS so the calibrator treats liquid ATM points
as more informative than illiquid wings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# Penalty (in vol points) applied to a quote whose model price fails to invert
# to a finite implied vol. Set to 1.0 = 100 vol points: still far above any real
# RMSE (~0.01-0.05), so it strongly discourages parameter regions that produce
# un-invertible prices, but 25x smaller in squared terms than the old 5.0 (=500
# vol points), which created needlessly steep cliffs that destabilized the
# global differential-evolution search.
FAILED_QUOTE_PENALTY = 1.0


@dataclass(frozen=True)
class IVQuote:
    """A single market-observed implied vol quote."""
    K: float
    T: float
    iv_mkt: float
    weight: float = 1.0


def build_iv_quotes(
    strikes: Sequence[float],
    maturities: Sequence[float],
    ivs: Sequence[float],
    spreads: Sequence[float] | None = None,
    eps: float = 1e-3,
) -> list[IVQuote]:
    """Build a weighted IVQuote list, weighting by inverse bid-ask spread.

    Standalone calibration users should prefer this over hand-constructing
    `IVQuote(..., weight=1.0)`: weighting tight (liquid) quotes more heavily
    than wide (illiquid) wings materially improves the fit, and the backtest
    harness already does exactly this internally. Passing `spreads=None`
    falls back to uniform weights.

    Parameters
    ----------
    strikes, maturities, ivs : equal-length sequences.
    spreads : per-quote bid-ask spread (same units as price). If None, uniform.
    eps : floor on the spread so a zero/locked quote doesn't get infinite weight.

    Returns
    -------
    list[IVQuote] with finite, positive IVs only (NaN/non-positive dropped).
    """
    n = len(strikes)
    if not (len(maturities) == len(ivs) == n):
        raise ValueError("strikes, maturities, ivs must be the same length.")
    if spreads is not None and len(spreads) != n:
        raise ValueError("spreads must match the length of strikes.")

    quotes: list[IVQuote] = []
    for i in range(n):
        iv = float(ivs[i])
        if not np.isfinite(iv) or iv <= 0:
            continue
        w = 1.0 if spreads is None else 1.0 / max(float(spreads[i]), eps)
        quotes.append(IVQuote(K=float(strikes[i]), T=float(maturities[i]),
                              iv_mkt=iv, weight=w))
    return quotes


def iv_rmse_objective(
    quotes: list[IVQuote],
    model_iv_fn: Callable[[float, float], float],
) -> float:
    """Compute weighted RMSE of model IV vs market IV across a quote list.

    Parameters
    ----------
    quotes : list of IVQuote.
    model_iv_fn : callable (K, T) -> model implied vol. Should NOT raise on
                  numerical failures — return NaN, which we filter out.

    Returns
    -------
    Weighted root mean squared error in vol points (not bps).
    """
    sq, w_sum = 0.0, 0.0
    for q in quotes:
        iv_model = model_iv_fn(q.K, q.T)
        if not np.isfinite(iv_model):
            # Penalty for numerical failure — drives the optimizer away from
            # parameter regions that produce un-invertible prices.
            sq += q.weight * FAILED_QUOTE_PENALTY ** 2
            w_sum += q.weight
            continue
        err = iv_model - q.iv_mkt
        sq += q.weight * err * err
        w_sum += q.weight
    return float(np.sqrt(sq / max(w_sum, 1e-12)))
