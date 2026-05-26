"""Calibration objective: weighted IV RMSE versus market.

A quote bundles together (strike, maturity, observed mid IV, weight). Model-
agnostic: any pricer that maps (K, T, params) -> IV plugs in. The weight is
typically 1/vega_BS or 1/spread_BS so the calibrator treats liquid ATM points
as more informative than illiquid wings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class IVQuote:
    """A single market-observed implied vol quote."""
    K: float
    T: float
    iv_mkt: float
    weight: float = 1.0


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
            # Heavy penalty for numerical failure — drives the optimizer away.
            sq += q.weight * (5.0) ** 2
            w_sum += q.weight
            continue
        err = iv_model - q.iv_mkt
        sq += q.weight * err * err
        w_sum += q.weight
    return float(np.sqrt(sq / max(w_sum, 1e-12)))
