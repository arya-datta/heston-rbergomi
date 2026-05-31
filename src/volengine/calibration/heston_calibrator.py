"""Heston calibration: differential evolution + L-BFGS-B refinement.

The objective is computed in IV space — for each market quote we (a) price
the call under the current Heston parameters via Carr-Madan FFT, (b) invert
Black-Scholes to recover the model IV, (c) compare to market IV.

This is more expensive than price-space MSE but gives a far better-conditioned
landscape: IV errors are roughly homogeneous across strikes/maturities, where
price MSE is dominated by ITM quotes and ignores the wings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

from volengine.calibration.objective import FAILED_QUOTE_PENALTY, IVQuote
from volengine.models.heston.carr_madan import heston_vanilla_price
from volengine.models.heston.parameters import HESTON_BOUNDS, HestonParameters
from volengine.surfaces.implied_vol import implied_vol


@dataclass
class HestonCalibrationResult:
    """Output of `calibrate_heston`."""
    params: HestonParameters
    rmse_vol_points: float
    n_quotes: int
    feller_ok: bool
    success: bool
    message: str


# Carr-Madan grid used inside the calibration inner loop. Coarser than the
# 8192/0.15 default used for one-off accuracy checks: at N=4096, eta=0.25 the
# FFT prices are accurate to ~1 bp on liquid strikes — far below calibration
# IV-fit error — while running ~2x faster. The final calibrated params can be
# re-priced at full resolution for reporting.
_CALIB_FFT_KWARGS = {"N": 4096, "eta": 0.25}


def _model_ivs(
    params: HestonParameters,
    S0: float,
    r: float,
    q: float,
    quotes: list[IVQuote],
) -> np.ndarray:
    """Vectorized model IVs for a list of quotes, grouped by maturity for FFT speed."""
    out = np.empty(len(quotes))
    out[:] = np.nan
    # Group indices by maturity so each maturity only invokes one FFT.
    by_T: dict[float, list[int]] = {}
    for i, qte in enumerate(quotes):
        by_T.setdefault(qte.T, []).append(i)
    for T, idxs in by_T.items():
        Ks = np.array([quotes[i].K for i in idxs])
        try:
            prices = heston_vanilla_price(Ks, T, S0, r, q, params, flag="call",
                                          **_CALIB_FFT_KWARGS)
        except (FloatingPointError, ValueError):
            continue
        prices = np.atleast_1d(prices)
        for j, i in enumerate(idxs):
            out[i] = implied_vol(float(prices[j]), S0, Ks[j], T, r, q, "call")
    return out


def _objective_factory(
    quotes: list[IVQuote],
    S0: float,
    r: float,
    q: float,
):
    """Closure that maps a 5-vector (kappa, theta, xi, rho, v0) -> weighted RMSE."""
    weights = np.array([qt.weight for qt in quotes])
    iv_mkt = np.array([qt.iv_mkt for qt in quotes])

    def f(x: np.ndarray) -> float:
        params = HestonParameters(*x)
        iv_model = _model_ivs(params, S0, r, q, quotes)
        bad = ~np.isfinite(iv_model)
        # Penalty for failed pricings pushes DE away from bad regions.
        err = np.where(bad, FAILED_QUOTE_PENALTY, iv_model - iv_mkt)
        return float(np.sqrt(np.sum(weights * err**2) / max(weights.sum(), 1e-12)))

    return f


def calibrate_heston(
    quotes: list[IVQuote],
    S0: float,
    r: float,
    q: float,
    initial: HestonParameters | None = None,
    de_maxiter: int = 80,
    de_popsize: int = 20,
    seed: int | None = 42,
    skip_global: bool = False,
) -> HestonCalibrationResult:
    """Calibrate Heston parameters to a set of IV quotes.

    Parameters
    ----------
    quotes : list of (K, T, iv_mkt, weight) tuples.
    S0, r, q : spot and rates.
    initial : optional warm start (typical: previous day's params).
    de_maxiter, de_popsize : differential evolution settings.
    seed : reproducibility for the global stage.
    skip_global : if True (and `initial` is supplied), skip DE and go straight
        to L-BFGS-B. Useful in rolling-window backtests where each day's
        previous-day params are already a good starting point.

    Returns
    -------
    HestonCalibrationResult.
    """
    if not quotes:
        raise ValueError("No quotes supplied to calibration.")

    obj = _objective_factory(quotes, S0, r, q)
    bounds = [HESTON_BOUNDS[k] for k in ("kappa", "theta", "xi", "rho", "v0")]

    # Global stage.
    if not skip_global:
        de = differential_evolution(
            obj,
            bounds=bounds,
            maxiter=de_maxiter,
            popsize=de_popsize,
            seed=seed,
            tol=1e-6,
            polish=False,
            init="sobol",
            workers=1,
        )
        x0 = de.x
    else:
        if initial is None:
            raise ValueError("skip_global=True requires an `initial` warm start.")
        x0 = np.array(initial.as_tuple())

    # Local refinement.
    local = minimize(
        obj,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-10},
    )
    params = HestonParameters(*local.x)
    return HestonCalibrationResult(
        params=params,
        rmse_vol_points=float(local.fun),
        n_quotes=len(quotes),
        feller_ok=params.feller_condition(),
        success=bool(local.success),
        message=str(local.message),
    )
