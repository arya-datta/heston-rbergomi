"""Rough Bergomi calibration.

rBergomi has no closed-form characteristic function; every price evaluation is
a Monte Carlo simulation. We share random numbers across objective evaluations
(by fixing the seed) so the objective is a *deterministic* function of (H, eta,
rho, xi0). This is essential for the local refinement stage — L-BFGS-B requires
gradients estimated via finite differences, and stochastic noise in those
differences will derail optimization.

The xi0 parameter is typically calibrated jointly. For simplicity we
parameterize it as a flat level; production work would calibrate a piecewise-
constant forward variance curve from ATM term structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

from volengine.calibration.objective import FAILED_QUOTE_PENALTY, IVQuote
from volengine.market.curves import Curve, as_curve
from volengine.models.rbergomi.parameters import RBERGOMI_BOUNDS, RBergomiParameters
from volengine.models.rbergomi.pricing import simulate_rbergomi
from volengine.surfaces.implied_vol import implied_vol


@dataclass
class RBergomiCalibrationResult:
    params: RBergomiParameters
    rmse_vol_points: float
    n_quotes: int
    success: bool
    message: str


def _model_ivs_rbergomi(
    params: RBergomiParameters,
    S0: float,
    r_curve: Curve,
    q_curve: Curve,
    quotes: list[IVQuote],
    n_paths: int,
    n_steps_per_year: int,
    seed: int,
) -> np.ndarray:
    """Compute rBergomi model IVs for each quote, simulating once per unique maturity.

    Each maturity uses its own zero rates r(T), q(T) from the supplied curves.
    """
    out = np.full(len(quotes), np.nan)
    by_T: dict[float, list[int]] = {}
    for i, qte in enumerate(quotes):
        by_T.setdefault(qte.T, []).append(i)

    for T, idxs in by_T.items():
        r_T, q_T = r_curve.zero_rate(T), q_curve.zero_rate(T)
        n_steps = max(20, int(np.ceil(n_steps_per_year * T)))
        S = simulate_rbergomi(
            S0=S0, T=T, params=params, r=r_T, q=q_T,
            n_paths=n_paths, n_steps=n_steps,
            seed=seed, antithetic=True,
        )
        ST = S[:, -1]
        disc = np.exp(-r_T * T)
        for i in idxs:
            K = quotes[i].K
            price = disc * np.maximum(ST - K, 0.0).mean()
            out[i] = implied_vol(float(price), S0, K, T, r_T, q_T, "call")
    return out


def _objective_factory_rbergomi(
    quotes: list[IVQuote],
    S0: float, r_curve: Curve, q_curve: Curve,
    n_paths: int,
    n_steps_per_year: int,
    seed: int,
):
    weights = np.array([qt.weight for qt in quotes])
    iv_mkt = np.array([qt.iv_mkt for qt in quotes])

    def f(x: np.ndarray) -> float:
        H, eta, rho, xi0 = x
        params = RBergomiParameters(H=H, eta=eta, rho=rho, xi0=xi0)
        iv_model = _model_ivs_rbergomi(params, S0, r_curve, q_curve, quotes,
                                       n_paths, n_steps_per_year, seed)
        bad = ~np.isfinite(iv_model)
        err = np.where(bad, FAILED_QUOTE_PENALTY, iv_model - iv_mkt)
        return float(np.sqrt(np.sum(weights * err**2) / max(weights.sum(), 1e-12)))

    return f


def calibrate_rbergomi(
    quotes: list[IVQuote],
    S0: float,
    r: float | Curve,
    q: float | Curve,
    initial: RBergomiParameters | None = None,
    n_paths: int = 20_000,
    n_steps_per_year: int = 100,
    de_maxiter: int = 40,
    de_popsize: int = 15,
    seed: int = 42,
    skip_global: bool = False,
) -> RBergomiCalibrationResult:
    """Calibrate rBergomi (H, eta, rho, xi0) to a set of IV quotes.

    Parameters
    ----------
    quotes : list of IVQuote.
    S0 : spot.
    r, q : risk-free rate and dividend yield. Either a float (flat) or a
        `volengine.market.Curve` for true term structure.
    initial : optional warm start.
    n_paths, n_steps_per_year : MC dimensions. The default 20k x 100 is a
        budget compromise between calibration time and parameter precision;
        increase n_paths if you see noisy local minima.
    de_maxiter, de_popsize : DE settings. rBergomi is 4D so DE converges
        faster than Heston's 5D problem.
    seed : MC seed. Fixed across the optimizer call so the objective is
        deterministic and L-BFGS-B finite differences work.
    skip_global : skip DE if `initial` is a good warm start.

    Returns
    -------
    RBergomiCalibrationResult.
    """
    if not quotes:
        raise ValueError("No quotes supplied to calibration.")

    r_curve, q_curve = as_curve(r), as_curve(q)
    obj = _objective_factory_rbergomi(
        quotes, S0, r_curve, q_curve, n_paths, n_steps_per_year, seed,
    )
    bounds = [RBERGOMI_BOUNDS[k] for k in ("H", "eta", "rho", "xi0")]

    if not skip_global:
        de = differential_evolution(
            obj, bounds=bounds,
            maxiter=de_maxiter, popsize=de_popsize,
            seed=seed, tol=1e-5, polish=False,
            init="sobol", workers=1,
        )
        x0 = de.x
    else:
        if initial is None:
            raise ValueError("skip_global=True requires an `initial` warm start.")
        x0 = np.array([initial.H, initial.eta, initial.rho, float(initial.xi0)])

    local = minimize(
        obj, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 80, "ftol": 1e-8},
    )
    H, eta, rho, xi0 = local.x
    return RBergomiCalibrationResult(
        params=RBergomiParameters(H=H, eta=eta, rho=rho, xi0=xi0),
        rmse_vol_points=float(local.fun),
        n_quotes=len(quotes),
        success=bool(local.success),
        message=str(local.message),
    )


def calibrate_rbergomi_fixed_xi0(
    quotes: list[IVQuote],
    S0: float,
    r: float | Curve,
    q: float | Curve,
    xi0_curve,
    initial: RBergomiParameters | None = None,
    n_paths: int = 20_000,
    n_steps_per_year: int = 100,
    de_maxiter: int = 40,
    de_popsize: int = 15,
    seed: int = 42,
    skip_global: bool = False,
) -> RBergomiCalibrationResult:
    """Calibrate only the dynamics parameters (H, eta, rho), with xi0 fixed.

    This is the *textbook* way to use rough Bergomi: the forward-variance curve
    xi0(t) is read from the market term structure (see
    `volengine.models.rbergomi.ForwardVarianceCurve.from_atm_term_structure`),
    and only the three roughness/vol-of-vol/leverage parameters are fit. The
    level structure is therefore pinned exactly by the market rather than being
    absorbed into a single flat xi0 scalar, which both improves the fit and
    makes the calibrated (H, eta, rho) interpretable.

    Parameters
    ----------
    xi0_curve : callable t -> xi0(t), e.g. a `ForwardVarianceCurve`. Held fixed.
    (other parameters as in `calibrate_rbergomi`.)

    Returns
    -------
    RBergomiCalibrationResult whose params.xi0 is the supplied curve.
    """
    if not quotes:
        raise ValueError("No quotes supplied to calibration.")
    if not callable(xi0_curve):
        raise TypeError("xi0_curve must be callable (e.g. a ForwardVarianceCurve).")

    r_curve, q_curve = as_curve(r), as_curve(q)
    weights = np.array([qt.weight for qt in quotes])
    iv_mkt = np.array([qt.iv_mkt for qt in quotes])

    def obj(x: np.ndarray) -> float:
        H, eta, rho = x
        params = RBergomiParameters(H=H, eta=eta, rho=rho, xi0=xi0_curve)
        iv_model = _model_ivs_rbergomi(params, S0, r_curve, q_curve, quotes,
                                       n_paths, n_steps_per_year, seed)
        bad = ~np.isfinite(iv_model)
        err = np.where(bad, FAILED_QUOTE_PENALTY, iv_model - iv_mkt)
        return float(np.sqrt(np.sum(weights * err**2) / max(weights.sum(), 1e-12)))

    bounds = [RBERGOMI_BOUNDS[k] for k in ("H", "eta", "rho")]
    if not skip_global:
        de = differential_evolution(
            obj, bounds=bounds, maxiter=de_maxiter, popsize=de_popsize,
            seed=seed, tol=1e-5, polish=False, init="sobol", workers=1,
        )
        x0 = de.x
    else:
        if initial is None:
            raise ValueError("skip_global=True requires an `initial` warm start.")
        x0 = np.array([initial.H, initial.eta, initial.rho])

    local = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 80, "ftol": 1e-8})
    H, eta, rho = local.x
    return RBergomiCalibrationResult(
        params=RBergomiParameters(H=H, eta=eta, rho=rho, xi0=xi0_curve),
        rmse_vol_points=float(local.fun),
        n_quotes=len(quotes),
        success=bool(local.success),
        message=str(local.message),
    )
