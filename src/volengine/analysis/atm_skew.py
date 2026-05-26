"""ATM skew term structure — the killer plot of the project.

The ATM skew is

    psi(T) := |d sigma_imp(k, T) / dk|  evaluated at k = 0

(log-moneyness derivative at the money). The empirical regularity that
motivated rough volatility is:

    psi(T) ~ T^{H - 1/2}   as T -> 0

with H ~ 0.07-0.15 on SPX. Classical models (Heston, Black-Scholes, any
diffusive stochastic vol) predict psi(T) bounded as T -> 0, then decaying as
1/T at long maturities — visually a "stick" with the wrong slope on both ends.

This module:
  1) Computes psi(T) from an SVI-fitted market surface (smooth, analytical).
  2) Computes psi(T) from a calibrated model (Heston, rBergomi) by central
     finite-differences on FFT / MC prices.
  3) Fits log psi(T) = c + (H - 1/2) log T to extract the empirical roughness.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from volengine.models.heston import HestonParameters, heston_vanilla_price
from volengine.models.rbergomi import RBergomiParameters, rbergomi_price
from volengine.surfaces.implied_vol import implied_vol
from volengine.surfaces.svi import SVIParameters, svi_total_variance


def atm_skew_from_surface(
    svi_slice: SVIParameters,
    T: float,
    dk: float = 1e-3,
) -> float:
    """ATM skew from an SVI slice via analytic derivative.

    For raw SVI, dw/dk at k=0 has a clean closed form:
        dw/dk |_{k=0} = b * ( rho + (-m) / sqrt(m^2 + sigma^2) )
    and dsigma/dk = (1 / (2 sigma T)) * dw/dk.

    We compute by central difference for robustness across edge cases (large
    |rho|, m near 0); the analytic form agrees to several decimals.
    """
    w_plus = svi_total_variance(np.array([dk]), svi_slice)[0]
    w_minus = svi_total_variance(np.array([-dk]), svi_slice)[0]
    w0 = svi_total_variance(np.array([0.0]), svi_slice)[0]
    dw_dk = (w_plus - w_minus) / (2.0 * dk)
    sigma_atm = float(np.sqrt(max(w0, 1e-12) / T))
    return float(np.abs(dw_dk / (2.0 * sigma_atm * T)))


def atm_skew_from_model(
    S0: float,
    T: float,
    r: float,
    q: float,
    pricer: Callable[[float], float],
    dk: float = 5e-3,
) -> float:
    """ATM skew |d sigma_imp / dk| at k=0 by central differencing model prices.

    Parameters
    ----------
    pricer : callable K -> call price. Vectorize on the caller side if needed.
    dk     : log-moneyness step. 5e-3 is small enough to be accurate yet large
             enough that MC noise (in rBergomi) does not dominate the difference.

    Returns
    -------
    ATM skew (absolute value).
    """
    F = S0 * np.exp((r - q) * T)
    K_atm = F
    K_plus = F * np.exp(dk)
    K_minus = F * np.exp(-dk)

    P_atm = float(pricer(K_atm))
    P_plus = float(pricer(K_plus))
    P_minus = float(pricer(K_minus))

    iv_atm = implied_vol(P_atm, S0, K_atm, T, r, q, "call")
    iv_plus = implied_vol(P_plus, S0, K_plus, T, r, q, "call")
    iv_minus = implied_vol(P_minus, S0, K_minus, T, r, q, "call")
    if not (np.isfinite(iv_atm) and np.isfinite(iv_plus) and np.isfinite(iv_minus)):
        return float("nan")
    return float(np.abs((iv_plus - iv_minus) / (2.0 * dk)))


def heston_atm_skew(
    params: HestonParameters,
    T: float,
    S0: float,
    r: float,
    q: float,
    dk: float = 5e-3,
) -> float:
    """ATM skew under a calibrated Heston model."""
    def pricer(K):
        return heston_vanilla_price(K, T, S0, r, q, params, flag="call")
    return atm_skew_from_model(S0, T, r, q, pricer, dk=dk)


def rbergomi_atm_skew(
    params: RBergomiParameters,
    T: float,
    S0: float,
    r: float,
    q: float,
    dk: float = 1e-2,
    n_paths: int = 100_000,
    n_steps: int = 200,
    seed: int = 0,
) -> float:
    """ATM skew under calibrated rBergomi.

    Uses a single MC simulation with three strike payoffs to share path noise —
    this is critical: if we ran three independent simulations the central
    difference would be drowned by MC variance.
    """
    from volengine.models.rbergomi.pricing import simulate_rbergomi
    S = simulate_rbergomi(S0, T, params, r, q, n_paths, n_steps, seed=seed)
    ST = S[:, -1]
    F = S0 * np.exp((r - q) * T)
    Ks = np.array([F * np.exp(-dk), F, F * np.exp(dk)])
    disc = np.exp(-r * T)
    prices = disc * np.maximum(ST[:, None] - Ks[None, :], 0.0).mean(axis=0)
    ivs = [implied_vol(float(p), S0, float(K), T, r, q, "call") for p, K in zip(prices, Ks)]
    if not all(np.isfinite(v) for v in ivs):
        return float("nan")
    return float(np.abs((ivs[2] - ivs[0]) / (2.0 * dk)))


def fit_skew_power_law(
    maturities: np.ndarray,
    skew: np.ndarray,
) -> tuple[float, float]:
    """Fit log psi(T) = c + alpha log T. Returns (alpha, c). Implied H = alpha + 0.5.

    Filter out NaN / non-positive points first. Returns NaN if too few points.
    """
    maturities = np.asarray(maturities, dtype=float)
    skew = np.asarray(skew, dtype=float)
    mask = np.isfinite(skew) & (skew > 0) & (maturities > 0)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    log_T = np.log(maturities[mask])
    log_psi = np.log(skew[mask])
    A = np.vstack([log_T, np.ones_like(log_T)]).T
    sol, *_ = np.linalg.lstsq(A, log_psi, rcond=None)
    alpha, c = float(sol[0]), float(sol[1])
    return alpha, c
