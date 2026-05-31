"""Variance swap pricing under Heston, rBergomi, and via model-free replication.

A variance swap pays at maturity T:

    Notional * ( RV(0, T) - K_var )

where RV is realized variance over [0, T] and K_var is the variance strike.
The fair strike is

    K_var = E^Q [ (1/T) integral_0^T V_s ds ].

Under Heston this expectation is analytic (the variance process is a CIR
process whose mean has a closed form). Under rBergomi it has a closed form
*directly in terms of the forward variance curve*:

    K_var^rBergomi = (1/T) integral_0^T xi0(s) ds

— elegant, and one of rBergomi's selling points. Under both models we can
also cross-check against the model-free Demeterfi-Derman-Kamal-Zou (1999)
static replication formula:

    K_var^repl = (2 e^{rT} / T) * [ int_0^F P(K)/K^2 dK + int_F^infty C(K)/K^2 dK ]

evaluated on the model's vanilla prices. The e^{rT} factor undoes the e^{-rT}
discounting baked into present-value option prices: the replication weights
forward (undiscounted) option payoffs, so discounted prices must be grown
back to time T. Centering the strip on the forward K* = F makes the
deterministic log-contract correction terms vanish.
"""

from __future__ import annotations

import numpy as np

from volengine.models.heston import HestonParameters
from volengine.models.rbergomi import RBergomiParameters

# NumPy 2.0 removed the long-deprecated np.trapz alias; np.trapezoid is its
# canonical replacement. Fall back to np.trapz on older NumPys for portability.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
if _trapezoid is None:  # pragma: no cover — only fires on a broken NumPy
    raise ImportError("NumPy is missing both `trapezoid` and `trapz`.")


def variance_swap_rate_heston(
    params: HestonParameters,
    T: float,
) -> float:
    """Fair variance strike under Heston.

    Closed form: E[(1/T) int_0^T V_s ds] = theta + (v0 - theta) * (1 - e^{-kappa T}) / (kappa T).
    """
    kappa, theta, _, _, v0 = params.as_tuple()
    if kappa <= 0:
        return float(v0)
    return float(theta + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / (kappa * T))


def variance_swap_rate_rbergomi(
    params: RBergomiParameters,
    T: float,
    n_quad: int = 200,
) -> float:
    """Fair variance strike under rBergomi: simple integral of forward variance.

    Uses trapezoid quadrature. With a flat xi0 this is just xi0; we keep the
    quadrature so a callable forward-variance curve works too.
    """
    t = np.linspace(1e-8, T, n_quad)
    xi0 = params.forward_variance(t)
    return float(_trapezoid(xi0, t) / T)


def variance_swap_rate_replication(
    price_call_fn,
    price_put_fn,
    S0: float,
    F: float,
    T: float,
    K_grid: np.ndarray,
    r: float,
) -> float:
    """Model-free static replication (Demeterfi-Derman-Kamal-Zou 1999).

        K_var ~= (2 e^{rT} / T) * int [ puts(K) below F + calls(K) above F ] / K^2 dK

    Parameters
    ----------
    price_call_fn : (K, T) -> *discounted* (present-value) call price.
    price_put_fn  : (K, T) -> *discounted* (present-value) put price.
    S0, F : spot and forward (F = S0 e^{(r-q)T}).
    T : maturity.
    K_grid : ascending strikes covering wings well into OTM territory.
    r : risk-free rate. Required for the e^{rT} growth factor that converts
        discounted option prices back to forward (undiscounted) payoffs — the
        replication weights time-T payoffs, not present values. Omitting this
        factor biases the strike low by exactly e^{rT}.

    Returns
    -------
    Variance swap fair strike from the replication formula.

    Notes
    -----
    The strip is centered on K* = F, which makes the deterministic log-contract
    correction terms vanish; only the option integral remains.
    """
    K_grid = np.asarray(K_grid, dtype=float)
    below = K_grid < F
    above = ~below
    # Trapezoid integration on each side.
    prices = np.empty_like(K_grid)
    if np.any(below):
        prices[below] = price_put_fn(K_grid[below], T)
    if np.any(above):
        prices[above] = price_call_fn(K_grid[above], T)
    integrand = prices / K_grid**2
    integral = _trapezoid(integrand, K_grid)
    return float(2.0 * np.exp(r * T) / T * integral)
