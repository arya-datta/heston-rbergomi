"""SVI (Stochastic Volatility Inspired) parameterization of the IV surface.

Gatheral (2004) raw SVI for total implied variance as a function of log-moneyness
k = log(K / F):

    w(k) = a + b * { rho (k - m) + sqrt[(k - m)^2 + sigma^2] }

where w = sigma_BS^2 * T is total variance. The five parameters carry the
following intuition:

    a     : vertical level of the smile
    b >= 0: overall slope (slope of wings)
    -1 < rho < 1: skew between left and right wing
    m     : horizontal shift of the smile minimum
    sigma > 0: smoothness of the minimum

We fit each maturity slice independently and check the no-butterfly-arbitrage
condition (g(k) >= 0; Gatheral & Jacquier 2014, Proposition 2.1). Cross-maturity
calendar-spread arbitrage (monotonicity of total variance in T at fixed k) is
checked at the surface level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class SVIParameters:
    """Raw-SVI parameters for a single maturity slice."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "SVIParameters":
        return cls(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])


def svi_total_variance(k: np.ndarray, p: SVIParameters) -> np.ndarray:
    """Raw-SVI total variance w(k) = sigma_BS^2 * T."""
    k = np.asarray(k, dtype=float)
    return p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma**2))


def svi_implied_vol(k: np.ndarray, T: float, p: SVIParameters) -> np.ndarray:
    """Implied vol sigma_BS(k, T) implied by an SVI slice at maturity T."""
    if T <= 0:
        raise ValueError("Maturity T must be positive.")
    w = svi_total_variance(k, p)
    # Clip to avoid sqrt of small negatives produced by butterfly violations.
    return np.sqrt(np.maximum(w, 1e-12) / T)


def svi_butterfly_function(k: np.ndarray, p: SVIParameters) -> np.ndarray:
    """g(k) per Gatheral-Jacquier (2014). Butterfly arb-free iff g >= 0 for all k.

        g(k) = [1 - k w'(k) / (2 w)]^2 - (w'(k))^2 / 4 [1/w + 1/4] + w''(k)/2

    Used as a soft penalty inside the calibration objective.
    """
    k = np.asarray(k, dtype=float)
    w = svi_total_variance(k, p)
    sqrt_term = np.sqrt((k - p.m) ** 2 + p.sigma**2)
    wp = p.b * (p.rho + (k - p.m) / sqrt_term)
    wpp = p.b * p.sigma**2 / sqrt_term**3
    safe_w = np.maximum(w, 1e-10)
    term1 = (1.0 - k * wp / (2.0 * safe_w)) ** 2
    term2 = (wp**2 / 4.0) * (1.0 / safe_w + 0.25)
    return term1 - term2 + wpp / 2.0


def fit_svi_slice(
    k: np.ndarray,
    iv: np.ndarray,
    T: float,
    weights: np.ndarray | None = None,
    butterfly_penalty: float = 1e3,
    initial: SVIParameters | None = None,
) -> SVIParameters:
    """Calibrate a raw-SVI slice to observed implied vols.

    Parameters
    ----------
    k : log-moneyness array (log K / F).
    iv: observed implied volatilities at those strikes.
    T : maturity in years.
    weights : optional per-point weights (default = uniform).
    butterfly_penalty : weight on integrated negative-g penalty.
    initial : optional warm-start parameters.

    Returns
    -------
    Fitted SVIParameters.

    Notes
    -----
    The objective is weighted MSE on total variance w = iv^2 * T plus a soft
    penalty on butterfly violations. We use L-BFGS-B with box constraints; if
    no warm start is supplied, a moment-based heuristic is used.
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    mask = np.isfinite(iv) & (iv > 0)
    k, iv = k[mask], iv[mask]
    if weights is None:
        weights = np.ones_like(iv)
    else:
        weights = np.asarray(weights, dtype=float)[mask]

    target_w = iv**2 * T
    atm_var = float(np.interp(0.0, k, target_w)) if len(k) >= 2 else 0.04 * T
    if initial is None:
        initial = SVIParameters(
            a=max(atm_var * 0.5, 1e-4),
            b=0.1,
            rho=-0.3,
            m=0.0,
            sigma=0.1,
        )

    # Dense grid for the butterfly penalty so the optimizer can't squeeze
    # negative g between two market strikes.
    k_dense = np.linspace(min(k.min(), -0.5) - 0.1, max(k.max(), 0.5) + 0.1, 200)

    def objective(x: np.ndarray) -> float:
        p = SVIParameters.from_array(x)
        w_model = svi_total_variance(k, p)
        fit_err = np.sum(weights * (w_model - target_w) ** 2)
        g = svi_butterfly_function(k_dense, p)
        butterfly = np.sum(np.minimum(g, 0.0) ** 2)
        return float(fit_err + butterfly_penalty * butterfly)

    bounds = [
        (1e-6, 1.0),       # a >= 0 (total variance is non-negative)
        (1e-6, 5.0),       # b >= 0
        (-0.999, 0.999),   # |rho| < 1
        (-2.0, 2.0),       # m within plausible log-moneyness range
        (1e-4, 5.0),       # sigma > 0
    ]
    result = minimize(
        objective,
        initial.as_array(),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return SVIParameters.from_array(result.x)


@dataclass
class SVISurface:
    """Multi-maturity SVI surface.

    Stores one SVIParameters per maturity. Provides vectorized lookup of
    implied vol at arbitrary (k, T) by interpolating total variance in T at
    each k along the slice — this preserves no-calendar-spread-arbitrage when
    the underlying slices are well-ordered in total variance.
    """

    maturities: np.ndarray
    slices: dict[float, SVIParameters]

    def implied_vol(self, k: float | np.ndarray, T: float) -> float | np.ndarray:
        """Implied vol at log-moneyness k and maturity T (linear-in-w interp)."""
        if T in self.slices:
            return svi_implied_vol(np.asarray(k), T, self.slices[T])
        Ts = np.asarray(sorted(self.slices.keys()))
        if T <= Ts[0]:
            return svi_implied_vol(np.asarray(k), Ts[0], self.slices[float(Ts[0])])
        if T >= Ts[-1]:
            return svi_implied_vol(np.asarray(k), Ts[-1], self.slices[float(Ts[-1])])
        i = int(np.searchsorted(Ts, T))
        T_lo, T_hi = float(Ts[i - 1]), float(Ts[i])
        w_lo = svi_total_variance(np.asarray(k), self.slices[T_lo])
        w_hi = svi_total_variance(np.asarray(k), self.slices[T_hi])
        alpha = (T - T_lo) / (T_hi - T_lo)
        w = (1.0 - alpha) * w_lo + alpha * w_hi
        return np.sqrt(np.maximum(w, 1e-12) / T)

    def check_calendar_arbitrage(self, k_grid: np.ndarray | None = None) -> bool:
        """Return True if total variance is monotone non-decreasing in T at every k."""
        if k_grid is None:
            k_grid = np.linspace(-0.5, 0.5, 50)
        Ts = sorted(self.slices.keys())
        prev_w = svi_total_variance(k_grid, self.slices[Ts[0]])
        for T in Ts[1:]:
            w = svi_total_variance(k_grid, self.slices[T])
            if np.any(w < prev_w - 1e-8):
                return False
            prev_w = w
        return True
