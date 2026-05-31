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

# Tolerance for boundary validation: optimizers (L-BFGS-B with box bounds)
# legitimately sit exactly on a bound, so we validate with a small slack to
# avoid spurious ValueErrors at e.g. rho = -0.999 or b = 1e-6.
_SVI_VALIDATION_TOL = 1e-6


@dataclass(frozen=True)
class SVIParameters:
    """Raw-SVI parameters for a single maturity slice.

    Validates the structural constraints of raw SVI at construction:
      - b >= 0      (wings open upward)
      - -1 < rho < 1 (correlation parameter)
      - sigma > 0    (curvature of the smile minimum)

    These are the constraints that keep w(k) a well-defined SVI slice. The
    stronger no-butterfly-arbitrage condition (g(k) >= 0 everywhere) is NOT
    enforced here — it's checked separately via `svi_butterfly_function`,
    because a slice can be a valid SVI parameterization yet still admit
    static arbitrage, and we want to be able to construct and inspect such
    slices rather than refuse to build them.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        tol = _SVI_VALIDATION_TOL
        if self.b < -tol:
            raise ValueError(f"SVI requires b >= 0, got b={self.b}")
        if not (-1.0 - tol < self.rho < 1.0 + tol):
            raise ValueError(f"SVI requires -1 < rho < 1, got rho={self.rho}")
        if self.sigma <= -tol:
            raise ValueError(f"SVI requires sigma > 0, got sigma={self.sigma}")

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    def min_total_variance(self) -> float:
        """Minimum of w(k) over all k. Negative => the slice has negative
        total variance somewhere (a hard arbitrage), reached at the smile
        vertex. w_min = a + b*sigma*sqrt(1 - rho^2)."""
        return self.a + self.b * self.sigma * np.sqrt(max(1.0 - self.rho**2, 0.0))

    @classmethod
    def from_array(cls, x: np.ndarray) -> SVIParameters:
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
    butterfly_penalty: float = 1.0,
    initial: SVIParameters | None = None,
) -> SVIParameters:
    """Calibrate a raw-SVI slice to observed implied vols.

    Parameters
    ----------
    k : log-moneyness array (log K / F).
    iv: observed implied volatilities at those strikes.
    T : maturity in years.
    weights : optional per-point weights (default = uniform).
    butterfly_penalty : weight on the no-arbitrage soft penalty.
    initial : optional warm-start parameters.

    Returns
    -------
    Fitted SVIParameters.

    Notes
    -----
    The objective is weighted MSE in **implied-vol space** plus a soft
    butterfly-arbitrage penalty. Two deliberate choices make this robust at
    short maturities (where the naive version collapses to a flat, zero-skew
    fit):

    1. Fitting `sigma_model - sigma_target` (each O(0.2)) rather than total
       variance `w = sigma^2 T` (which is ~0.002 at one month) keeps the fit
       error on a maturity-independent scale, so it isn't swamped by the
       penalty.
    2. Penalizing `g(k) * w(k)` rather than `g(k)` removes the `1/w` blow-up in
       the no-arb function: at short T, `1/w ~ 500`, so any skew produces a huge
       spurious `g`, and a fixed penalty would force `b -> 0`. Scaling by `w`
       makes the penalty maturity-consistent. The penalty is evaluated only on
       a grid spanning the observed strikes (plus a small margin) — penalizing
       far-extrapolated regions over-constrains the wings.

    Uses L-BFGS-B with box constraints; a moment-based heuristic warm start is
    used when none is supplied.
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

    # Butterfly-penalty grid: span the observed strikes plus a small margin.
    # Penalizing far-extrapolated k over-constrains the wings and (at short T)
    # forces the skew to zero.
    k_lo, k_hi = k.min() - 0.05, k.max() + 0.05
    k_dense = np.linspace(k_lo, k_hi, 200)

    def objective(x: np.ndarray) -> float:
        p = SVIParameters.from_array(x)
        w_model = svi_total_variance(k, p)
        iv_model = np.sqrt(np.maximum(w_model, 1e-12) / T)
        fit_err = np.sum(weights * (iv_model - iv) ** 2)
        # Penalize g * w (scale-stable) instead of g (which blows up as 1/w).
        w_dense = np.maximum(svi_total_variance(k_dense, p), 1e-12)
        g = svi_butterfly_function(k_dense, p)
        butterfly = np.sum(np.minimum(g * w_dense, 0.0) ** 2)
        return float(fit_err + butterfly_penalty * butterfly)

    bounds = [
        (1e-6, 1.0),       # a >= 0 (total variance is non-negative)
        (1e-6, 5.0),       # b >= 0
        (-0.999, 0.999),   # |rho| < 1
        (-1.0, 1.0),       # m: smile vertex stays near the data range, not run
                           #    to a far bound (which collapses the fit when the
                           #    strike coverage is narrow / asymmetric)
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
