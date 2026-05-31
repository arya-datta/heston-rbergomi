"""Heston parameter container with bounds and Feller-condition diagnostics.

The Heston (1993) model under the risk-neutral measure Q:

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v
    d<W^S, W^v>_t = rho dt

with five parameters (kappa, theta, xi, rho, v0). Mean reversion speed kappa
pulls v toward long-run variance theta; xi is the vol-of-vol; rho is the
spot/vol correlation (typically negative for equity); v0 is the initial
variance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HestonParameters:
    """Five Heston parameters under risk-neutral measure.

    Attributes
    ----------
    kappa : mean reversion speed of variance (> 0).
    theta : long-run mean of variance (> 0).
    xi    : volatility of variance / 'vol-of-vol' (> 0).
    rho   : correlation between spot and variance Brownian motions (in (-1, 1)).
    v0    : initial variance (> 0).
    """

    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    def feller_condition(self) -> bool:
        """Feller: 2 kappa theta >= xi^2 keeps the variance process strictly positive.

        Calibrated equity surfaces routinely violate Feller; we don't enforce
        it as a hard constraint, just expose the diagnostic. QE simulation
        handles sub-Feller regimes by construction.
        """
        return 2.0 * self.kappa * self.theta >= self.xi**2

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.kappa, self.theta, self.xi, self.rho, self.v0)

    @classmethod
    def from_tuple(cls, t: tuple[float, ...]) -> HestonParameters:
        return cls(*t)


HESTON_BOUNDS = {
    "kappa": (1e-3, 20.0),
    "theta": (1e-6, 1.0),     # variance, not vol: 1.0 ~= 100% vol cap
    "xi":    (1e-3, 5.0),
    "rho":   (-0.999, 0.999),
    "v0":    (1e-6, 1.0),
}
"""Sensible parameter bounds for SPX-like surfaces. Used by both calibrators."""
