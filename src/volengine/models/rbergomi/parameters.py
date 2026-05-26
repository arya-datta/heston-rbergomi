"""Rough Bergomi parameters.

Bayer-Friz-Gatheral (2016) rBergomi model under risk-neutral measure Q:

    dS_t / S_t = sqrt(V_t) dW_t,                 (rate / div omitted; add via numeraire)
    V_t       = xi0(t) * exp( eta * sqrt(2H) * Y_t  -  0.5 * eta^2 * t^{2H} ),
    Y_t       = integral_0^t (t - s)^{H - 1/2} dZ_s        (Volterra Brownian),
    d<W, Z>_t = rho dt.

Three calibration parameters: H in (0, 0.5) (roughness — empirical SPX has
H ~ 0.07-0.15), eta > 0 (vol-of-vol scale), rho in (-1, 1) (leverage). The
fourth input xi0 is the *forward variance curve* — typically read from the
ATM term structure rather than calibrated parameter-by-parameter. Here we
parameterize xi0 as a flat level for simplicity but the design admits a
piecewise-constant or interpolated curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class RBergomiParameters:
    """Rough Bergomi parameters with a callable forward-variance curve.

    Attributes
    ----------
    H : Hurst-style roughness exponent in (0, 0.5). SPX-empirical ~0.1.
    eta : vol-of-vol scaling. > 0. Typical SPX ~1.5-2.5.
    rho : spot/vol correlation in (-1, 1). Typical equity ~-0.7 to -0.95.
    xi0 : forward variance curve. Either a float (flat) or a callable t -> xi0(t).
    """

    H: float
    eta: float
    rho: float
    xi0: float | Callable[[np.ndarray], np.ndarray]

    def forward_variance(self, t: np.ndarray) -> np.ndarray:
        """Evaluate xi0(t). Returns an array shaped like t."""
        if callable(self.xi0):
            return np.asarray(self.xi0(t), dtype=float)
        return np.full_like(np.asarray(t, dtype=float), float(self.xi0))

    def as_tuple(self) -> tuple[float, float, float, float | Callable]:
        return (self.H, self.eta, self.rho, self.xi0)


RBERGOMI_BOUNDS = {
    "H":   (0.02, 0.49),
    "eta": (0.1, 5.0),
    "rho": (-0.999, 0.0),    # equity skew demands rho < 0 in practice
    "xi0": (1e-6, 1.0),
}
