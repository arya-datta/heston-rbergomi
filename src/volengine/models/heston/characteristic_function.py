"""Heston characteristic function — Albrecher's "Little Heston Trap" form.

The Heston (1993) characteristic function of log-spot ln S_T has the form

    phi(u; T) = exp{ C(u, T) + D(u, T) v0 + i u ln(S_0 e^{(r-q) T}) }

Two algebraically equivalent forms exist. The original Heston (1993) form
suffers a branch-cut bug: as T grows, the complex logarithm in C wraps around,
producing discontinuous prices. Albrecher, Mayer, Schoutens, Tistaert (2007)
showed the "little trap" form (a sign flip inside g) is identical in value but
stays continuous across the complex plane. We use the trap form unconditionally
— it costs nothing and removes a class of silent bugs that would otherwise
appear only at long maturities or extreme strikes.
"""

from __future__ import annotations

import numpy as np

from volengine.models.heston.parameters import HestonParameters


def heston_char_fn(
    u: complex | np.ndarray,
    T: float,
    S0: float,
    r: float,
    q: float,
    params: HestonParameters,
) -> complex | np.ndarray:
    """Heston characteristic function of ln S_T evaluated at complex argument u.

    Parameters
    ----------
    u : complex frequency argument (scalar or array — vectorizes naturally).
    T : maturity in years (> 0).
    S0, r, q : spot, risk-free rate, dividend yield.
    params : HestonParameters.

    Returns
    -------
    phi(u; T) as a (possibly array-valued) complex number.

    Notes
    -----
    The choice of complex-square-root branch is the standard principal branch
    used by NumPy. The "little trap" form expresses C with g2 = (b - d)/(b + d)
    rather than the original g1 = (b + d)/(b - d), avoiding the discontinuity.
    """
    kappa, theta, xi, rho, v0 = params.as_tuple()
    iu = 1j * u

    # Standard Heston discriminant.
    b = kappa - rho * xi * iu
    d = np.sqrt(b**2 + xi**2 * (iu + u**2))

    # 'Little trap' form: use g2 = (b - d) / (b + d) instead of (b + d)/(b - d).
    g2 = (b - d) / (b + d)
    exp_dT = np.exp(-d * T)

    C = (
        (r - q) * iu * T
        + (kappa * theta / xi**2)
        * ((b - d) * T - 2.0 * np.log((1.0 - g2 * exp_dT) / (1.0 - g2)))
    )
    D = (b - d) / xi**2 * (1.0 - exp_dT) / (1.0 - g2 * exp_dT)

    return np.exp(C + D * v0 + iu * np.log(S0))
