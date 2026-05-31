"""Forward variance curve xi0(t) for rough Bergomi.

In rough Bergomi the forward variance curve

    xi0(t) = E^Q[ V_t ]

is an *input*, read off the market term structure of variance — not a fitted
scalar. This is one of the model's defining features: only the three dynamics
parameters (H, eta, rho) are calibrated; the level structure is pinned exactly
by the market. Collapsing xi0 to a flat number (as the basic calibrator does)
discards that structure and forces the dynamics parameters to compensate.

This module provides:

- `ForwardVarianceCurve`            : a callable t -> xi0(t), piecewise-constant
                                      or linearly interpolated, flat-extrapolated.
- `ForwardVarianceCurve.from_atm_term_structure(maturities, atm_vols)` :
                                      build xi0 from the ATM implied-vol term
                                      structure via the total-variance derivative
                                      xi0(t) = d/dT [ sigma_atm(T)^2 * T ].

A `ForwardVarianceCurve` is accepted directly as the `xi0` field of
`RBergomiParameters` (which already supports a callable), so

    p = RBergomiParameters(H=0.1, eta=1.9, rho=-0.9, xi0=curve)

simulates with the full term structure preserved by the martingale correction.
"""

from __future__ import annotations

import numpy as np


class ForwardVarianceCurve:
    """Callable forward-variance curve xi0(t), floored at a small positive value.

    Parameters
    ----------
    times : ascending knot maturities (years).
    fwd_var : forward variance xi0 at each knot (>= 0).
    kind : "piecewise" (right-continuous step, the BFG convention) or "linear".
    floor : minimum returned variance, to keep V_t well-defined.
    """

    def __init__(
        self,
        times: np.ndarray,
        fwd_var: np.ndarray,
        kind: str = "piecewise",
        floor: float = 1e-8,
    ) -> None:
        t = np.asarray(times, dtype=float)
        v = np.asarray(fwd_var, dtype=float)
        if t.ndim != 1 or t.shape != v.shape:
            raise ValueError("times and fwd_var must be 1-D and the same length.")
        if np.any(np.diff(t) <= 0):
            raise ValueError("times must be strictly ascending.")
        if kind not in ("piecewise", "linear"):
            raise ValueError("kind must be 'piecewise' or 'linear'.")
        self.times = t
        self.fwd_var = np.maximum(v, floor)
        self.kind = kind
        self.floor = floor

    def __call__(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if self.kind == "linear":
            out = np.interp(t, self.times, self.fwd_var)  # flat-extrapolated
        else:
            # Piecewise-constant, right-continuous: xi0(t) takes the knot value
            # of the first knot >= t (clamped to the last knot).
            idx = np.searchsorted(self.times, t, side="left")
            idx = np.clip(idx, 0, len(self.times) - 1)
            out = self.fwd_var[idx]
        return np.maximum(out, self.floor)

    @classmethod
    def from_atm_term_structure(
        cls,
        maturities: np.ndarray,
        atm_vols: np.ndarray,
        kind: str = "linear",
    ) -> ForwardVarianceCurve:
        """Build xi0(t) from the ATM implied-vol term structure.

        The ATM total variance is w(T) = sigma_atm(T)^2 * T. The instantaneous
        forward variance is its maturity-derivative,

            xi0(t) = dw/dT |_{T=t},

        approximated here by finite differences of the (T, w) term structure.
        The knot points are placed at the midpoints of the maturity intervals,
        where the forward variance is best represented.

        Parameters
        ----------
        maturities : ascending maturities (years) of the ATM term structure.
        atm_vols : ATM implied vols at those maturities.
        kind : interpolation for the resulting curve.

        Returns
        -------
        ForwardVarianceCurve.
        """
        T = np.asarray(maturities, dtype=float)
        sig = np.asarray(atm_vols, dtype=float)
        if T.shape != sig.shape or T.ndim != 1 or len(T) < 2:
            raise ValueError("Need >= 2 aligned (maturity, atm_vol) points.")
        order = np.argsort(T)
        T, sig = T[order], sig[order]
        w = sig**2 * T                      # ATM total variance
        # Forward variance on each interval = slope of w; place at interval mid.
        fwd = np.diff(w) / np.diff(T)
        t_mid = 0.5 * (T[:-1] + T[1:])
        # Prepend the first interval's level at t=0 so short maturities are
        # covered, and keep it non-negative.
        t_knots = np.concatenate([[0.0], t_mid])
        v_knots = np.concatenate([[fwd[0]], fwd])
        return cls(t_knots, np.maximum(v_knots, 1e-8), kind=kind)

    def mean_variance(self, T: float, n_quad: int = 200) -> float:
        """(1/T) integral_0^T xi0(s) ds — the model-implied variance-swap rate."""
        s = np.linspace(1e-8, T, n_quad)
        return float(np.trapezoid(self(s), s) / T)

    def __repr__(self) -> str:
        return (f"ForwardVarianceCurve({len(self.times)} knots, kind={self.kind}, "
                f"xi0[0]={self.fwd_var[0]:.4f})")
