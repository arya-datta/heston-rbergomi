"""Discount and dividend term-structure curves.

The pricers in this library take scalar continuously-compounded rates `r` and
`q` *to a single maturity T*. A real desk does not have a flat rate: the
risk-free curve (SOFR/OIS) and the implied dividend curve both have term
structure, and using the wrong rate for a given maturity biases the forward
F = S0 e^{(r-q)T} and the discount factor e^{-rT}, which in turn biases the
inverted implied vols — most visibly at the long end.

This module provides a minimal curve abstraction:

- `FlatCurve(rate)`            : constant zero rate (the legacy behaviour).
- `ZeroCurve(times, zeros)`    : piecewise-linear continuously-compounded zero
                                 rates, flat-extrapolated beyond the knots.
- `as_curve(x)`                : coerce a float OR a Curve to a Curve, so call
                                 sites accept either transparently.

The same classes serve the risk-free curve and the dividend-yield curve — a
dividend yield is just a continuously-compounded "rate" on the carry side.

A curve exposes the continuously-compounded zero rate to maturity T,
`zero_rate(T)`, and the discount factor `discount(T) = exp(-zero_rate(T) * T)`.
Calibrators and the data pipeline call `zero_rate(T)` once per maturity, so
adopting curves costs nothing for flat-rate users (a `FlatCurve` returns the
same scalar everywhere) while enabling true term structure where it matters.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Curve(Protocol):
    """A continuously-compounded zero-rate term-structure curve."""

    def zero_rate(self, T: float) -> float:
        """Continuously-compounded zero rate to maturity T (years)."""
        ...

    def discount(self, T: float) -> float:
        """Discount factor P(0, T) = exp(-zero_rate(T) * T)."""
        ...


class FlatCurve:
    """Constant zero rate — the legacy scalar-rate behaviour."""

    def __init__(self, rate: float) -> None:
        self.rate = float(rate)

    def zero_rate(self, T: float) -> float:  # noqa: ARG002 - flat by construction
        return self.rate

    def discount(self, T: float) -> float:
        return float(np.exp(-self.rate * T))

    def forward_rate(self, T1: float, T2: float) -> float:
        """Continuously-compounded forward rate over [T1, T2] (== rate, flat)."""
        return self.rate

    def __repr__(self) -> str:
        return f"FlatCurve(rate={self.rate:.4%})"


class ZeroCurve:
    """Piecewise-linear continuously-compounded zero-rate curve.

    Parameters
    ----------
    times : ascending maturities (years) at which zero rates are quoted.
    zero_rates : continuously-compounded zero rates at those maturities.

    Interpolation is linear in the zero rate; extrapolation is flat (the
    nearest endpoint rate). Linear-in-zero is the simplest choice that keeps
    discount factors monotone for an upward curve; production desks often
    prefer linear-in-(rate*T) (i.e. log-discount) — switch `_interp` if needed.
    """

    def __init__(self, times: np.ndarray, zero_rates: np.ndarray) -> None:
        t = np.asarray(times, dtype=float)
        z = np.asarray(zero_rates, dtype=float)
        if t.ndim != 1 or t.shape != z.shape:
            raise ValueError("times and zero_rates must be 1-D and the same length.")
        if np.any(np.diff(t) <= 0):
            raise ValueError("times must be strictly ascending.")
        self.times = t
        self.zeros = z

    def zero_rate(self, T: float) -> float:
        # np.interp already flat-extrapolates beyond the endpoints.
        return float(np.interp(T, self.times, self.zeros))

    def discount(self, T: float) -> float:
        return float(np.exp(-self.zero_rate(T) * T))

    def forward_rate(self, T1: float, T2: float) -> float:
        """Continuously-compounded forward rate over [T1, T2].

            f(T1, T2) = [z(T2) T2 - z(T1) T1] / (T2 - T1)
        """
        if T2 <= T1:
            raise ValueError("T2 must exceed T1.")
        zt1 = self.zero_rate(T1) * T1
        zt2 = self.zero_rate(T2) * T2
        return float((zt2 - zt1) / (T2 - T1))

    def __repr__(self) -> str:
        return f"ZeroCurve({len(self.times)} knots, {self.times[0]:.2f}-{self.times[-1]:.2f}y)"


def as_curve(x: float | Curve) -> Curve:
    """Coerce a float (flat rate) or an existing Curve into a Curve.

    Lets call sites accept `r: float | Curve` and treat it uniformly:

        r_curve = as_curve(r)
        r_T = r_curve.zero_rate(T)
    """
    if isinstance(x, (int, float)):
        return FlatCurve(float(x))
    if hasattr(x, "zero_rate"):
        return x  # already a Curve
    raise TypeError(f"Cannot interpret {x!r} as a rate curve.")
