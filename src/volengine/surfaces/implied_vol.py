"""Black-Scholes pricing and implied volatility inversion.

The Black-Scholes formula for a European call on a stock paying continuous
dividend yield q is

    C(S, K, T, r, q, sigma) = S e^{-qT} N(d1) - K e^{-rT} N(d2),

with d1 = [log(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)) and
d2 = d1 - sigma sqrt(T). Puts follow by parity.

Implied vol is recovered by Brent's method on the monotone (in sigma) call
price. Brent is preferred over Newton because vega vanishes deep ITM/OTM and
near expiry, making Newton iterations unstable on the wings of the surface.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptionFlag = Literal["call", "put"]


def black_scholes_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    q: float,
    sigma: float | np.ndarray,
    flag: OptionFlag = "call",
) -> float | np.ndarray:
    """Black-Scholes European option price under continuous dividend yield.

    Parameters
    ----------
    S, K, T : spot, strike, time-to-maturity in years.
    r, q    : continuously compounded risk-free rate and dividend yield.
    sigma   : Black-Scholes volatility (annualized).
    flag    : "call" or "put".

    Returns
    -------
    Option price. Vectorizes over array-valued inputs.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Guard against T == 0 / sigma == 0 by falling back to intrinsic value.
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    intrinsic_call = np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    call = np.where((T <= 0) | (sigma <= 0), intrinsic_call, call)

    if flag == "call":
        return float(call) if call.ndim == 0 else call
    # Put-call parity: P = C - S e^{-qT} + K e^{-rT}
    put = call - S * np.exp(-q * T) + K * np.exp(-r * T)
    return float(put) if put.ndim == 0 else put


def black_scholes_vega(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """Vega = dC/dsigma. Same for calls and puts."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    return S * math.exp(-q * T) * norm.pdf(d1) * sqrtT


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    flag: OptionFlag = "call",
    vol_lo: float = 1e-4,
    vol_hi: float = 5.0,
) -> float:
    """Invert Black-Scholes to recover implied volatility.

    Uses Brent's method on [vol_lo, vol_hi]. If the target price is below
    intrinsic (arbitrageable) or above the no-arb upper bound, returns NaN
    rather than raising — calibration code skips NaNs naturally.

    Parameters
    ----------
    price : observed market price.
    S, K, T, r, q : usual contract / market inputs.
    flag : "call" or "put".
    vol_lo, vol_hi : Brent bracket. 5.0 is enormous (500%) but cheap and safe.

    Returns
    -------
    Implied volatility, or NaN if no root in the bracket.
    """
    if T <= 0 or price <= 0:
        return float("nan")

    fwd_disc = S * math.exp(-q * T)
    strike_disc = K * math.exp(-r * T)
    if flag == "call":
        lower = max(fwd_disc - strike_disc, 0.0)
        upper = fwd_disc
    else:
        lower = max(strike_disc - fwd_disc, 0.0)
        upper = strike_disc

    # Arbitrage check with a tiny tolerance for noisy quotes.
    if price < lower - 1e-8 or price > upper + 1e-8:
        return float("nan")

    def objective(sigma: float) -> float:
        return float(black_scholes_price(S, K, T, r, q, sigma, flag) - price)

    try:
        f_lo = objective(vol_lo)
        f_hi = objective(vol_hi)
        if f_lo * f_hi > 0:
            return float("nan")
        return float(brentq(objective, vol_lo, vol_hi, xtol=1e-8, maxiter=100))
    except (ValueError, RuntimeError):
        return float("nan")
