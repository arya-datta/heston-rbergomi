"""Carr-Madan (1999) FFT pricing for vanilla calls under Heston.

The Carr-Madan transform writes the modified call price as

    c_T(k) = e^{-alpha k} C_T(k),    k = log strike,

then takes its Fourier transform analytically:

    psi(v) = e^{-r T} phi(v - i (alpha + 1); T) / [alpha^2 + alpha - v^2 + i (2 alpha + 1) v]

where phi is the characteristic function of the *log spot*. Inverting psi via
an FFT gives prices on a uniform log-strike grid in one shot.

The damping parameter alpha must satisfy E[S_T^{alpha+1}] < infinity. For Heston
alpha in [0.75, 1.5] is the usual practitioner default; alpha = 1.5 works well
for SPX-scale parameters.
"""

from __future__ import annotations

import numpy as np

from volengine.models.heston.characteristic_function import heston_char_fn
from volengine.models.heston.parameters import HestonParameters


def carr_madan_price(
    strikes: np.ndarray,
    T: float,
    S0: float,
    r: float,
    q: float,
    params: HestonParameters,
    alpha: float = 1.5,
    N: int = 8192,
    eta: float = 0.15,
) -> np.ndarray:
    """FFT-priced European calls on a uniform log-strike grid, interpolated to `strikes`.

    Parameters
    ----------
    strikes : strikes at which to return prices.
    T, S0, r, q : maturity, spot, rates.
    params : HestonParameters.
    alpha : Carr-Madan damping coefficient (must make E[S^{alpha+1}] finite).
    N     : FFT length (power of 2 for speed). 8192 hits 4-decimal-place
            agreement with direct numerical integration on SPX-scale params;
            4096 is fine for surface-level calibration but borderline for
            single-strike spot checks against `scipy.integrate.quad`.
    eta   : Fourier-space grid spacing. Smaller eta = finer log-strike grid but
            shorter Fourier range. lambda = 2 pi / (N eta).

    Returns
    -------
    Array of call prices, one per input strike.

    Notes
    -----
    Carr-Madan returns prices on log-strikes k_j = -b + j * lambda where
    lambda = 2 pi / (N eta) and b = N lambda / 2. We linearly interpolate from
    that grid to the user-requested strikes. For higher-accuracy work on a few
    strikes the brute-force integral is sometimes preferable; FFT wins
    decisively when pricing the whole calibration surface.
    """
    lam = 2.0 * np.pi / (N * eta)
    b = N * lam / 2.0
    u = np.arange(N) * eta

    # Simpson-weighted Carr-Madan integrand.
    phi = heston_char_fn(u - 1j * (alpha + 1.0), T, S0, r, q, params)
    psi = np.exp(-r * T) * phi / (alpha**2 + alpha - u**2 + 1j * (2.0 * alpha + 1.0) * u)

    simpson = (3.0 + (-1.0) ** (np.arange(N) + 1)) / 3.0
    simpson[0] = 1.0 / 3.0
    x = np.exp(1j * b * u) * psi * eta * simpson

    fft_result = np.fft.fft(x).real
    log_strikes_grid = -b + lam * np.arange(N)
    call_grid = np.exp(-alpha * log_strikes_grid) / np.pi * fft_result

    # Interpolate to the requested strikes.
    log_strikes = np.log(np.asarray(strikes, dtype=float))
    return np.interp(log_strikes, log_strikes_grid, call_grid)


def heston_vanilla_price(
    K: float | np.ndarray,
    T: float,
    S0: float,
    r: float,
    q: float,
    params: HestonParameters,
    flag: str = "call",
    **fft_kwargs,
) -> float | np.ndarray:
    """Convenience wrapper: vanilla call/put price under Heston via Carr-Madan FFT.

    Put prices come from put-call parity rather than re-running the FFT — both
    are exact under Heston since the characteristic function gives the spot
    distribution directly.
    """
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    call_prices = carr_madan_price(K_arr, T, S0, r, q, params, **fft_kwargs)
    if flag == "call":
        out = call_prices
    elif flag == "put":
        out = call_prices - S0 * np.exp(-q * T) + K_arr * np.exp(-r * T)
    else:
        raise ValueError(f"flag must be 'call' or 'put', got {flag!r}")
    return float(out[0]) if np.isscalar(K) else out
