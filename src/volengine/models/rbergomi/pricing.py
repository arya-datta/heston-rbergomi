"""Rough Bergomi simulation and vanilla pricing.

Given the Volterra Y_t from the hybrid scheme and the driving Brownian Z_t,
the rBergomi instantaneous variance and spot process are:

    V_t = xi0(t) * exp[ eta * sqrt(2 H) * Y_t  -  0.5 * eta^2 * t^{2H} ],
    dS_t / S_t = sqrt(V_t) [ rho dZ_t + sqrt(1 - rho^2) dW_t^perp ],

where dW^perp is independent of Z. The martingale correction
-0.5 * eta^2 * t^{2H} = -0.5 * Var(eta sqrt(2H) Y_t) keeps E[V_t] = xi0(t),
so that the forward variance curve is preserved by construction — a key
feature of rBergomi: xi0 is read off the market, not calibrated.

European vanillas are then priced by straightforward Monte Carlo on S_T.
"""

from __future__ import annotations

import numpy as np

from volengine.models.rbergomi.hybrid_scheme import HybridScheme
from volengine.models.rbergomi.parameters import RBergomiParameters


def simulate_rbergomi(
    S0: float,
    T: float,
    params: RBergomiParameters,
    r: float,
    q: float,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    antithetic: bool = True,
    return_variance: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Simulate spot paths under rough Bergomi.

    Parameters
    ----------
    S0, T : spot and maturity.
    params : RBergomiParameters (H, eta, rho, xi0).
    r, q  : risk-free rate and dividend yield (continuous compounding).
    n_paths, n_steps : MC dimensions. Memory cost is O(n_paths * n_steps).
    seed : optional RNG seed.
    antithetic : use antithetic variates (recommended).
    return_variance : if True, also return the instantaneous variance paths.

    Returns
    -------
    S : ndarray, shape (n_paths, n_steps + 1).
    V : same shape (returned only if return_variance is True).
    """
    rng = np.random.default_rng(seed)
    scheme = HybridScheme(H=params.H, T=T, n_steps=n_steps, kappa=1)
    Y, Z = scheme.simulate(n_paths, rng=rng, antithetic=antithetic)
    n_paths_eff = Y.shape[0]

    t_grid = np.linspace(0.0, T, n_steps + 1)
    xi0_t = params.forward_variance(t_grid)
    # Martingale correction so that E[V_t] = xi0(t):
    correction = 0.5 * params.eta**2 * (t_grid ** (2.0 * params.H))
    V = xi0_t[None, :] * np.exp(params.eta * np.sqrt(2.0 * params.H) * Y - correction[None, :])

    # Build the orthogonal Brownian W^perp, then form dW_S = rho dZ + sqrt(1-rho^2) dW^perp.
    dW_perp = rng.standard_normal((n_paths_eff, n_steps)) * np.sqrt(T / n_steps)
    if antithetic:
        # Match antithetic structure of Z: dW^perp must use the same pairing
        # to preserve the variance reduction. Re-randomize one half to match.
        half = n_paths_eff // 2
        dW_perp = np.concatenate([dW_perp[:half], -dW_perp[:half]], axis=0)

    dZ = np.diff(Z, axis=1)  # shape (n_paths_eff, n_steps)
    dW_S = params.rho * dZ + np.sqrt(1.0 - params.rho**2) * dW_perp

    # Log-spot Euler-Maruyama (exact for this geometric-Brownian step under
    # the conditional measure since V is held constant within each dt):
    dt = T / n_steps
    log_increments = (r - q - 0.5 * V[:, :-1]) * dt + np.sqrt(np.maximum(V[:, :-1], 0.0)) * dW_S
    log_S = np.log(S0) + np.concatenate(
        [np.zeros((n_paths_eff, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    S = np.exp(log_S)
    if return_variance:
        return S, V
    return S


def rbergomi_price(
    K: float | np.ndarray,
    T: float,
    S0: float,
    r: float,
    q: float,
    params: RBergomiParameters,
    n_paths: int = 50_000,
    n_steps: int = 100,
    seed: int | None = None,
    flag: str = "call",
) -> float | np.ndarray:
    """MC price of European vanillas under rBergomi.

    Returns
    -------
    Price (scalar or array, matching shape of K).
    """
    S = simulate_rbergomi(S0, T, params, r, q, n_paths, n_steps, seed=seed)
    ST = S[:, -1]
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    disc = np.exp(-r * T)
    if flag == "call":
        payoffs = np.maximum(ST[:, None] - K_arr[None, :], 0.0)
    elif flag == "put":
        payoffs = np.maximum(K_arr[None, :] - ST[:, None], 0.0)
    else:
        raise ValueError(f"flag must be 'call' or 'put', got {flag!r}")
    price = disc * payoffs.mean(axis=0)
    return float(price[0]) if np.isscalar(K) else price
