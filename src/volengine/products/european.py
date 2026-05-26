"""Convenience pricers for European vanillas under both models.

Both functions return *prices*. Convert to implied vol with surfaces.implied_vol.
"""

from __future__ import annotations

import numpy as np

from volengine.models.heston import HestonParameters, heston_vanilla_price
from volengine.models.rbergomi import RBergomiParameters, rbergomi_price


def european_price_heston(
    K: float | np.ndarray,
    T: float,
    S0: float,
    r: float,
    q: float,
    params: HestonParameters,
    flag: str = "call",
) -> float | np.ndarray:
    """European call/put under Heston via Carr-Madan FFT."""
    return heston_vanilla_price(K, T, S0, r, q, params, flag=flag)


def european_price_rbergomi(
    K: float | np.ndarray,
    T: float,
    S0: float,
    r: float,
    q: float,
    params: RBergomiParameters,
    flag: str = "call",
    n_paths: int = 50_000,
    n_steps: int = 100,
    seed: int | None = None,
) -> float | np.ndarray:
    """European call/put under rough Bergomi via Monte Carlo (hybrid scheme)."""
    return rbergomi_price(K, T, S0, r, q, params,
                          n_paths=n_paths, n_steps=n_steps, seed=seed, flag=flag)
