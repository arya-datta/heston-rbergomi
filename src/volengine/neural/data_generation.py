"""Generate synthetic (params -> IV surface) training data for a NN calibrator.

We sample rBergomi parameters (H, eta, rho, xi0) uniformly within calibration
bounds, then for each draw compute IVs across a fixed (maturity x moneyness)
grid via MC simulation. The output is a (n_samples, n_T * n_k) array of IVs
paired with the (n_samples, 4) parameter array. A neural network can then
learn the inverse map IV-grid -> params.

A real production training pipeline would:
  - Use stratified sampling within bounds, not uniform.
  - Use a control-variate or quasi-MC scheme for noise reduction.
  - Pre-train on Heston (analytic, cheap) then fine-tune on rBergomi.
  - Apply standardization / log-transforms to targets for stability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from volengine.models.rbergomi import RBergomiParameters, simulate_rbergomi
from volengine.models.rbergomi.parameters import RBERGOMI_BOUNDS
from volengine.surfaces.implied_vol import implied_vol


@dataclass
class TrainingGrid:
    """Fixed grid on which IVs are evaluated for every parameter draw."""
    maturities: np.ndarray
    log_moneyness: np.ndarray  # k = log(K / F)

    @property
    def n_features(self) -> int:
        return len(self.maturities) * len(self.log_moneyness)


def generate_training_data(
    grid: TrainingGrid,
    n_samples: int,
    S0: float = 100.0,
    r: float = 0.0,
    q: float = 0.0,
    n_paths: int = 10_000,
    n_steps_per_year: int = 80,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (params, iv_grid) training pairs.

    Returns
    -------
    X : (n_samples, 4) parameter array (H, eta, rho, xi0).
    Y : (n_samples, n_T * n_k) IV array, row-major (T outer, k inner).
    """
    rng = np.random.default_rng(seed)

    H_lo, H_hi = RBERGOMI_BOUNDS["H"]
    eta_lo, eta_hi = RBERGOMI_BOUNDS["eta"]
    rho_lo, rho_hi = RBERGOMI_BOUNDS["rho"]
    xi_lo, xi_hi = RBERGOMI_BOUNDS["xi0"]

    X = np.column_stack([
        rng.uniform(H_lo, H_hi, n_samples),
        rng.uniform(eta_lo, eta_hi, n_samples),
        rng.uniform(rho_lo, rho_hi, n_samples),
        rng.uniform(xi_lo, xi_hi, n_samples),
    ])

    n_T = len(grid.maturities)
    n_k = len(grid.log_moneyness)
    Y = np.full((n_samples, n_T * n_k), np.nan)

    for i in range(n_samples):
        params = RBergomiParameters(H=X[i, 0], eta=X[i, 1], rho=X[i, 2], xi0=X[i, 3])
        for j, T in enumerate(grid.maturities):
            n_steps = max(20, int(np.ceil(n_steps_per_year * T)))
            S = simulate_rbergomi(S0, T, params, r, q, n_paths, n_steps,
                                  seed=int(seed + i * 1009 + j))
            ST = S[:, -1]
            F = S0 * np.exp((r - q) * T)
            for l, k in enumerate(grid.log_moneyness):
                K = F * np.exp(k)
                price = np.exp(-r * T) * np.maximum(ST - K, 0.0).mean()
                iv = implied_vol(float(price), S0, K, T, r, q, "call")
                Y[i, j * n_k + l] = iv

    return X, Y
