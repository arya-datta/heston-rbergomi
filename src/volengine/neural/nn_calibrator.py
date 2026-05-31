"""Neural-network calibrator for rBergomi (Horvath-Muguruza-Tomas 2019).

Trains a small fully-connected MLP that maps an observed IV grid -> the
4D rBergomi parameter vector. Inference is then milliseconds, compared to
~30-60 s for the full traditional calibration. Imports are guarded so the
rest of the library does not require torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from volengine.models.rbergomi import RBergomiParameters
from volengine.neural.data_generation import TrainingGrid, generate_training_data

if TYPE_CHECKING:  # pragma: no cover
    import torch
    import torch.nn as nn

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


@dataclass
class NeuralCalibrator:
    """Trained MLP for rBergomi calibration. Use `train_calibrator` to build one."""

    model: nn.Module
    grid: TrainingGrid
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def calibrate(self, iv_grid: np.ndarray) -> RBergomiParameters:
        """Invert: observed IVs on `self.grid` -> rBergomi parameters."""
        if not _HAS_TORCH:
            raise RuntimeError("torch is not installed.")
        with torch.no_grad():
            y = (iv_grid.reshape(-1) - self.y_mean) / self.y_std
            yt = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
            xt = self.model(yt).squeeze(0).numpy()
            x = xt * self.x_std + self.x_mean
        return RBergomiParameters(H=float(x[0]), eta=float(x[1]),
                                  rho=float(x[2]), xi0=float(x[3]))


def train_calibrator(
    grid: TrainingGrid,
    n_samples: int = 5000,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    **data_kwargs,
) -> NeuralCalibrator:
    """Generate training data and fit an MLP price -> param inverter.

    Returns
    -------
    Trained NeuralCalibrator. Call `.calibrate(iv_grid)` to invert.
    """
    if not _HAS_TORCH:
        raise RuntimeError(
            "torch is not installed. Install via `pip install volengine[neural]`."
        )

    X, Y = generate_training_data(grid, n_samples=n_samples, seed=seed, **data_kwargs)
    mask = np.all(np.isfinite(Y), axis=1)
    X, Y = X[mask], Y[mask]

    x_mean, x_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    y_mean, y_std = Y.mean(axis=0), Y.std(axis=0) + 1e-8
    X_norm = (X - x_mean) / x_std
    Y_norm = (Y - y_mean) / y_std

    n_in = Y.shape[1]
    n_out = X.shape[1]
    model = nn.Sequential(
        nn.Linear(n_in, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, n_out),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_norm, dtype=torch.float32)
    Y_t = torch.tensor(Y_norm, dtype=torch.float32)
    n = len(X_t)

    for _epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            pred = model(Y_t[idx])
            loss = loss_fn(pred, X_t[idx])
            loss.backward()
            opt.step()

    return NeuralCalibrator(model=model, grid=grid,
                            x_mean=x_mean, x_std=x_std,
                            y_mean=y_mean, y_std=y_std)
