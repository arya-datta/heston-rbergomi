"""Bennedsen-Lunde-Pakkanen (2017) hybrid scheme for Brownian semistationary processes.

The Volterra process

    Y_t = integral_0^t (t - s)^{alpha} dZ_s,    alpha = H - 1/2,    alpha in (-1/2, 0)

is the singular building block of rough Bergomi. A direct Riemann-sum
discretization converges as O(n^{-1/2 - alpha}) and is hopelessly slow because
the kernel blows up at s -> t.

The hybrid scheme splits the integral into:

  (i)  a 'kappa-near' Wiener-Ito chaos block that captures the singular
       region [t - kappa dt, t] *exactly* in joint distribution, and
  (ii) a 'far' Riemann sum over [0, t - kappa dt] where the kernel is smooth.

With kappa = 1 (one near block) the scheme converges at the regular MC rate
O(n^{-1/2}). Errors for SPX-scale parameters are sub-1bp at ~100 steps/year.
We implement kappa = 1, the universally cited practitioner choice.

This module simulates only the Volterra process Y_t and driving Brownian Z_t.
Rough Bergomi spot/variance simulation combining Y with a correlated W lives
in `pricing.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HybridScheme:
    """Bennedsen-Lunde-Pakkanen hybrid simulator for Y_t = int_0^t (t-s)^a dZ_s.

    Parameters
    ----------
    H : Hurst parameter in (0, 0.5).
    T : maximum maturity.
    n_steps : number of time steps from 0 to T.
    kappa : number of singular near-blocks. kappa = 1 is the standard choice.
    """

    H: float
    T: float
    n_steps: int
    kappa: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.H < 0.5:
            raise ValueError(f"H must be in (0, 0.5), got {self.H}")
        if self.kappa != 1:
            raise NotImplementedError("Only kappa = 1 is implemented (BLP standard).")
        self.alpha = self.H - 0.5
        self.dt = self.T / self.n_steps

        # Optimal trapezoid-style 'b_k*' collocation point (BLP Section 3.2):
        #   b_k* = [(k^{alpha+1} - (k-1)^{alpha+1}) / (alpha + 1)]^{1/alpha}
        # The far-block weight at lag k is gamma_k = (b_k* * dt)^alpha.
        k_arr = np.arange(2, self.n_steps + 1, dtype=float)
        b_star = (
            (k_arr ** (self.alpha + 1) - (k_arr - 1) ** (self.alpha + 1)) / (self.alpha + 1)
        ) ** (1.0 / self.alpha)
        gamma_k = (b_star * self.dt) ** self.alpha
        # Convolution kernel G of length n_steps + 1: G[0]=G[1]=0 (singular block
        # is added separately), G[k] = gamma_k for k = 2, ..., n_steps.
        self._G = np.concatenate([[0.0, 0.0], gamma_k])

    def simulate(
        self,
        n_paths: int,
        rng: np.random.Generator | None = None,
        antithetic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw `n_paths` joint samples of (Y_t, Z_t) on the time grid.

        Returns
        -------
        Y : ndarray of shape (n_paths, n_steps + 1). Volterra process; Y[:, 0] = 0.
        Z : ndarray of shape (n_paths, n_steps + 1). Driving Brownian; Z[:, 0] = 0.
        """
        rng = rng or np.random.default_rng()
        dt, alpha, N = self.dt, self.alpha, self.n_steps

        # Joint distribution per step: (dZ_i, W2_i) where dZ_i = Z(t_i) - Z(t_{i-1})
        # and W2_i = int_{t_{i-1}}^{t_i} (t_i - s)^alpha dZ_s. Covariance from
        # BLP eq. 3.7:
        cov = np.array([
            [dt,                                  dt ** (alpha + 1) / (alpha + 1)],
            [dt ** (alpha + 1) / (alpha + 1),     dt ** (2 * alpha + 1) / (2 * alpha + 1)],
        ])
        L = np.linalg.cholesky(cov)

        if antithetic:
            half = n_paths // 2
            normals = rng.standard_normal((half, N, 2))
            normals = np.concatenate([normals, -normals], axis=0)
            n_paths = normals.shape[0]
        else:
            normals = rng.standard_normal((n_paths, N, 2))

        joint = normals @ L.T              # (n_paths, N, 2)
        dZ = joint[..., 0]                 # driving BM increments
        W2 = joint[..., 1]                 # singular near-block contributions

        Z = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dZ, axis=1)], axis=1)

        # Vectorized FFT convolution: far-block contribution at step i (1-indexed)
        # is (G * dZ)[i] = sum_{k=2}^{i} G[k] * dZ_{i-k+1}.
        L_fft = 1 << int(np.ceil(np.log2(2 * N + 2)))
        G_f = np.fft.rfft(self._G, L_fft)
        dZ_padded = np.zeros((n_paths, L_fft))
        dZ_padded[:, :N] = dZ
        dZ_f = np.fft.rfft(dZ_padded, axis=1)
        conv = np.fft.irfft(dZ_f * G_f[None, :], L_fft, axis=1)
        far_block = conv[:, 1 : N + 1]      # (n_paths, N)

        Y_step = W2 + far_block             # Y at t_1, ..., t_N
        Y = np.concatenate([np.zeros((n_paths, 1)), Y_step], axis=1)
        return Y, Z
