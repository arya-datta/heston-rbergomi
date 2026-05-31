"""Andersen (2008) Quadratic-Exponential scheme for Heston Monte Carlo.

Reference: Andersen, L. (2008), *Simple and Efficient Simulation of the Heston
Model*, Journal of Computational Finance, 11(3).

Why QE and not Euler
--------------------
A naive Euler discretization of the variance SDE,

    v_{t+dt} = v_t + kappa (theta - v_t) dt + xi sqrt(v_t) sqrt(dt) Z,

routinely drives v below zero, especially under sub-Feller calibration
(2 kappa theta < xi^2). Reflecting / truncating Euler is biased and converges
poorly. The QE scheme moment-matches the conditional distribution of v_{t+dt}
given v_t to either:

  (a) a non-central chi-squared squared-Gaussian (when psi <= psi_c), or
  (b) an exponential-with-Dirac-mass-at-zero (when psi > psi_c),

where psi = s^2 / m^2 is the variance-to-mean ratio of v_{t+dt} | v_t. The
critical threshold psi_c = 1.5 is the value at which the two approximations
agree on their first two moments. QE is exact in moments and produces strictly
non-negative paths.

For the log-spot we use the "Broadie-Kaya-style" exact integral of the variance
in the drift, then condition on the variance process to integrate the spot SDE
(the 'martingale correction' / 'gamma' weights described in Andersen Section 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from volengine.models.heston.parameters import HestonParameters

# Optional Numba acceleration. The JIT kernel is an exact re-implementation of
# the NumPy loop below; both consume the same pre-generated random draws, so
# they agree to floating-point precision. Falls back to NumPy if numba absent.
try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore - no-op decorator fallback
        def wrap(f):
            return f
        return wrap if (args and callable(args[0])) is False else args[0]

# Andersen's defaults — gamma1 = gamma2 = 0.5 is the central scheme; the
# critical psi threshold of 1.5 is the universally cited choice.
_PSI_C = 1.5
_GAMMA1 = 0.5
_GAMMA2 = 0.5


@njit(cache=True, fastmath=False)
def _qe_kernel(  # pragma: no cover - exercised via simulate_paths(backend="numba")
    S, V, Z_v, Z_s, U, S0, v0, kappa, theta, xi, rho, r, q, dt,
    gamma1, gamma2, psi_c, martingale,
):
    """Per-path Andersen QE loop. Mirrors the vectorized NumPy version exactly.

    Numba JIT-compiles the scalar double loop (paths x steps); for many time
    steps this beats the NumPy version, which pays Python-loop overhead once per
    step. Writes spot and variance into the pre-allocated S, V arrays.
    """
    n_paths = Z_v.shape[0]
    n_steps = Z_v.shape[1]
    K0 = -rho * kappa * theta / xi * dt
    K1 = gamma1 * dt * (kappa * rho / xi - 0.5) - rho / xi
    K2 = gamma2 * dt * (kappa * rho / xi - 0.5) + rho / xi
    K3 = gamma1 * dt * (1.0 - rho * rho)
    K4 = gamma2 * dt * (1.0 - rho * rho)
    A = K2 + 0.5 * K4
    Bc = K1 + 0.5 * K3
    exp_kdt = np.exp(-kappa * dt)
    log_S0 = np.log(S0)

    for pth in range(n_paths):
        S[pth, 0] = S0
        V[pth, 0] = v0
        logS = log_S0
        v = v0
        for i in range(n_steps):
            m = theta + (v - theta) * exp_kdt
            s2 = (v * xi * xi * exp_kdt / kappa * (1.0 - exp_kdt)
                  + theta * xi * xi / (2.0 * kappa) * (1.0 - exp_kdt) ** 2)
            psi = s2 / max(m * m, 1e-16)
            log_M = 0.0
            ok = True
            if psi <= psi_c:
                inv = 1.0 / psi
                b2 = 2.0 * inv - 1.0 + np.sqrt(2.0 * inv) * np.sqrt(2.0 * inv - 1.0)
                a = m / (1.0 + b2)
                v_next = a * (np.sqrt(b2) + Z_v[pth, i]) ** 2
                d = 1.0 - 2.0 * A * a
                if d > 1e-10:
                    log_M = b2 * (A * a) / d - 0.5 * np.log(d)
                else:
                    ok = False
            else:
                p = (psi - 1.0) / (psi + 1.0)
                beta = (1.0 - p) / max(m, 1e-16)
                u = U[pth, i]
                if u <= p:
                    v_next = 0.0
                else:
                    v_next = np.log((1.0 - p) / max(1.0 - u, 1e-16)) / beta
                d = beta - A
                if d > 1e-10:
                    M = p + (1.0 - p) * beta / d
                    if M > 0.0:
                        log_M = np.log(M)
                    else:
                        ok = False
                else:
                    ok = False
            if martingale and ok:
                K0_eff = -Bc * v - log_M
            else:
                K0_eff = K0
            diff = K3 * v + K4 * v_next
            logS = (logS + (r - q) * dt + K0_eff + K1 * v + K2 * v_next
                    + np.sqrt(max(diff, 0.0)) * Z_s[pth, i])
            v = v_next
            S[pth, i + 1] = np.exp(logS)
            V[pth, i + 1] = v_next


@dataclass
class HestonQESimulator:
    """Andersen QE simulator for the Heston model.

    Parameters
    ----------
    params : Heston parameters.
    r, q   : risk-free rate and dividend yield.

    Methods
    -------
    simulate_paths(S0, T, n_paths, n_steps, seed, antithetic)
        Returns (paths, variances) of shape (n_paths, n_steps + 1).

    terminal_spots(S0, T, n_paths, n_steps, seed, antithetic)
        Returns the terminal spot array only (lighter on memory).
    """

    params: HestonParameters
    r: float
    q: float

    def simulate_paths(
        self,
        S0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        antithetic: bool = True,
        martingale_correction: bool = True,
        backend: str = "numpy",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate full spot + variance paths under the QE scheme.

        Parameters
        ----------
        backend : "numpy" (default) or "numba". Both consume the same
            pre-generated random draws and so produce identical paths to
            floating-point precision; "numba" JIT-compiles the per-path scalar
            loop and is faster when n_steps is large (it avoids the Python
            per-step overhead of the vectorized NumPy version). Requires numba;
            falls back to NumPy with a note if it is not installed.
        martingale_correction : if True (default), use Andersen's (2008, sec.
            3.5) martingale-corrected log-spot drift. Instead of the constant
            K0 = -rho*kappa*theta/xi*dt, K0 becomes path-dependent,
                K0*(V_t) = -(K1 + K3/2) V_t - log M(K2 + K4/2),
            where M is the conditional moment generating function of V_{t+dt}
            given V_t under the QE distribution (squared-Gaussian on the low-psi
            branch, exponential-with-mass on the high-psi branch). This makes
            E[S_{t+dt} | F_t] = S_t e^{(r-q)dt} hold *exactly*, so the simulated
            discounted spot is a true martingale — removing the O(dt) drift bias
            of the plain scheme. Falls back to the constant K0 on any path where
            the MGF moment condition (1 - 2 A a > 0, or beta - A > 0) is
            violated (rare for SPX-scale parameters).

        Returns
        -------
        S : ndarray of shape (n_paths, n_steps + 1), spot paths.
        V : ndarray of shape (n_paths, n_steps + 1), variance paths.
        """
        rng = np.random.default_rng(seed)
        kappa, theta, xi, rho, v0 = self.params.as_tuple()
        dt = T / n_steps

        # Antithetic doubles paths along axis 0; total path count is preserved.
        if antithetic:
            half = n_paths // 2
            Z_v = rng.standard_normal((half, n_steps))
            Z_s = rng.standard_normal((half, n_steps))
            U = rng.uniform(size=(half, n_steps))
            Z_v = np.concatenate([Z_v, -Z_v], axis=0)
            Z_s = np.concatenate([Z_s, -Z_s], axis=0)
            U = np.concatenate([U, U], axis=0)
            n_paths = Z_v.shape[0]
        else:
            Z_v = rng.standard_normal((n_paths, n_steps))
            Z_s = rng.standard_normal((n_paths, n_steps))
            U = rng.uniform(size=(n_paths, n_steps))

        S = np.empty((n_paths, n_steps + 1))
        V = np.empty((n_paths, n_steps + 1))
        S[:, 0] = S0
        V[:, 0] = v0

        # --- Numba backend: JIT the per-path loop, same draws => same result. ---
        if backend == "numba":
            if not _HAS_NUMBA:  # pragma: no cover
                import warnings
                warnings.warn("numba not installed; falling back to numpy backend.",
                              stacklevel=2)
            else:
                _qe_kernel(S, V, Z_v, Z_s, U, float(S0), float(v0),
                           kappa, theta, xi, rho, self.r, self.q, dt,
                           _GAMMA1, _GAMMA2, _PSI_C, bool(martingale_correction))
                return S, V
        elif backend != "numpy":
            raise ValueError(f"backend must be 'numpy' or 'numba', got {backend!r}")

        # Cache the constants used by the log-spot update (Andersen eq. 33).
        K0 = -rho * kappa * theta / xi * dt
        K1 = _GAMMA1 * dt * (kappa * rho / xi - 0.5) - rho / xi
        K2 = _GAMMA2 * dt * (kappa * rho / xi - 0.5) + rho / xi
        K3 = _GAMMA1 * dt * (1.0 - rho**2)
        K4 = _GAMMA2 * dt * (1.0 - rho**2)
        exp_kdt = np.exp(-kappa * dt)
        # MGF argument for the martingale correction: the V_{t+dt} coefficient
        # after integrating out the spot Brownian increment Z (which contributes
        # +0.5*(K3 V_t + K4 V_{t+dt})).
        A_mgf = K2 + 0.5 * K4
        B_mgf = K1 + 0.5 * K3   # coefficient on V_t

        logS = np.log(S[:, 0])
        for i in range(n_steps):
            v_curr = V[:, i]
            # Conditional moments of v_{t+dt} | v_t (Andersen eq. 17-18).
            m = theta + (v_curr - theta) * exp_kdt
            s2 = (
                v_curr * xi**2 * exp_kdt / kappa * (1.0 - exp_kdt)
                + theta * xi**2 / (2.0 * kappa) * (1.0 - exp_kdt) ** 2
            )
            psi = s2 / np.maximum(m**2, 1e-16)

            v_next = np.empty_like(v_curr)
            # log M(A_mgf) for the martingale correction, per path.
            log_M = np.zeros_like(v_curr)
            mgf_ok = np.ones_like(v_curr, dtype=bool)

            # --- Low-psi branch: squared-Gaussian approximation. ---
            low = psi <= _PSI_C
            if np.any(low):
                psi_lo = psi[low]
                inv_psi = 1.0 / psi_lo
                # Andersen eq. 27-29.
                b2 = 2.0 * inv_psi - 1.0 + np.sqrt(2.0 * inv_psi) * np.sqrt(2.0 * inv_psi - 1.0)
                a = m[low] / (1.0 + b2)
                v_next[low] = a * (np.sqrt(b2) + Z_v[low, i]) ** 2
                # Noncentral chi-square (1 dof, noncentrality b2) MGF of a*X at A:
                #   M(A) = (1 - 2 A a)^{-1/2} exp(b2 * A a / (1 - 2 A a))
                #   log M = -0.5 log(1 - 2 A a) + b2 * A a / (1 - 2 A a)
                d_low = 1.0 - 2.0 * A_mgf * a
                good_low = d_low > 1e-10
                d_safe = np.where(good_low, d_low, 1.0)
                log_M_low = b2 * (A_mgf * a) / d_safe - 0.5 * np.log(d_safe)
                idx_low = np.nonzero(low)[0]
                log_M[idx_low] = np.where(good_low, log_M_low, 0.0)
                mgf_ok[idx_low] = good_low

            # --- High-psi branch: exponential with point mass at 0. ---
            high = ~low
            if np.any(high):
                psi_hi = psi[high]
                p = (psi_hi - 1.0) / (psi_hi + 1.0)
                beta = (1.0 - p) / np.maximum(m[high], 1e-16)
                u_hi = U[high, i]
                v_next_hi = np.where(u_hi <= p, 0.0, np.log((1.0 - p) / np.maximum(1.0 - u_hi, 1e-16)) / beta)
                v_next[high] = v_next_hi
                # Exponential-with-mass MGF at A:  M(A) = p + (1-p) beta / (beta - A)
                d_high = beta - A_mgf
                good_high = d_high > 1e-10
                d_safe = np.where(good_high, d_high, 1.0)
                M_high = p + (1.0 - p) * beta / d_safe
                good_high = good_high & (M_high > 0.0)
                idx_high = np.nonzero(high)[0]
                log_M[idx_high] = np.where(good_high, np.log(np.where(M_high > 0.0, M_high, 1.0)), 0.0)
                mgf_ok[idx_high] = good_high

            # Drift constant: martingale-corrected (path-dependent) or plain.
            if martingale_correction:
                K0_eff = np.where(mgf_ok, -B_mgf * v_curr - log_M, K0)
            else:
                K0_eff = K0

            # Log-spot update (Andersen eq. 33), conditional on (v_curr, v_next).
            logS = (
                logS
                + (self.r - self.q) * dt
                + K0_eff
                + K1 * v_curr
                + K2 * v_next
                + np.sqrt(K3 * v_curr + K4 * v_next) * Z_s[:, i]
            )
            S[:, i + 1] = np.exp(logS)
            V[:, i + 1] = v_next

        return S, V

    def terminal_spots(
        self,
        S0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        antithetic: bool = True,
    ) -> np.ndarray:
        """Convenience: return only S_T. Useful for European pricing."""
        S, _ = self.simulate_paths(S0, T, n_paths, n_steps, seed, antithetic)
        return S[:, -1]

    def price_european(
        self,
        S0: float,
        T: float,
        K: float | np.ndarray,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        antithetic: bool = True,
        flag: str = "call",
        control_variate: bool = True,
        return_stderr: bool = False,
    ) -> float | np.ndarray | tuple:
        """Price European vanillas by QE Monte Carlo.

        Mirrors `volengine.models.rbergomi.pricing.rbergomi_price`: optional
        terminal-spot control variate (known mean E[S_T] = S0 e^{(r-q)T}) and
        optional Monte Carlo standard error. The control variate reduces
        variance by a factor that grows with payoff/S_T correlation (~1.5x ATM,
        larger ITM), at no bias cost.

        Returns
        -------
        price : float or ndarray (matching K).
        If return_stderr is True, returns (price, stderr).
        """
        ST = self.terminal_spots(S0, T, n_paths, n_steps, seed, antithetic)
        K_arr = np.atleast_1d(np.asarray(K, dtype=float))
        disc = np.exp(-self.r * T)
        n = ST.shape[0]

        if flag == "call":
            payoffs = np.maximum(ST[:, None] - K_arr[None, :], 0.0)
        elif flag == "put":
            payoffs = np.maximum(K_arr[None, :] - ST[:, None], 0.0)
        else:
            raise ValueError(f"flag must be 'call' or 'put', got {flag!r}")

        if control_variate:
            EST = S0 * np.exp((self.r - self.q) * T)
            ST_centered = ST - ST.mean()
            var_ST = float(np.mean(ST_centered**2))
            if var_ST > 0:
                cov = (payoffs - payoffs.mean(axis=0)[None, :]) * ST_centered[:, None]
                beta = cov.mean(axis=0) / var_ST
                payoffs = payoffs - beta[None, :] * (ST[:, None] - EST)

        price = disc * payoffs.mean(axis=0)
        is_scalar = np.isscalar(K) or (np.asarray(K).ndim == 0)
        out_price = float(price[0]) if is_scalar else price
        if not return_stderr:
            return out_price
        stderr = disc * payoffs.std(axis=0, ddof=1) / np.sqrt(n)
        out_stderr = float(stderr[0]) if is_scalar else stderr
        return out_price, out_stderr
