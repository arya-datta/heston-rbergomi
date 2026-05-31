"""Tests for the Andersen QE Heston Monte Carlo scheme.

The key benchmark: QE-MC vanilla prices should match Carr-Madan FFT prices
within MC error (~30 bps at 50k paths).
"""

import numpy as np
import pytest

from volengine.models.heston import HestonQESimulator, heston_vanilla_price
from volengine.models.heston.qe_simulation import _HAS_NUMBA


def test_qe_variance_strictly_nonnegative(heston_params_typical, market):
    sim = HestonQESimulator(params=heston_params_typical, r=market["r"], q=market["q"])
    _, V = sim.simulate_paths(market["S0"], T=0.5, n_paths=2000, n_steps=50, seed=0)
    assert np.all(V >= 0.0)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
def test_qe_numba_backend_matches_numpy(heston_params_typical, market):
    """The JIT kernel and the NumPy loop share random draws, so they must agree
    to floating-point precision — on both the martingale-corrected and plain
    schemes."""
    sim = HestonQESimulator(params=heston_params_typical, r=market["r"], q=market["q"])
    for mc in (True, False):
        S_np, V_np = sim.simulate_paths(market["S0"], T=0.5, n_paths=3000, n_steps=60,
                                        seed=7, martingale_correction=mc, backend="numpy")
        S_nb, V_nb = sim.simulate_paths(market["S0"], T=0.5, n_paths=3000, n_steps=60,
                                        seed=7, martingale_correction=mc, backend="numba")
        assert np.allclose(S_np, S_nb, rtol=1e-8, atol=1e-8)
        assert np.allclose(V_np, V_nb, rtol=1e-8, atol=1e-8)


def test_qe_backend_rejects_unknown(heston_params_typical, market):
    sim = HestonQESimulator(params=heston_params_typical, r=market["r"], q=market["q"])
    with pytest.raises(ValueError, match="backend"):
        sim.simulate_paths(market["S0"], T=0.5, n_paths=100, n_steps=10, backend="cuda")


@pytest.mark.slow
def test_qe_vanilla_price_matches_fft(heston_params_typical, market):
    T, K = 0.5, 100.0
    fft = float(heston_vanilla_price(K, T, market["S0"], market["r"],
                                      market["q"], heston_params_typical))
    sim = HestonQESimulator(params=heston_params_typical, r=market["r"], q=market["q"])
    ST = sim.terminal_spots(market["S0"], T, n_paths=40_000, n_steps=80, seed=42)
    mc = float(np.exp(-market["r"] * T) * np.maximum(ST - K, 0).mean())
    # Tolerance is dominated by MC standard error; 30 bps of spot is the target.
    assert abs(mc - fft) < 0.30 * market["S0"] / 100.0  # ~0.30


@pytest.mark.slow
def test_qe_terminal_mean_is_martingale(market):
    """With the martingale correction, E[S_T] = S0 e^{(r-q)T} within MC error.

    Asserts the mean lands within 3.5 sample standard errors of the exact
    forward — a real test of unbiasedness, not the old loose 0.5 window. Uses a
    deliberately sub-Feller, high-vol-of-vol setting where the *uncorrected*
    scheme's drift bias is largest.
    """
    from volengine.models.heston import HestonParameters
    p = HestonParameters(kappa=1.0, theta=0.06, xi=0.9, rho=-0.7, v0=0.06)  # 2kt=0.12 < xi^2=0.81
    r, q, T, N = 0.03, 0.01, 1.0, 100_000
    sim = HestonQESimulator(params=p, r=r, q=q)
    ST = sim.terminal_spots(market["S0"], T, n_paths=N, n_steps=100, seed=7)
    fwd = market["S0"] * np.exp((r - q) * T)
    stderr = ST.std(ddof=1) / np.sqrt(N)
    assert abs(ST.mean() - fwd) < 3.5 * stderr


@pytest.mark.slow
def test_qe_martingale_correction_reduces_bias(market):
    """The correction must bring E[S_T] closer to the exact forward than the
    uncorrected scheme, on shared random numbers (so the comparison is clean)."""
    from volengine.models.heston import HestonParameters
    p = HestonParameters(kappa=1.0, theta=0.06, xi=0.9, rho=-0.7, v0=0.06)
    r, q, T, N = 0.03, 0.01, 1.0, 100_000
    sim = HestonQESimulator(params=p, r=r, q=q)
    fwd = market["S0"] * np.exp((r - q) * T)
    st_corr, _ = sim.simulate_paths(market["S0"], T, N, 100, seed=7,
                                    martingale_correction=True)
    st_plain, _ = sim.simulate_paths(market["S0"], T, N, 100, seed=7,
                                     martingale_correction=False)
    bias_corr = abs(st_corr[:, -1].mean() - fwd)
    bias_plain = abs(st_plain[:, -1].mean() - fwd)
    assert bias_corr < bias_plain
