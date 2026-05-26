"""Tests for the Andersen QE Heston Monte Carlo scheme.

The key benchmark: QE-MC vanilla prices should match Carr-Madan FFT prices
within MC error (~30 bps at 50k paths).
"""

import numpy as np
import pytest

from volengine.models.heston import HestonQESimulator, heston_vanilla_price


def test_qe_variance_strictly_nonnegative(heston_params_typical, market):
    sim = HestonQESimulator(params=heston_params_typical, r=market["r"], q=market["q"])
    _, V = sim.simulate_paths(market["S0"], T=0.5, n_paths=2000, n_steps=50, seed=0)
    assert np.all(V >= 0.0)


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
def test_qe_terminal_mean_unbiased_under_zero_drift(market):
    """Under r = q = 0, E[S_T] = S_0 exactly (martingale property)."""
    from volengine.models.heston import HestonParameters
    p = HestonParameters(kappa=2.0, theta=0.04, xi=0.4, rho=-0.5, v0=0.04)
    sim = HestonQESimulator(params=p, r=0.0, q=0.0)
    ST = sim.terminal_spots(market["S0"], T=0.5, n_paths=50_000, n_steps=80, seed=7)
    # MC standard error scales like S0 * sigma * sqrt(T) / sqrt(N) ~= 0.6 -> 3 sigma ~ 1.8.
    assert abs(ST.mean() - market["S0"]) < 0.5
