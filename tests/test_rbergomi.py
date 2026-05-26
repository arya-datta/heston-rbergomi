"""Tests for the rough Bergomi hybrid scheme and pricer."""

import numpy as np
import pytest

from volengine.models.rbergomi import HybridScheme, RBergomiParameters, simulate_rbergomi


def test_hybrid_scheme_y_has_zero_mean(rng):
    H, T, n_steps = 0.1, 0.5, 100
    scheme = HybridScheme(H=H, T=T, n_steps=n_steps)
    Y, Z = scheme.simulate(n_paths=4000, rng=rng, antithetic=True)
    # Antithetic ensures sample mean is exactly zero.
    assert np.allclose(Y.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-12)


@pytest.mark.slow
def test_hybrid_scheme_y_variance_matches_theory(rng):
    """Var(Y_t) = int_0^t s^{2 alpha} ds = t^{2H} / (2H), where alpha = H - 1/2."""
    H, T, n_steps = 0.1, 1.0, 200
    scheme = HybridScheme(H=H, T=T, n_steps=n_steps)
    Y, _ = scheme.simulate(n_paths=20_000, rng=rng)
    t = T  # check at terminal time
    theoretical = t ** (2 * H) / (2 * H)
    empirical = Y[:, -1].var()
    # Hybrid scheme + MC noise: 5% relative tolerance is the BLP norm.
    assert abs(empirical - theoretical) / theoretical < 0.10


@pytest.mark.slow
def test_rbergomi_spot_is_martingale_under_zero_rates(rbergomi_params_typical, market):
    """Under r = q = 0, S_t is a Q-martingale: E[S_T] = S_0."""
    ST = simulate_rbergomi(
        S0=market["S0"], T=0.5, params=rbergomi_params_typical, r=0.0, q=0.0,
        n_paths=20_000, n_steps=100, seed=11,
    )[:, -1]
    # 3 sigma bound on the mean estimator.
    assert abs(ST.mean() - market["S0"]) < 1.0


@pytest.mark.slow
def test_rbergomi_variance_curve_preserved(market):
    """E[V_t] should equal xi0 at every t (martingale correction property)."""
    from volengine.models.rbergomi.pricing import simulate_rbergomi
    p = RBergomiParameters(H=0.1, eta=2.0, rho=-0.9, xi0=0.04)
    _, V = simulate_rbergomi(market["S0"], T=0.5, params=p, r=0.0, q=0.0,
                              n_paths=20_000, n_steps=100, seed=3,
                              return_variance=True)
    mean_V = V.mean(axis=0)
    assert np.allclose(mean_V, 0.04, atol=5e-3)
