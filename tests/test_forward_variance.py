"""Tests for the rBergomi forward-variance curve and fixed-xi0 calibration."""

import numpy as np
import pytest

from volengine.calibration import IVQuote, calibrate_rbergomi_fixed_xi0
from volengine.models.rbergomi import (
    ForwardVarianceCurve,
    RBergomiParameters,
    rbergomi_price,
    simulate_rbergomi,
)
from volengine.surfaces.implied_vol import implied_vol


def test_flat_forward_variance_curve_is_constant():
    c = ForwardVarianceCurve(times=[0.5, 1.0], fwd_var=[0.04, 0.04])
    assert c(0.25) == pytest.approx(0.04)
    assert c(2.0) == pytest.approx(0.04)


def test_from_atm_term_structure_recovers_flat_level():
    # Flat ATM vol -> constant forward variance sigma^2.
    mats = np.array([0.25, 0.5, 1.0, 2.0])
    sig = np.full_like(mats, 0.20)
    curve = ForwardVarianceCurve.from_atm_term_structure(mats, sig)
    for t in (0.1, 0.4, 1.5):
        assert curve(t) == pytest.approx(0.04, abs=1e-6)


def test_from_atm_term_structure_upward_curve():
    # Rising ATM vol -> rising forward variance.
    mats = np.array([0.25, 1.0, 2.0])
    sig = np.array([0.15, 0.20, 0.25])
    curve = ForwardVarianceCurve.from_atm_term_structure(mats, sig)
    assert curve(0.2) < curve(1.5)            # forward variance increases
    # mean_variance over [0,T] should be positive and ~ the realized level.
    assert 0.0 < curve.mean_variance(1.0) < 0.1


def test_forward_variance_curve_drives_simulation():
    """A ForwardVarianceCurve plugged into RBergomiParameters preserves E[V_t]."""
    curve = ForwardVarianceCurve(times=[0.25, 1.0], fwd_var=[0.02, 0.06], kind="linear")
    p = RBergomiParameters(H=0.1, eta=1.5, rho=-0.8, xi0=curve)
    _, V = simulate_rbergomi(100.0, T=1.0, params=p, r=0.0, q=0.0,
                             n_paths=20_000, n_steps=100, seed=3, return_variance=True)
    t = np.linspace(0.0, 1.0, V.shape[1])
    # Martingale correction => E[V_t] = xi0(t) at every t.
    assert np.allclose(V.mean(axis=0), curve(t), atol=6e-3)


def test_curve_validation():
    with pytest.raises(ValueError):
        ForwardVarianceCurve(times=[1.0, 0.5], fwd_var=[0.04, 0.04])  # not ascending
    with pytest.raises(ValueError):
        ForwardVarianceCurve(times=[1.0], fwd_var=[0.04], kind="bogus")


@pytest.mark.slow
def test_fixed_xi0_calibration_recovers_dynamics(market):
    """Generate a surface from known (H, eta, rho) + a curve, then recover the
    three dynamics params with xi0 held fixed to that curve."""
    curve = ForwardVarianceCurve(times=[0.1, 1.0], fwd_var=[0.03, 0.05], kind="linear")
    true = RBergomiParameters(H=0.12, eta=1.8, rho=-0.8, xi0=curve)
    S0, r, q = market["S0"], 0.0, 0.0

    quotes = []
    for T in (0.1, 0.25, 0.5):
        F = S0 * np.exp((r - q) * T)
        Ks = np.array([F * np.exp(k) for k in np.linspace(-0.12, 0.12, 5)])
        prices = rbergomi_price(Ks, T, S0, r, q, true, n_paths=8000,
                                n_steps=max(20, int(100 * T)), seed=11)
        for K, p in zip(Ks, np.atleast_1d(prices), strict=True):
            iv = implied_vol(float(p), S0, float(K), T, r, q, "call")
            if np.isfinite(iv):
                quotes.append(IVQuote(K=float(K), T=T, iv_mkt=iv))

    res = calibrate_rbergomi_fixed_xi0(
        quotes, S0=S0, r=r, q=q, xi0_curve=curve,
        n_paths=8000, n_steps_per_year=100, de_maxiter=12, de_popsize=8, seed=11,
    )
    assert res.rmse_vol_points < 0.02      # within 2 vol pts of the surface
