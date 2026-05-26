"""End-to-end calibration sanity tests: recover known params from synthetic data.

These tests price a synthetic surface using known parameters, then verify the
calibrator recovers parameters tight enough to reprice that same surface within
a few vol points. They're marked `slow` because each run executes a full DE +
L-BFGS-B optimization (and for rBergomi, an MC inside every DE evaluation).
"""

import numpy as np
import pytest

from volengine.calibration import IVQuote, calibrate_heston, calibrate_rbergomi
from volengine.models.heston import HestonParameters, heston_vanilla_price
from volengine.models.rbergomi import RBergomiParameters, rbergomi_price
from volengine.surfaces.implied_vol import implied_vol


@pytest.mark.slow
def test_heston_recovers_synthetic_surface(market):
    """Generate synthetic Heston surface, calibrate, check IV RMSE < 50 bps."""
    true = HestonParameters(kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
    quotes = []
    for T in [0.1, 0.25, 0.5, 1.0]:
        F = market["S0"] * np.exp((market["r"] - market["q"]) * T)
        for k in np.linspace(-0.2, 0.2, 9):
            K = F * np.exp(k)
            price = heston_vanilla_price(K, T, market["S0"], market["r"], market["q"], true)
            iv = implied_vol(float(price), market["S0"], K, T, market["r"], market["q"], "call")
            if np.isfinite(iv):
                quotes.append(IVQuote(K=K, T=T, iv_mkt=iv, weight=1.0))

    res = calibrate_heston(quotes, S0=market["S0"], r=market["r"], q=market["q"],
                           de_maxiter=30, de_popsize=12, seed=0)
    # Recovery should be < 50 bps in vol points (it's synthetic — should be ~5bps).
    assert res.rmse_vol_points < 0.005


@pytest.mark.slow
def test_rbergomi_recovers_synthetic_surface(market):
    """Generate synthetic rBergomi surface, calibrate, check IV RMSE < 150 bps.

    Tolerance is wider than the Heston test because every objective evaluation
    here involves MC noise. The synthetic ground-truth surface itself carries
    MC error, so a perfect calibrator could not beat that floor — 150 bps is
    the empirical noise floor with n_paths=8000.
    """
    true = RBergomiParameters(H=0.12, eta=1.8, rho=-0.8, xi0=0.04)
    S0, r, q = market["S0"], market["r"], market["q"]

    # Build synthetic quotes — share seeds with calibration so MC noise cancels.
    quotes = []
    for T in [0.1, 0.25, 0.5]:
        F = S0 * np.exp((r - q) * T)
        Ks = np.array([F * np.exp(k) for k in np.linspace(-0.15, 0.15, 5)])
        prices = rbergomi_price(Ks, T, S0, r, q, true,
                                n_paths=8000, n_steps=max(20, int(100 * T)),
                                seed=42)
        for K, p in zip(Ks, np.atleast_1d(prices)):
            iv = implied_vol(float(p), S0, float(K), T, r, q, "call")
            if np.isfinite(iv):
                quotes.append(IVQuote(K=float(K), T=T, iv_mkt=iv, weight=1.0))

    assert len(quotes) >= 10, f"only {len(quotes)} synthetic quotes survived"
    res = calibrate_rbergomi(
        quotes, S0=S0, r=r, q=q,
        n_paths=8000, n_steps_per_year=100,
        de_maxiter=15, de_popsize=10, seed=42,
    )
    # 150 bps tolerance for MC-noisy calibration.
    assert res.rmse_vol_points < 0.015, (
        f"rBergomi RMSE {res.rmse_vol_points * 100:.2f} vol pts exceeds 1.5 vol pts"
    )
