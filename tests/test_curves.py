"""Tests for the rate/dividend term-structure curves and their integration."""

import numpy as np
import pytest

from volengine.calibration import IVQuote
from volengine.calibration.heston_calibrator import _model_ivs
from volengine.market import FlatCurve, ZeroCurve, as_curve
from volengine.models.heston import HestonParameters, heston_vanilla_price
from volengine.surfaces.implied_vol import implied_vol


def test_flat_curve_constant_and_discount():
    c = FlatCurve(0.04)
    assert c.zero_rate(0.5) == 0.04
    assert c.zero_rate(5.0) == 0.04
    assert c.discount(2.0) == pytest.approx(np.exp(-0.04 * 2.0))
    assert c.forward_rate(1.0, 3.0) == 0.04


def test_zero_curve_interpolates_and_flat_extrapolates():
    c = ZeroCurve(times=[0.25, 1.0, 5.0], zero_rates=[0.03, 0.04, 0.045])
    # Knot values exact.
    assert c.zero_rate(1.0) == pytest.approx(0.04)
    # Linear interpolation halfway between 0.25 and 1.0 (in time).
    mid = 0.03 + (0.04 - 0.03) * (0.625 - 0.25) / (1.0 - 0.25)
    assert c.zero_rate(0.625) == pytest.approx(mid)
    # Flat extrapolation beyond the endpoints.
    assert c.zero_rate(0.01) == pytest.approx(0.03)
    assert c.zero_rate(10.0) == pytest.approx(0.045)


def test_zero_curve_forward_rate_consistency():
    c = ZeroCurve(times=[1.0, 2.0], zero_rates=[0.03, 0.05])
    # f(1,2) = [z(2)*2 - z(1)*1] / 1 = 0.10 - 0.03 = 0.07
    assert c.forward_rate(1.0, 2.0) == pytest.approx(0.07)


def test_zero_curve_validation():
    with pytest.raises(ValueError):
        ZeroCurve(times=[1.0, 0.5], zero_rates=[0.03, 0.04])  # not ascending
    with pytest.raises(ValueError):
        ZeroCurve(times=[1.0, 2.0], zero_rates=[0.03])         # length mismatch


def test_as_curve_coercion():
    assert isinstance(as_curve(0.05), FlatCurve)
    zc = ZeroCurve([1.0, 2.0], [0.03, 0.04])
    assert as_curve(zc) is zc
    assert as_curve(0.05).zero_rate(1.0) == 0.05
    with pytest.raises(TypeError):
        as_curve("not a rate")


def test_calibrator_uses_per_maturity_rates():
    """_model_ivs must price each maturity with its own zero rate.

    Build a synthetic surface where each maturity is priced with the curve's
    rate. Re-inverting with the SAME curve recovers the input IVs exactly;
    using a flat rate equal to the short end mis-prices the long maturities.
    """
    S0 = 100.0
    params = HestonParameters(kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
    r_curve = ZeroCurve(times=[0.1, 1.0], zero_rates=[0.02, 0.06])  # steep curve
    q_curve = FlatCurve(0.01)

    quotes = []
    for T in (0.1, 0.5, 1.0):
        r_T, q_T = r_curve.zero_rate(T), q_curve.zero_rate(T)
        F = S0 * np.exp((r_T - q_T) * T)
        for k in np.linspace(-0.1, 0.1, 5):
            K = F * np.exp(k)
            price = heston_vanilla_price(K, T, S0, r_T, q_T, params)
            iv = implied_vol(float(price), S0, K, T, r_T, q_T, "call")
            if np.isfinite(iv):
                quotes.append(IVQuote(K=float(K), T=T, iv_mkt=iv))

    # With the correct curve, the model reproduces the surface to ~0.
    iv_curve = _model_ivs(params, S0, r_curve, q_curve, quotes)
    err_curve = np.nanmax(np.abs(iv_curve - np.array([q.iv_mkt for q in quotes])))
    assert err_curve < 1e-3

    # With a flat rate pinned to the short end, the long-dated quotes are
    # mis-priced (wrong forward/discount) — error is materially larger.
    iv_flat = _model_ivs(params, S0, FlatCurve(0.02), q_curve, quotes)
    err_flat = np.nanmax(np.abs(iv_flat - np.array([q.iv_mkt for q in quotes])))
    assert err_flat > 5 * err_curve
