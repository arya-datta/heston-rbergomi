"""Tests for Black-Scholes pricing and IV inversion."""

import numpy as np
import pytest

from volengine.surfaces.implied_vol import (
    black_scholes_price,
    black_scholes_vega,
    implied_vol,
)


def test_put_call_parity(market):
    K, T, sigma = 105.0, 0.5, 0.2
    c = black_scholes_price(market["S0"], K, T, market["r"], market["q"], sigma, "call")
    p = black_scholes_price(market["S0"], K, T, market["r"], market["q"], sigma, "put")
    lhs = c - p
    rhs = market["S0"] * np.exp(-market["q"] * T) - K * np.exp(-market["r"] * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


@pytest.mark.parametrize("K,sigma", [(80, 0.15), (100, 0.20), (130, 0.30)])
def test_iv_roundtrip(K, sigma, market):
    T = 0.75
    price = black_scholes_price(market["S0"], K, T, market["r"], market["q"], sigma, "call")
    iv = implied_vol(price, market["S0"], K, T, market["r"], market["q"], "call")
    assert iv == pytest.approx(sigma, abs=1e-5)


def test_iv_returns_nan_below_intrinsic(market):
    # Quote below intrinsic should return NaN, not raise.
    bad_price = -1.0
    iv = implied_vol(bad_price, market["S0"], 80.0, 0.5, market["r"], market["q"], "call")
    assert np.isnan(iv)


def test_vega_matches_finite_diff(market):
    K, T, sigma = 100.0, 0.5, 0.25
    h = 1e-5
    vega = black_scholes_vega(market["S0"], K, T, market["r"], market["q"], sigma)
    fd = (
        black_scholes_price(market["S0"], K, T, market["r"], market["q"], sigma + h, "call")
        - black_scholes_price(market["S0"], K, T, market["r"], market["q"], sigma - h, "call")
    ) / (2 * h)
    assert vega == pytest.approx(fd, rel=1e-5)
