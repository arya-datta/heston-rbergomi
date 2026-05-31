"""Tests for the Heston characteristic function and Carr-Madan FFT pricer."""

import numpy as np
import pytest
from scipy.integrate import quad

from volengine.models.heston import (
    HestonParameters,
    carr_madan_price,
    heston_char_fn,
    heston_vanilla_price,
)
from volengine.surfaces.implied_vol import black_scholes_price


def _heston_call_by_integration(K, T, S0, r, q, params, alpha=1.5):
    """Direct Carr-Madan integral (brute force) for cross-checking FFT pricer."""
    def integrand(v):
        phi = heston_char_fn(v - 1j * (alpha + 1), T, S0, r, q, params)
        psi = np.exp(-r * T) * phi / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
        return (np.exp(-1j * v * np.log(K)) * psi).real
    val, _ = quad(integrand, 0, 200.0, limit=200)
    return np.exp(-alpha * np.log(K)) / np.pi * val


def test_charfn_at_zero_is_unity(heston_params_typical, market):
    phi0 = heston_char_fn(0.0, 0.5, market["S0"], market["r"], market["q"], heston_params_typical)
    assert np.abs(phi0 - 1.0) < 1e-10


def test_carr_madan_matches_direct_integration(heston_params_typical, market):
    T = 0.5
    K = 100.0
    fft = float(carr_madan_price(np.array([K]), T, market["S0"], market["r"],
                                  market["q"], heston_params_typical)[0])
    direct = _heston_call_by_integration(K, T, market["S0"], market["r"],
                                          market["q"], heston_params_typical)
    # 4 decimal places is the BLP / Albrecher canonical benchmark.
    assert fft == pytest.approx(direct, abs=1e-3)


def test_carr_madan_recovers_bs_in_zero_vol_of_vol_limit(market):
    """xi -> 0 and constant v0 = theta should reduce Heston to Black-Scholes."""
    p = HestonParameters(kappa=2.0, theta=0.04, xi=1e-6, rho=0.0, v0=0.04)
    T, K = 0.5, 100.0
    heston_price = float(heston_vanilla_price(K, T, market["S0"], market["r"], market["q"], p))
    bs = black_scholes_price(market["S0"], K, T, market["r"], market["q"], 0.2, "call")
    assert heston_price == pytest.approx(bs, abs=2e-3)


def test_carr_madan_implied_vols_monotone_in_strike(heston_params_typical, market):
    T = 0.5
    Ks = np.linspace(80, 120, 9)
    prices = heston_vanilla_price(Ks, T, market["S0"], market["r"], market["q"], heston_params_typical)
    # Call prices should be monotone non-increasing in K.
    assert np.all(np.diff(prices) <= 1e-8)
