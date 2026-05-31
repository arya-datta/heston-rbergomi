"""Tests for Monte Carlo quality features: control variates and standard errors.

These verify the two MC-honesty improvements:
  - rbergomi_price / HestonQESimulator.price_european report a finite, positive
    standard error when asked.
  - The terminal-spot control variate reduces variance without biasing the
    price (the CV and plain estimates agree within a few standard errors).
"""

import numpy as np
import pytest

from volengine.models.heston import HestonQESimulator, heston_vanilla_price
from volengine.models.rbergomi import rbergomi_price


def test_rbergomi_price_returns_finite_stderr(rbergomi_params_typical, market):
    price, stderr = rbergomi_price(
        100.0, T=0.5, S0=market["S0"], r=market["r"], q=market["q"],
        params=rbergomi_params_typical, n_paths=8000, n_steps=80, seed=1,
        return_stderr=True,
    )
    assert np.isfinite(price) and price > 0
    assert np.isfinite(stderr) and stderr > 0
    assert stderr < price          # a sane MC at 8k paths


@pytest.mark.slow
def test_rbergomi_control_variate_reduces_variance(rbergomi_params_typical, market):
    """The terminal-spot control variate should cut the ATM stderr materially.

    For an ATM call under stochastic vol the payoff/S_T correlation is moderate
    (~0.6), so the reduction is ~1.5x — not the larger factor seen for ITM
    strikes where the payoff is nearly linear in S_T. We assert a conservative
    >1.3x reduction at ATM.
    """
    kw = dict(K=market["S0"], T=0.5, S0=market["S0"], r=market["r"],
              q=market["q"], params=rbergomi_params_typical,
              n_paths=20_000, n_steps=100, seed=7, return_stderr=True)
    _, se_plain = rbergomi_price(**kw, control_variate=False)
    _, se_cv = rbergomi_price(**kw, control_variate=True)
    assert se_cv < se_plain / 1.3


@pytest.mark.slow
def test_rbergomi_control_variate_stronger_itm(rbergomi_params_typical, market):
    """Deep-ITM payoff is nearly linear in S_T -> CV reduction is larger than ATM."""
    base = dict(T=0.5, S0=market["S0"], r=market["r"], q=market["q"],
                params=rbergomi_params_typical, n_paths=20_000, n_steps=100,
                seed=7, return_stderr=True)
    _, se_atm_plain = rbergomi_price(K=market["S0"], **base, control_variate=False)
    _, se_atm_cv = rbergomi_price(K=market["S0"], **base, control_variate=True)
    _, se_itm_plain = rbergomi_price(K=0.7 * market["S0"], **base, control_variate=False)
    _, se_itm_cv = rbergomi_price(K=0.7 * market["S0"], **base, control_variate=True)
    atm_factor = se_atm_plain / se_atm_cv
    itm_factor = se_itm_plain / se_itm_cv
    assert itm_factor > atm_factor      # CV more effective where correlation is higher


@pytest.mark.slow
def test_rbergomi_control_variate_is_unbiased(rbergomi_params_typical, market):
    """CV and plain price estimates must agree within combined MC error."""
    kw = dict(K=market["S0"], T=0.5, S0=market["S0"], r=market["r"],
              q=market["q"], params=rbergomi_params_typical,
              n_paths=20_000, n_steps=100, seed=7, return_stderr=True)
    p_plain, se_plain = rbergomi_price(**kw, control_variate=False)
    p_cv, se_cv = rbergomi_price(**kw, control_variate=True)
    # Within 4 combined standard errors — they share the same paths so this is
    # a very tight check that CV didn't introduce a bias.
    tol = 4.0 * np.hypot(se_plain, se_cv)
    assert abs(p_cv - p_plain) < max(tol, 1e-3)


@pytest.mark.slow
def test_qe_price_european_matches_fft_with_stderr(heston_params_typical, market):
    """QE price_european with CV should match FFT within its reported stderr."""
    T, K = 0.5, 100.0
    fft = float(heston_vanilla_price(K, T, market["S0"], market["r"],
                                     market["q"], heston_params_typical))
    sim = HestonQESimulator(params=heston_params_typical, r=market["r"], q=market["q"])
    mc, se = sim.price_european(market["S0"], T, K, n_paths=40_000, n_steps=80,
                                seed=42, return_stderr=True, control_variate=True)
    assert np.isfinite(se) and se > 0
    # MC should land within ~3.5 standard errors of the FFT truth.
    assert abs(mc - fft) < 3.5 * se + 1e-3
