"""Tests for the SVI surface module."""

import numpy as np
import pytest

from volengine.surfaces.svi import (
    SVIParameters,
    fit_svi_slice,
    svi_butterfly_function,
    svi_implied_vol,
    svi_total_variance,
)


def test_svi_total_variance_nonnegative():
    p = SVIParameters(a=0.01, b=0.1, rho=-0.5, m=0.0, sigma=0.2)
    k = np.linspace(-1, 1, 50)
    assert np.all(svi_total_variance(k, p) >= 0)


def test_svi_butterfly_arb_free_for_smooth_smile():
    # Well-behaved SVI slice should pass butterfly check on a wide grid.
    p = SVIParameters(a=0.02, b=0.05, rho=-0.4, m=0.0, sigma=0.3)
    k = np.linspace(-0.5, 0.5, 100)
    g = svi_butterfly_function(k, p)
    assert np.all(g > -1e-6)


def test_svi_fit_recovers_clean_synthetic_slice():
    T = 0.5
    true = SVIParameters(a=0.015, b=0.08, rho=-0.45, m=0.0, sigma=0.25)
    k = np.linspace(-0.4, 0.4, 30)
    iv = svi_implied_vol(k, T, true)
    fit = fit_svi_slice(k, iv, T)
    # Refit IV should agree everywhere even if individual params don't (SVI
    # has minor non-identifiability under noise — match in IV space, not in
    # parameter space).
    iv_refit = svi_implied_vol(k, T, fit)
    assert np.max(np.abs(iv_refit - iv)) < 5e-3
