"""Shared fixtures for the test suite."""

import numpy as np
import pytest

from volengine.models.heston import HestonParameters
from volengine.models.rbergomi import RBergomiParameters


@pytest.fixture(scope="session")
def market():
    """Standard SPX-like market parameters used across tests."""
    return dict(S0=100.0, r=0.03, q=0.01)


@pytest.fixture(scope="session")
def heston_params_typical():
    """SPX-typical Heston parameters (Feller-violating, as is the norm)."""
    return HestonParameters(kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)


@pytest.fixture(scope="session")
def rbergomi_params_typical():
    """Rough Bergomi parameters of the empirical SPX regime."""
    return RBergomiParameters(H=0.1, eta=1.9, rho=-0.9, xi0=0.04)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)
