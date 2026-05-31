"""Rough Bergomi (Bayer-Friz-Gatheral 2016) stochastic volatility model.

Exports
-------
- RBergomiParameters   : (H, eta, rho, xi0) container
- ForwardVarianceCurve : callable xi0(t) built from the ATM term structure
- HybridScheme         : Bennedsen-Lunde-Pakkanen simulation of fBm increments
- rbergomi_price       : MC pricer for European vanillas
"""

from volengine.models.rbergomi.forward_variance import ForwardVarianceCurve
from volengine.models.rbergomi.hybrid_scheme import HybridScheme
from volengine.models.rbergomi.parameters import RBergomiParameters
from volengine.models.rbergomi.pricing import rbergomi_price, simulate_rbergomi

__all__ = [
    "RBergomiParameters",
    "ForwardVarianceCurve",
    "HybridScheme",
    "rbergomi_price",
    "simulate_rbergomi",
]
