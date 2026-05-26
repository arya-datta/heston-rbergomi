"""Vanilla products supported across both models: European options and variance swaps."""

from volengine.products.european import european_price_heston, european_price_rbergomi
from volengine.products.variance_swap import (
    variance_swap_rate_heston,
    variance_swap_rate_rbergomi,
    variance_swap_rate_replication,
)

__all__ = [
    "european_price_heston",
    "european_price_rbergomi",
    "variance_swap_rate_heston",
    "variance_swap_rate_rbergomi",
    "variance_swap_rate_replication",
]
