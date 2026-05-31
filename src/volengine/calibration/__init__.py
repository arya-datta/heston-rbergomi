"""Two-stage calibrators for Heston and rough Bergomi.

Each calibrator combines:

  (1) a global pass (SciPy `differential_evolution`) to find the basin, then
  (2) a local refinement (L-BFGS-B) to polish.

The objective in both cases is weighted RMSE of model implied volatilities
versus market — IV-space rather than price-space because IV errors are
roughly homogeneous across moneyness, where price errors are heavily skewed.
"""

from volengine.calibration.heston_calibrator import HestonCalibrationResult, calibrate_heston
from volengine.calibration.objective import (
    FAILED_QUOTE_PENALTY,
    IVQuote,
    build_iv_quotes,
    iv_rmse_objective,
)
from volengine.calibration.rbergomi_calibrator import RBergomiCalibrationResult, calibrate_rbergomi

__all__ = [
    "IVQuote",
    "build_iv_quotes",
    "iv_rmse_objective",
    "FAILED_QUOTE_PENALTY",
    "calibrate_heston",
    "HestonCalibrationResult",
    "calibrate_rbergomi",
    "RBergomiCalibrationResult",
]
