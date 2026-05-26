"""Two-stage calibrators for Heston and rough Bergomi.

Each calibrator combines:

  (1) a global pass (SciPy `differential_evolution`) to find the basin, then
  (2) a local refinement (L-BFGS-B) to polish.

The objective in both cases is weighted RMSE of model implied volatilities
versus market — IV-space rather than price-space because IV errors are
roughly homogeneous across moneyness, where price errors are heavily skewed.
"""

from volengine.calibration.objective import IVQuote, iv_rmse_objective
from volengine.calibration.heston_calibrator import calibrate_heston, HestonCalibrationResult
from volengine.calibration.rbergomi_calibrator import calibrate_rbergomi, RBergomiCalibrationResult

__all__ = [
    "IVQuote",
    "iv_rmse_objective",
    "calibrate_heston",
    "HestonCalibrationResult",
    "calibrate_rbergomi",
    "RBergomiCalibrationResult",
]
