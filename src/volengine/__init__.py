"""volengine — Heston / rough Bergomi volatility modeling engine.

Sub-packages
------------
- surfaces       : implied vol inversion, SVI parameterization
- models.heston  : Heston characteristic function, Carr-Madan FFT, Andersen QE
- models.rbergomi: rough Bergomi hybrid-scheme simulation and pricing
- calibration    : two-stage (global + local) calibrators for both models
- products       : European options and variance swaps
- analysis       : ATM-skew term-structure extraction
- backtesting    : OpenBB data loader and rolling recalibration engine
"""

__version__ = "0.1.0"
