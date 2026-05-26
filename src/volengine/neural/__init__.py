"""Optional: neural network calibrator for rBergomi (Horvath-Muguruza-Tomas 2019).

Train a small fully-connected net on simulated rBergomi prices, then invert
the price -> parameter map in milliseconds. Requires PyTorch; install with
`pip install volengine[neural]`.

The training/inference workflow is illustrative — production work would use
a much larger grid, control-variate-corrected MC during data generation, and
proper uncertainty quantification.
"""

# Lazy import: don't fail at package import time if torch is absent.
# generate_training_data + TrainingGrid live in data_generation (torch-free);
# NeuralCalibrator + train_calibrator live in nn_calibrator (require torch).
try:
    from volengine.neural.data_generation import TrainingGrid, generate_training_data
    from volengine.neural.nn_calibrator import NeuralCalibrator, train_calibrator
    __all__ = [
        "TrainingGrid",
        "generate_training_data",
        "NeuralCalibrator",
        "train_calibrator",
    ]
except ImportError:  # pragma: no cover
    # torch missing — expose only the torch-free pieces.
    from volengine.neural.data_generation import TrainingGrid, generate_training_data
    __all__ = ["TrainingGrid", "generate_training_data"]
