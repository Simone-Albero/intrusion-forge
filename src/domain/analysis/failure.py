import numpy as np


def is_failure(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Boolean mask marking each sample where the prediction differs from the true label."""
    return np.asarray(y_true) != np.asarray(y_pred)
