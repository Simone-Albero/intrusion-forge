from typing import Callable

import numpy as np
import pandas as pd
import shap


def kernel_shap_values(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    background: pd.DataFrame,
    samples: pd.DataFrame,
) -> np.ndarray:
    """Model-agnostic SHAP values via KernelExplainer.

    `predict_fn` maps a 2D feature array to an ``(n, n_outputs)`` probability
    matrix. Returns values shaped ``(n_samples, n_features, n_outputs)`` (a 2D
    single-output result is promoted to 3D).
    """
    explainer = shap.KernelExplainer(predict_fn, background)
    values = np.asarray(explainer(samples).values)
    if values.ndim == 2:
        values = values[:, :, None]
    return values
