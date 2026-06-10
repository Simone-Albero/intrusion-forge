from typing import Callable

import numpy as np
import pandas as pd
import shap


def summarize_background(background: pd.DataFrame, k: int) -> object:
    """Compress the background to k weighted k-means centroids (shap.kmeans).

    Returned object is accepted by KernelExplainer in place of the raw rows;
    backgrounds with at most k rows pass through unchanged.
    """
    if len(background) <= k:
        return background
    return shap.kmeans(background, k)


def kernel_shap_values(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    background: object,
    samples: pd.DataFrame,
    *,
    nsamples: int | str = "auto",
) -> np.ndarray:
    """Model-agnostic SHAP values via KernelExplainer.

    `predict_fn` maps a 2D feature array to an ``(n, n_outputs)`` probability
    matrix. `background` is a DataFrame or a `summarize_background` result;
    `nsamples` bounds the coalition samples per explained instance. Returns
    values shaped ``(n_samples, n_features, n_outputs)`` (lower-dimensional
    results are promoted to 3D).
    """
    explainer = shap.KernelExplainer(predict_fn, background)
    values = explainer.shap_values(samples, nsamples=nsamples, silent=True)
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim == 2:
        values = values[:, :, None]
    return values
