import numpy as np
import pandas as pd


def compute_class_weights(labels: pd.Series) -> dict[int, float]:
    """Per-class weight, log-damped inverse frequency, normalized to a max of 1."""
    counts = labels.value_counts()
    weights = len(labels) / (len(counts) * counts)
    weights = np.log1p(weights) / np.log1p(weights).max()
    return weights.to_dict()
