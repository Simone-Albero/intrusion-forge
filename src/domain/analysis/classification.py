from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.multiclass import unique_labels

from src.domain.analysis.failure import is_failure
from src.domain.analysis.grouping import RowsBy

_METRICS: list[tuple[str, Callable]] = [
    ("precision", precision_score),
    ("recall", recall_score),
    ("f1", f1_score),
]


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Overall accuracy, macro/weighted precision/recall/F1, plus one row per class."""
    metrics: dict = {"accuracy": float(accuracy_score(y_true, y_pred))}

    for average in ("macro", "weighted"):
        for name, metric_fn in _METRICS:
            metrics[f"{name}_{average}"] = float(
                metric_fn(y_true, y_pred, average=average, zero_division=0)
            )

    # Every class observed or predicted, passed explicitly so the rows below line up
    # with the arrays sklearn returns.
    labels = unique_labels(y_true, y_pred)
    per_class = {
        name: metric_fn(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).tolist()
        for name, metric_fn in _METRICS
    }
    metrics["classes"] = [
        {"class_id": int(c), **{name: values[i] for name, values in per_class.items()}}
        for i, c in enumerate(labels)
    ]

    return metrics


def region_failures(
    region: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, mcp: np.ndarray
) -> pd.DataFrame:
    """One row per region holding evaluated rows: its error counts, rate and mean MCP risk."""
    by_region = RowsBy(region)
    n_eval = by_region.sizes
    n_error = by_region.reduce(is_failure(y_true, y_pred), np.sum)
    return pd.DataFrame(
        {
            "region": by_region.ids,
            "n_eval": n_eval,
            "n_error": n_error,
            "failure_rate": n_error / n_eval,
            "mcp_risk": by_region.reduce(mcp),
        }
    )
