from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.multiclass import unique_labels

from src.domain.analysis.failure import is_failure
from src.domain.analysis.grouping import RowGroups

_METRIC_FNS: list[tuple[str, Callable]] = [
    ("precision", precision_score),
    ("recall", recall_score),
    ("f1", f1_score),
]


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Overall accuracy, macro/weighted precision/recall/F1, plus one row per class."""
    full: dict = {"accuracy": float(accuracy_score(y_true, y_pred))}

    for avg in ("macro", "weighted"):
        for name, fn in _METRIC_FNS:
            full[f"{name}_{avg}"] = float(
                fn(y_true, y_pred, average=avg, zero_division=0)
            )

    # Every class observed or predicted, passed explicitly so the rows below line up
    # with the arrays sklearn returns.
    labels = unique_labels(y_true, y_pred)
    per_class = {
        name: fn(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist()
        for name, fn in _METRIC_FNS
    }
    full["classes"] = [
        {"class_id": int(c), **{name: values[i] for name, values in per_class.items()}}
        for i, c in enumerate(labels)
    ]

    return full


def region_failures(
    region: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, mcp: np.ndarray
) -> pd.DataFrame:
    """One row per region holding evaluated rows: its error counts, rate and mean MCP risk."""
    groups = RowGroups(region)
    n_eval = groups.sizes
    n_error = groups.reduce(is_failure(y_true, y_pred), np.sum)
    return pd.DataFrame(
        {
            "region": groups.ids,
            "n_eval": n_eval,
            "n_error": n_error,
            "failure_rate": n_error / n_eval,
            "mcp_risk": groups.reduce(mcp),
        }
    )
