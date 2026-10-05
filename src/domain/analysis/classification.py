from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.multiclass import unique_labels

from src.domain.analysis.failure import is_failure
from src.domain.analysis.grouping import RowsBy

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


def empirical_region_rate(
    failures: pd.DataFrame, *, region_class: pd.Series
) -> pd.Series:
    """Failure rate of each region; one without rows takes its class's, then the overall."""
    counts = (
        failures.set_index("region")[["n_eval", "n_error"]]
        .reindex(region_class.index)
        .fillna(0)
    )
    by_class = counts.groupby(region_class).sum()
    class_rate = by_class["n_error"] / by_class["n_eval"].replace(0, np.nan)
    overall = counts["n_error"].sum() / counts["n_eval"].sum()
    fallback = region_class.map(class_rate).fillna(overall)
    return (counts["n_error"] / counts["n_eval"].replace(0, np.nan)).fillna(fallback)
