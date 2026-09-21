from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.multiclass import unique_labels

from src.domain.analysis.failure import is_failure

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

    # sklearn's per-class arrays are ordered by unique_labels(y_true, y_pred), which
    # includes labels only ever predicted: enumerating np.unique(y_true) instead would
    # shift every metric onto the wrong class.
    labels = unique_labels(y_true, y_pred)
    per_class = {
        name: fn(y_true, y_pred, average=None, zero_division=0).tolist()
        for name, fn in _METRIC_FNS
    }
    full["per_class"] = [
        {"class": int(c), **{name: values[i] for name, values in per_class.items()}}
        for i, c in enumerate(labels)
    ]

    return full


def _cluster_error_rates(
    clusters: np.ndarray,
    error_mask: np.ndarray,
    extra_scores: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """One row per cluster: error counts, rate and mean extra scores, worst rate first."""
    failed = clusters[error_mask]
    extra_scores = extra_scores or {}
    rows = []
    for c in np.unique(clusters):
        mask = clusters == c
        n_total = int(mask.sum())
        n_error = int((failed == c).sum())
        rows.append(
            {
                "cluster_id": int(c),
                "n_error": n_error,
                "n_total": n_total,
                "error_rate": (n_error / n_total) if n_total > 0 else None,
                **{
                    name: float(scores[mask].mean()) if n_total > 0 else None
                    for name, scores in extra_scores.items()
                },
            }
        )
    return sorted(rows, key=lambda r: r["error_rate"] or 0.0, reverse=True)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mcp: np.ndarray,
    clusters: np.ndarray | None = None,
) -> dict:
    """Two tables of observed failures: one row per class, one row per cluster."""
    confidences = 1.0 - mcp
    error_mask = is_failure(y_true, y_pred)

    class_rows = []
    for label in np.unique(y_true):
        mask = y_true == label
        n_total = int(mask.sum())
        n_error = int(error_mask[mask].sum())
        class_rows.append(
            {
                "class": int(label),
                "n_error": n_error,
                "n_total": n_total,
                "error_rate": n_error / n_total if n_total > 0 else None,
                "mean_confidence": (
                    float(confidences[mask].mean()) if n_total > 0 else None
                ),
            }
        )

    return {
        "classes": sorted(
            class_rows, key=lambda r: r["error_rate"] or 0.0, reverse=True
        ),
        "clusters": (
            _cluster_error_rates(clusters, error_mask, extra_scores={"mcp_risk": mcp})
            if clusters is not None
            else []
        ),
    }


def per_sample_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mcp: np.ndarray,
    clusters: np.ndarray,
) -> pd.DataFrame:
    """Per-sample risk table (label-free scores plus labels) for the instance baselines."""
    return pd.DataFrame(
        {
            "cluster": np.asarray(clusters),
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "mcp_risk": np.asarray(mcp),
        }
    ).astype({"mcp_risk": "float32"})
