from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

_METRIC_FNS: list[tuple[str, Callable]] = [
    ("precision", precision_score),
    ("recall", recall_score),
    ("f1", f1_score),
]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scalar_prefix: str = "scalar/testing",
) -> tuple[dict[str, float], dict]:
    """Compute scalar and full classification metrics from predictions.

    Returns a tuple (scalars, full):
      - `scalars`: flat dict with prefixed keys (e.g. "scalar/testing/accuracy"),
        suitable for splatting into a LogBundle.
      - `full`: nested dict with the raw metric names + per-class arrays.
    """
    acc = float(accuracy_score(y_true, y_pred))
    scalars: dict[str, float] = {f"{scalar_prefix}/accuracy": acc}
    full: dict = {"accuracy": acc}

    for avg in ("macro", "weighted"):
        for name, fn in _METRIC_FNS:
            val = float(fn(y_true, y_pred, average=avg, zero_division=0))
            scalars[f"{scalar_prefix}/{name}_{avg}"] = val
            full[f"{name}_{avg}"] = val

    for name, fn in _METRIC_FNS:
        full[f"{name}_per_class"] = fn(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()

    return scalars, full


def _cluster_error_rates(
    clusters: np.ndarray, error_mask: np.ndarray
) -> dict[str, dict]:
    """Return {cluster_id: {n_error, n_total, error_rate}} sorted by error_rate desc."""
    failed = clusters[error_mask]
    stats: dict[str, dict] = {}
    for c in np.unique(clusters):
        n_total = int((clusters == c).sum())
        n_error = int((failed == c).sum())
        stats[str(c)] = {
            "n_error": n_error,
            "n_total": n_total,
            "error_rate": (n_error / n_total) if n_total > 0 else None,
        }
    return dict(
        sorted(stats.items(), key=lambda x: x[1]["error_rate"] or 0.0, reverse=True)
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    clusters: np.ndarray | None = None,
) -> dict:
    """Per-class prediction quality and cluster-level error rates.

    Returns:
        {
            "classes": {
                "<label>": {
                    "tot_failures", "tot_samples", "failure_rate",
                    "mean_confidence", "cluster_in_fn", "cluster_in_tp"
                }, ...
            },
            "clusters": {
                "global":   {<cluster_id>: {n_error, n_total, error_rate}, ...} | None,
                "by_class": {<label>: {<cluster_id>: {...}}, ...}              | None,
            },
        }

    `confidences` may be a 1D max-prob array or a 2D (n_samples, n_classes)
    probability matrix; in the latter case the max per row is used.
    """
    confidences = np.asarray(confidences)
    if confidences.ndim == 2:
        confidences = confidences.max(axis=1)

    has_cluster = clusters is not None
    global_error_mask = y_true != y_pred

    cluster_errors_total = (
        _cluster_error_rates(clusters, global_error_mask) if has_cluster else None
    )
    cluster_errors_by_class: dict[str, dict] | None = {} if has_cluster else None

    classes: dict[str, dict] = {}
    for label in np.unique(y_true):
        mask = y_true == label
        n_total = int(mask.sum())
        n_errors = int((y_true[mask] != y_pred[mask]).sum())
        error_mask = mask & global_error_mask

        if has_cluster:
            wrong_preds = y_pred[error_mask]
            wrong_clusters = clusters[error_mask]
            cluster_in_fn = {
                str(cls): np.unique(wrong_clusters[wrong_preds == cls]).tolist()
                for cls in np.unique(wrong_preds)
            }
            tp_clusters = clusters[mask & ~global_error_mask]
            cluster_in_tp = np.unique(tp_clusters).tolist()

            class_clusters = clusters[mask]
            cluster_errors_by_class[str(label)] = _cluster_error_rates(
                class_clusters, error_mask[mask]
            )
        else:
            cluster_in_fn = cluster_in_tp = None

        classes[str(label)] = {
            "tot_failures": n_errors,
            "tot_samples": n_total,
            "failure_rate": n_errors / n_total if n_total > 0 else None,
            "mean_confidence": float(confidences[mask].mean()) if n_total > 0 else None,
            "cluster_in_fn": cluster_in_fn,
            "cluster_in_tp": cluster_in_tp,
        }

    classes = dict(
        sorted(
            classes.items(), key=lambda x: x[1]["failure_rate"] or 0.0, reverse=True
        )
    )

    return {
        "classes": classes,
        "clusters": {
            "global": cluster_errors_total,
            "by_class": cluster_errors_by_class,
        },
    }
