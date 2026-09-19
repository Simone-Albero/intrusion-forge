import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from pipelines.classify_training import ClassifyContext, SplitPlan, SplitPredictions
from src.core.io import save_df
from src.core.log import LogBundle
from src.core.utils import timed
from src.domain.analysis.confidence import mcp_risk
from src.domain.analysis.failure import is_failure
from src.domain.plot.base import Plot
from src.domain.plot.metrics import confusion_matrix_plot
from src.domain.plot.primitives import bar_plot, scatter_plot
from src.domain.plot.style import extended_palette
from src.domain.projection import stratified_subsample, tsne_projection

logger = logging.getLogger(__name__)


_METRIC_FNS: list[tuple[str, Callable]] = [
    ("precision", precision_score),
    ("recall", recall_score),
    ("f1", f1_score),
]


def _compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Overall accuracy, macro/weighted precision/recall/F1, plus per-class arrays."""
    full: dict = {"accuracy": float(accuracy_score(y_true, y_pred))}

    for avg in ("macro", "weighted"):
        for name, fn in _METRIC_FNS:
            full[f"{name}_{avg}"] = float(
                fn(y_true, y_pred, average=avg, zero_division=0)
            )

    for name, fn in _METRIC_FNS:
        full[f"{name}_per_class"] = fn(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()

    return full


def _cluster_error_rates(
    clusters: np.ndarray,
    error_mask: np.ndarray,
    extra_scores: dict[str, np.ndarray] | None = None,
) -> dict[str, dict]:
    """Per-cluster error counts, rate and mean extra scores, sorted by rate descending."""
    failed = clusters[error_mask]
    extra_scores = extra_scores or {}
    stats: dict[str, dict] = {}
    for c in np.unique(clusters):
        mask = clusters == c
        n_total = int(mask.sum())
        n_error = int((failed == c).sum())
        stats[str(c)] = {
            "n_error": n_error,
            "n_total": n_total,
            "error_rate": (n_error / n_total) if n_total > 0 else None,
            **{
                name: float(scores[mask].mean()) if n_total > 0 else None
                for name, scores in extra_scores.items()
            },
        }
    return dict(
        sorted(stats.items(), key=lambda x: x[1]["error_rate"] or 0.0, reverse=True)
    )


def _evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    clusters: np.ndarray | None = None,
) -> dict:
    """Per-class prediction quality plus per-cluster error rates and mean risk scores."""
    y_proba = np.asarray(y_proba)
    mcp = mcp_risk(y_proba)
    confidences = 1.0 - mcp

    has_cluster = clusters is not None
    global_error_mask = is_failure(y_true, y_pred)

    cluster_errors_total = (
        _cluster_error_rates(
            clusters,
            global_error_mask,
            extra_scores={"mcp_risk": mcp},
        )
        if has_cluster
        else None
    )
    cluster_errors_by_class: dict[str, dict] | None = {} if has_cluster else None

    classes: dict[str, dict] = {}
    for label in np.unique(y_true):
        mask = y_true == label
        n_total = int(mask.sum())
        n_errors = int(is_failure(y_true[mask], y_pred[mask]).sum())
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
            "mean_confidence": (
                float(confidences[mask].mean()) if n_total > 0 else None
            ),
            "cluster_in_fn": cluster_in_fn,
            "cluster_in_tp": cluster_in_tp,
        }

    classes = dict(
        sorted(
            classes.items(),
            key=lambda x: x[1]["failure_rate"] or 0.0,
            reverse=True,
        )
    )

    return {
        "classes": classes,
        "clusters": {
            "global": cluster_errors_total,
            "by_class": cluster_errors_by_class,
        },
    }


def _per_sample_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    clusters: np.ndarray,
) -> pd.DataFrame:
    """Per-sample risk table (label-free scores plus labels) for the instance baselines."""
    yp = np.asarray(y_proba)
    return pd.DataFrame(
        {
            "cluster": np.asarray(clusters),
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "mcp_risk": mcp_risk(yp),
        }
    ).astype({"mcp_risk": "float32"})


_TSNE_MIN_POINTS = 6  # tsne_projection's perplexity floor (5) requires n_samples > 5


def _projection_figure(
    space: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_mapping: dict,
    n_samples: int = 2000,
) -> Plot | None:
    """t-SNE scatter of `space`, sampled to prioritize the classes with the most errors."""
    classes = np.unique(y_true)
    correct = y_pred == y_true
    mis = ~correct
    total_mis = int(mis.sum())
    mis_per_class = {int(c): int((mis & (y_true == c)).sum()) for c in classes}
    keep_classes: list[int] = []
    cumulative = 0
    for c in sorted(mis_per_class, key=mis_per_class.get, reverse=True):
        if len(keep_classes) >= 2 and total_mis and cumulative >= 0.9 * total_mis:
            break
        keep_classes.append(c)
        cumulative += mis_per_class[c]
    if not keep_classes:
        keep_classes = [int(c) for c in classes]

    names = {c: label_mapping.get(str(c), str(c)) for c in keep_classes}
    prob_pos = np.flatnonzero(np.isin(y_true, keep_classes))
    # Fixed seed (matching tsne_projection's own default) so "raw" and "latent" draw
    # the same visualized rows on a single split, where y_true/y_pred are identical.
    sub = stratified_subsample(
        y_true[prob_pos], n_samples=n_samples, stratify=False, random_state=42
    )
    vis_idx = prob_pos[sub]
    if len(vis_idx) < _TSNE_MIN_POINTS:
        return None

    return scatter_plot(
        tsne_projection(space[vis_idx], n_components=2),
        y_true[vis_idx],
        highlight_mask=~correct[vis_idx],
        names=names,
        marker_size=35.0,
        marker_alpha=0.85,
        legend_on_top=True,
    )


def _build_test_figures(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_mapping: dict,
    n_samples: int = 2000,
) -> dict[str, Plot]:
    """Confusion matrix, per-class F1 bar and t-SNE scatter of raw features."""
    figures: dict[str, Plot] = {}

    classes = np.unique(y_true)
    class_names = [label_mapping.get(str(int(c)), str(c)) for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    figures["figure/testing/confusion_matrix"] = confusion_matrix_plot(
        cm, class_names=class_names, normalize=None
    )

    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_dict = {
        label_mapping.get(str(int(c)), str(c)): float(v)
        for c, v in zip(classes, f1_per_class)
    }
    figures["figure/testing/f1_per_class"] = bar_plot(
        list(f1_dict.keys()),
        list(f1_dict.values()),
        orientation="v",
        color=extended_palette(len(f1_dict)),
        sort=None,
        ylim=(0, 1),
    )

    raw_fig = _projection_figure(X, y_true, y_pred, label_mapping, n_samples)
    if raw_fig is not None:
        figures["figure/testing/raw"] = raw_fig
    return figures


def _latent_figures(
    plan: SplitPlan, predictions: SplitPredictions, label_col: str, label_mapping: dict
) -> dict[str, Plot]:
    """One t-SNE latent scatter per split, keyed under that split's own prefix.

    The K latent spaces come from K different models and are not mutually aligned, so
    they stay fold-scoped instead of being merged into one figure.
    """
    figures: dict[str, Plot] = {}
    for split, embedding in zip(plan.splits, predictions.embeddings):
        if embedding is None:
            continue
        eval_df = plan.universe.iloc[split.eval_idx]
        figure = _projection_figure(
            embedding,
            eval_df[label_col].to_numpy(),
            predictions.y_pred[split.eval_idx],
            label_mapping,
        )
        if figure is not None:
            figures[f"figure/testing/{split.fold_prefix}latent"] = figure
    return figures


@timed
def publish_evaluation(
    context: ClassifyContext, plan: SplitPlan, predictions: SplitPredictions
) -> None:
    """Turn the merged out-of-fold predictions into metrics, figures and per-sample dumps.

    A single split's `eval_idx` covers only the test rows, so the merged evaluation is
    test-only; k-fold's `eval_idx` values partition the whole universe, so it is not.
    """
    label_col, df_meta = context.label_col, context.df_meta
    label_mapping = df_meta["label_mapping"]

    eval_pos = np.flatnonzero(predictions.covered)
    eval_universe = plan.universe.iloc[eval_pos]
    y_true = eval_universe[label_col].to_numpy()
    y_pred = predictions.y_pred[eval_pos]
    y_proba = predictions.y_proba[eval_pos]
    clusters = (
        eval_universe["cluster"].to_numpy()
        if "cluster" in eval_universe.columns
        else None
    )

    full_metrics = {
        **_compute_classification_metrics(y_true, y_pred),
        "eval_mode": plan.mode,
    }
    pred_infos = {
        **_evaluate_predictions(y_true, y_pred, y_proba, clusters),
        "eval_mode": plan.mode,
    }
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true), normalize="true")
    figures = {
        **_build_test_figures(
            eval_universe[context.feat_cols].to_numpy(), y_true, y_pred, label_mapping
        ),
        **_latent_figures(plan, predictions, label_col, label_mapping),
    }
    if clusters is not None:
        save_df(
            _per_sample_scores(y_true, y_pred, y_proba, clusters),
            context.paths.outputs / "analysis/predictions/oof_samples.parquet",
        )

    context.bus.publish(
        LogBundle.from_dict(
            {
                **figures,
                "json/testing/summary": full_metrics,
                "json/analysis/predictions/clusters": pred_infos,
                "pickle/analysis/confusion_matrices/test": cm,
            }
        )
    )
    if plan.is_kfold:
        logger.info(
            "k-fold OOF evaluation: %d samples over %d folds",
            len(eval_pos),
            len(plan.splits),
        )
