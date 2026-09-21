import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.domain.projection import stratified_subsample, tsne_projection

from .base import Plot
from .metrics import confusion_matrix_plot
from .primitives import bar_plot, line_plot, scatter_plot
from .style import extended_palette


def training_history_figures(history: dict[str, list[float]]) -> dict[str, Plot]:
    """One line plot per scalar in the per-step DL training history, keyed by scalar name."""
    return {
        f"{name}_curve": line_plot({name: values}, y_label=name, show_legend=False)
        for name, values in history.items()
        if values
    }


_TSNE_MIN_POINTS = 6  # tsne_projection's perplexity floor (5) requires n_samples > 5


def _projection_selection(
    y_true: np.ndarray, y_pred: np.ndarray, label_mapping: dict, n_samples: int
) -> tuple[np.ndarray, dict] | tuple[None, None]:
    """Row positions to visualize (most-missed classes first, subsampled) and their names.

    (None, None) when fewer than _TSNE_MIN_POINTS rows survive subsampling.
    """
    classes = np.unique(y_true)
    mis = y_pred != y_true
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

    prob_pos = np.flatnonzero(np.isin(y_true, keep_classes))
    # Fixed seed (matching tsne_projection's own default) so "raw" and "latent" draw
    # the same visualized rows on a single split, where y_true/y_pred are identical.
    sub = stratified_subsample(y_true[prob_pos], n_samples=n_samples, random_state=42)
    vis_idx = prob_pos[sub]
    if len(vis_idx) < _TSNE_MIN_POINTS:
        return None, None
    return vis_idx, {c: label_mapping.get(str(c), str(c)) for c in keep_classes}


def _scatter_projection(
    space_vis: np.ndarray, y_true_vis: np.ndarray, y_pred_vis: np.ndarray, names: dict
) -> Plot:
    """t-SNE scatter of an already-selected row subset."""
    return scatter_plot(
        tsne_projection(space_vis, n_components=2),
        y_true_vis,
        highlight_mask=y_pred_vis != y_true_vis,
        names=names,
        marker_size=35.0,
        marker_alpha=0.85,
        legend_on_top=True,
    )


def build_test_figures(
    eval_df: pd.DataFrame,
    feat_cols: list[str],
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cm: np.ndarray,
    classes: np.ndarray,
    label_mapping: dict,
    n_samples: int = 2000,
) -> dict[str, Plot]:
    """Confusion matrix, per-class F1 bar and t-SNE scatter of raw features.

    Takes `cm`/`classes` already computed rather than recomputing them, since the caller
    needs the same confusion matrix for its own pickle dump.
    """
    class_names = [label_mapping.get(str(int(c)), str(c)) for c in classes]
    figures: dict[str, Plot] = {
        "confusion_matrix": confusion_matrix_plot(cm, class_names=class_names)
    }

    f1_per_class = f1_score(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    f1_dict = {
        label_mapping.get(str(int(c)), str(c)): float(v)
        for c, v in zip(classes, f1_per_class)
    }
    figures["f1_per_class"] = bar_plot(
        list(f1_dict.keys()),
        list(f1_dict.values()),
        orientation="v",
        color=extended_palette(len(f1_dict)),
        sort=None,
        ylim=(0, 1),
    )

    vis_idx, names = _projection_selection(y_true, y_pred, label_mapping, n_samples)
    if vis_idx is not None:
        figures["raw"] = _scatter_projection(
            eval_df.iloc[vis_idx][feat_cols].to_numpy(),
            y_true[vis_idx],
            y_pred[vis_idx],
            names,
        )
    return figures


def latent_figures(
    splits: list[tuple[str, np.ndarray, np.ndarray | None]],
    *,
    universe_labels: np.ndarray,
    universe_y_pred: np.ndarray,
    label_mapping: dict,
) -> dict[str, Plot]:
    """One t-SNE latent scatter per split, keyed `{fold_prefix}latent`.

    `splits` is (fold_prefix, eval_idx, embedding) per split. The K latent spaces come
    from K different models and are not mutually aligned, so they stay fold-scoped
    instead of being merged into one figure.
    """
    figures: dict[str, Plot] = {}
    for fold_prefix, eval_idx, embedding in splits:
        if embedding is None:
            continue
        y_true_split = universe_labels[eval_idx]
        y_pred_split = universe_y_pred[eval_idx]
        vis_idx, names = _projection_selection(
            y_true_split, y_pred_split, label_mapping, 2000
        )
        if vis_idx is not None:
            figures[f"{fold_prefix}latent"] = _scatter_projection(
                embedding[vis_idx], y_true_split[vis_idx], y_pred_split[vis_idx], names
            )
    return figures
