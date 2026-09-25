import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.domain.plot.base import Plot
from src.domain.plot.metrics import confusion_matrix_plot
from src.domain.plot.primitives import bar_plot, line_plot, scatter_plot
from src.domain.plot.style import extended_palette
from src.domain.projection import (
    TSNE_MIN_SAMPLES,
    stratified_subsample,
    tsne_projection,
)


def training_history_figures(history: dict[str, list[float]]) -> dict[str, Plot]:
    """One line plot per scalar in the per-step DL training history, keyed by scalar name."""
    return {
        f"{name}_curve": line_plot({name: values}, y_label=name, show_legend=False)
        for name, values in history.items()
        if values
    }


_TSNE_SAMPLES = 2000


def _projection_selection(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: dict[int, str]
) -> tuple[np.ndarray, dict] | tuple[None, None]:
    """Sampled rows of the fewest classes (≥ 2) holding 90% of the errors, and names.

    (None, None) when fewer than TSNE_MIN_SAMPLES rows survive subsampling.
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
    # Fixed seed so "raw" and "latent" draw the same rows on a single split, where
    # y_true/y_pred are identical.
    sub = stratified_subsample(
        y_true[prob_pos], n_samples=_TSNE_SAMPLES, random_state=42
    )
    vis_idx = prob_pos[sub]
    if len(vis_idx) < TSNE_MIN_SAMPLES:
        return None, None
    return vis_idx, {c: class_names.get(c, str(c)) for c in keep_classes}


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
    cm_classes: np.ndarray,
    class_names: dict[int, str],
) -> dict[str, Plot]:
    """Confusion matrix over `cm_classes`, F1 over the observed classes, raw t-SNE."""
    cm_names = [class_names.get(int(c), str(c)) for c in cm_classes]
    figures: dict[str, Plot] = {
        "confusion_matrix": confusion_matrix_plot(cm, class_names=cm_names)
    }

    observed = np.unique(y_true)
    f1_per_class = f1_score(
        y_true, y_pred, labels=observed, average=None, zero_division=0
    )
    f1_dict = {
        class_names.get(int(c), str(c)): float(v)
        for c, v in zip(observed, f1_per_class)
    }
    figures["f1_per_class"] = bar_plot(
        list(f1_dict.keys()),
        list(f1_dict.values()),
        orientation="v",
        color=extended_palette(len(f1_dict)),
        sort=None,
        ylim=(0, 1),
    )

    vis_idx, names = _projection_selection(y_true, y_pred, class_names)
    if vis_idx is not None:
        figures["raw"] = _scatter_projection(
            eval_df.iloc[vis_idx][feat_cols].to_numpy(),
            y_true[vis_idx],
            y_pred[vis_idx],
            names,
        )
    return figures


def latent_figures(
    folds: list[tuple[str, np.ndarray, np.ndarray | None]],
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: dict[int, str],
) -> dict[str, Plot]:
    """One t-SNE latent scatter per fold, keyed `{prefix}latent`.

    `folds` is (prefix, eval_idx, embedding) per fold, with eval_idx indexing `y_true`
    and `y_pred`. The K latent spaces come from K different models and are not mutually
    aligned, so they stay fold-scoped instead of being merged into one figure.
    """
    figures: dict[str, Plot] = {}
    for prefix, eval_idx, embedding in folds:
        if embedding is None:
            continue
        y_true_fold = y_true[eval_idx]
        y_pred_fold = y_pred[eval_idx]
        vis_idx, names = _projection_selection(y_true_fold, y_pred_fold, class_names)
        if vis_idx is not None:
            figures[f"{prefix}latent"] = _scatter_projection(
                embedding[vis_idx], y_true_fold[vis_idx], y_pred_fold[vis_idx], names
            )
    return figures
