import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.domain.plot.base import Plot
from src.domain.plot.metrics import confusion_matrix_plot
from src.domain.plot.primitives import bar_plot, line_plot, scatter_plot
from src.domain.plot.style import extended_palette
from src.domain.projection import (
    TSNE_MAX_SAMPLES,
    TSNE_MIN_SAMPLES,
    stratified_subsample,
    tsne_projection,
)


def training_history_figures(history: dict[str, list[float]]) -> dict[str, Plot]:
    """One line plot per scalar in the per-step DL training history, keyed by scalar name."""
    return {
        f"training_{name}_curve": line_plot(
            {name: values}, y_label=name, show_legend=False
        )
        for name, values in history.items()
        if values
    }


def _projection_selection(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: dict[int, str],
    *,
    pool: np.ndarray | None,
) -> tuple[np.ndarray, dict] | tuple[None, None]:
    """Sampled rows, from `pool` (every row when None), of the fewest classes (≥ 2)
    holding 90% of the errors, and their names.

    The classes are chosen on every row's errors, the sample on the pool's rows.
    (None, None) when fewer than TSNE_MIN_SAMPLES rows survive subsampling.
    """
    classes = np.unique(y_true)
    is_error = y_pred != y_true
    total_errors = int(is_error.sum())
    errors_per_class = {int(c): int((is_error & (y_true == c)).sum()) for c in classes}
    shown_classes: list[int] = []
    cumulative = 0
    for c in sorted(errors_per_class, key=errors_per_class.get, reverse=True):
        if (
            len(shown_classes) >= 2
            and total_errors
            and cumulative >= 0.9 * total_errors
        ):
            break
        shown_classes.append(c)
        cumulative += errors_per_class[c]
    if not shown_classes:
        shown_classes = [int(c) for c in classes]

    candidate_rows = np.flatnonzero(np.isin(y_true, shown_classes))
    if pool is not None:
        candidate_rows = np.intersect1d(candidate_rows, pool)
    # Fixed seed so "raw" and "latent" draw the same rows: y_true/y_pred are identical
    # for both, being the same model's predictions on the same test rows.
    sampled = stratified_subsample(
        y_true[candidate_rows], n_samples=TSNE_MAX_SAMPLES, random_state=42
    )
    shown_rows = candidate_rows[sampled]
    if len(shown_rows) < TSNE_MIN_SAMPLES:
        return None, None
    return shown_rows, {c: class_names.get(c, str(c)) for c in shown_classes}


def _scatter_projection(
    X_shown: np.ndarray, y_true_shown: np.ndarray, y_pred_shown: np.ndarray, names: dict
) -> Plot:
    """t-SNE scatter of an already-selected row subset."""
    return scatter_plot(
        tsne_projection(X_shown, n_components=2),
        y_true_shown,
        highlight_mask=y_pred_shown != y_true_shown,
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
    pool: np.ndarray | None,
) -> dict[str, Plot]:
    """Confusion matrix, per-class F1 and the raw t-SNE of rows drawn from `pool`."""
    cm_names = [class_names.get(int(c), str(c)) for c in cm_classes]
    figures: dict[str, Plot] = {
        "confusion_matrix": confusion_matrix_plot(cm, class_names=cm_names)
    }

    observed = np.unique(y_true)
    f1_per_class = f1_score(
        y_true, y_pred, labels=observed, average=None, zero_division=0
    )
    f1_by_class = {
        class_names.get(int(c), str(c)): float(v)
        for c, v in zip(observed, f1_per_class)
    }
    figures["f1_per_class"] = bar_plot(
        list(f1_by_class.keys()),
        list(f1_by_class.values()),
        orientation="v",
        color=extended_palette(len(f1_by_class)),
        sort=None,
        ylim=(0, 1),
    )

    shown_rows, names = _projection_selection(y_true, y_pred, class_names, pool=pool)
    if shown_rows is not None:
        figures["raw"] = _scatter_projection(
            eval_df.iloc[shown_rows][feat_cols].to_numpy(),
            y_true[shown_rows],
            y_pred[shown_rows],
            names,
        )
    return figures


def latent_figures(
    embedding: np.ndarray,
    rows: np.ndarray,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: dict[int, str],
) -> dict[str, Plot]:
    """t-SNE of the latent space, keyed `latent`; row `rows[i]` has `embedding[i]`."""
    shown_rows, names = _projection_selection(y_true, y_pred, class_names, pool=rows)
    if shown_rows is None:
        return {}
    return {
        "latent": _scatter_projection(
            embedding[np.searchsorted(rows, shown_rows)],
            y_true[shown_rows],
            y_pred[shown_rows],
            names,
        )
    }
