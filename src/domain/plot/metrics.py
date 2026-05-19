import numpy as np
from matplotlib.axes import Axes

from .base import Plot, _apply_labels, _ensure_ax, _finalize, _format_value
from .style import NEUTRAL_COLOR, PALETTE


def confusion_matrix_plot(
    cm: np.ndarray,
    *,
    class_names: list[str] | None = None,
    normalize: str | None = "row",
    cmap: str = "Blues",
    show_colorbar: bool = True,
    extra_metrics: dict[str, float] | None = None,
    title: str = "",
    figsize: tuple[float, float] = (5, 5),
    ax: Axes | None = None,
) -> Plot | None:
    """Plot a confusion matrix with optional row/column normalization and a metrics row."""
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("`cm` must be a square 2D array (n_classes x n_classes).")
    if normalize not in ("row", "col", None):
        raise ValueError("`normalize` must be 'row', 'col', or None.")

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    elif len(class_names) != n_classes:
        raise ValueError("`class_names` length must match cm.shape[0].")

    matrix = cm.astype(float)
    if normalize is not None:
        axis = 1 if normalize == "row" else 0
        sums = matrix.sum(axis=axis, keepdims=True)
        matrix = np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums != 0)

    is_normalized = normalize is not None or (
        cm.max() <= 1.0 and not np.all(cm == cm.astype(int))
    )
    display = matrix if is_normalized else cm.astype(float)
    cell_kind = "normalized_cm" if is_normalized else "count"

    ax, fig = _ensure_ax(ax, figsize)

    vmax = 1.0 if is_normalized else float(matrix.max()) if matrix.size else 1.0
    im = ax.imshow(
        matrix, cmap=cmap, interpolation="nearest", aspect="equal", vmin=0.0, vmax=vmax
    )

    threshold = vmax / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                _format_value(display[i, j], kind=cell_kind),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.grid(False)

    if show_colorbar and fig is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Proportion" if is_normalized else "Count")

    if extra_metrics:
        metrics_text = "   ".join(
            f"{name}: {_format_value(value, kind='score')}"
            for name, value in extra_metrics.items()
        )
        ax.text(
            0.5,
            -0.18,
            metrics_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(
                facecolor="#f5f5f5", edgecolor="#cccccc", boxstyle="round,pad=0.3"
            ),
        )

    _apply_labels(ax, x_label="Predicted label", y_label="True label", title=title)
    return _finalize(fig)


def roc_plot(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    *,
    color: str | None = None,
    show_baseline: bool = True,
    title: str = "",
    figsize: tuple[float, float] = (5.5, 5.5),
    ax: Axes | None = None,
) -> Plot | None:
    """ROC curve with AUC annotation and optional random-classifier baseline."""
    ax, fig = _ensure_ax(ax, figsize)

    curve_color = color if color is not None else PALETTE[0]
    ax.plot(fpr, tpr, color=curve_color, linewidth=1.8)
    ax.fill_between(fpr, 0, tpr, color=curve_color, alpha=0.15)

    if show_baseline:
        ax.plot([0, 1], [0, 1], color=NEUTRAL_COLOR, linewidth=1.0, linestyle=":")

    ax.text(
        0.05,
        0.95,
        f"AUC = {_format_value(auc_score, kind='score')}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="both", alpha=0.15, linewidth=0.5)

    _apply_labels(
        ax, x_label="False Positive Rate", y_label="True Positive Rate", title=title
    )
    return _finalize(fig)
