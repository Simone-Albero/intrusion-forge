import numpy as np
from matplotlib.axes import Axes

from .base import Plot, _apply_labels, _ensure_ax, _finalize, _format_value


def confusion_matrix_plot(
    cm: np.ndarray,
    *,
    class_names: list[str] | None = None,
    normalize: str | None = "row",
    cmap: str = "Blues",
    show_colorbar: bool = True,
    extra_metrics: dict[str, float] | None = None,
    title: str = "",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    max_annotated_classes: int = 20,
    max_label_chars: int = 18,
) -> Plot | None:
    """Plot a confusion matrix with optional row/column normalization and a metrics row.

    Layout adapts to class count: figsize grows with it, and in-cell annotations
    shrink and are dropped above `max_annotated_classes` (the colorbar remains).
    """
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("`cm` must be a square 2D array (n_classes x n_classes).")
    if normalize not in ("row", "col", None):
        raise ValueError("`normalize` must be 'row', 'col', or None.")

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    elif len(class_names) != n_classes:
        raise ValueError("`class_names` length must match cm.shape[0].")
    display_names = [
        n if len(n) <= max_label_chars else n[: max_label_chars - 1] + "…"
        for n in class_names
    ]
    if figsize is None:
        side = max(5.0, 0.55 * n_classes + 1.5)
        figsize = (side, side)

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

    if n_classes <= max_annotated_classes:
        threshold = vmax / 2.0
        cell_fontsize = max(6.0, 12.0 - 0.35 * n_classes)
        for i in range(n_classes):
            for j in range(n_classes):
                text = _format_value(display[i, j], kind=cell_kind)
                if text.startswith("0."):
                    text = text[1:]
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize,
                    color="white" if matrix[i, j] > threshold else "black",
                )

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(display_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(display_names)
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
