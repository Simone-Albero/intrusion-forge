import matplotlib.pyplot as plt
import numpy as np

from .base import Plot, _apply_labels, _fig_to_plot, _format_value


def confusion_matrix_plot(
    cm: np.ndarray,
    *,
    class_names: list[str] | None = None,
    cmap: str = "Blues",
    show_colorbar: bool = True,
    title: str = "",
    figsize: tuple[float, float] | None = None,
    max_annotated_classes: int = 20,
    max_label_chars: int = 18,
) -> Plot:
    """Plot a confusion matrix, with a metrics row."""
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("`cm` must be a square 2D array (n_classes x n_classes).")

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
    is_normalized = matrix.max() <= 1.0 and not np.all(matrix == matrix.astype(int))
    cell_kind = "normalized_cm" if is_normalized else "count"

    fig, ax = plt.subplots(figsize=figsize)

    vmax = 1.0 if is_normalized else float(matrix.max()) if matrix.size else 1.0
    im = ax.imshow(
        matrix, cmap=cmap, interpolation="nearest", aspect="equal", vmin=0.0, vmax=vmax
    )

    if n_classes <= max_annotated_classes:
        threshold = vmax / 2.0
        cell_fontsize = max(6.0, 12.0 - 0.35 * n_classes)
        for i in range(n_classes):
            for j in range(n_classes):
                text = _format_value(matrix[i, j], kind=cell_kind)
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

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Proportion" if is_normalized else "Count")

    _apply_labels(ax, x_label="Predicted label", y_label="True label", title=title)
    return _fig_to_plot(fig)
