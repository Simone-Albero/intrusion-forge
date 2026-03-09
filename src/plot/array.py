from typing import Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def confusion_matrix_to_plot(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    class_names: list[str] | None = None,
    cmap: str = "Blues",
    figsize: tuple[float, float] = (8, 6),
    show_colorbar: bool = True,
    values_decimals: int | None = None,
) -> plt.Figure:
    """Plot a confusion matrix, handling both raw counts and normalized values."""
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("`cm` must be a square 2D array (n_classes x n_classes).")

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    elif len(class_names) != n_classes:
        raise ValueError("`class_names` length must match cm.shape[0].")

    is_normalized = (cm.max() <= 1.0) or not np.all(cm == cm.astype(int))

    if values_decimals is not None:
        values_format = f".{values_decimals}f"
    elif is_normalized:
        values_format = ".2f"
    else:
        values_format = "d"  # integer format for raw counts

    fig, ax = plt.subplots(figsize=figsize)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        ax=ax,
        cmap=cmap,
        values_format=values_format,
        colorbar=show_colorbar,
        im_kw={"interpolation": "nearest"},
        xticks_rotation=45,
        include_values=True,
    )

    if show_colorbar and disp.im_ is not None:
        colorbar = disp.im_.colorbar
        if colorbar is not None:
            colorbar.set_label(
                "Proportion" if is_normalized else "Count",
                fontsize=10,
            )

    if is_normalized and "normaliz" not in title.lower():
        title = f"{title} (Normalized)"

    ax.set_title(title, fontsize=14, pad=16)
    ax.set_xlabel("Predicted label", labelpad=10)
    ax.set_ylabel("True label", labelpad=10)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_aspect("equal")

    fig.tight_layout()
    return fig


def roc_auc_plot(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> plt.Figure:
    """Plot ROC curve with AUC annotation."""
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})", color="blue")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    return fig


# High-contrast palettes
_FILL_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]

_OUTLINE_COLORS = [
    "#FF0000",
    "#000000",
    "#0000FF",
    "#FF00FF",
    "#00CCCC",
    "#FF8800",
    "#6600CC",
    "#00CC66",
    "#FF0066",
    "#CCCC00",
]


def _get_fill_cmap(n: int):
    """Return a ListedColormap with n high-contrast fill colors."""
    colors = (_FILL_COLORS * ((n // len(_FILL_COLORS)) + 1))[:n]
    return mcolors.ListedColormap(colors)


def _validate_inputs(
    X: np.ndarray,
    y_1: np.ndarray,
    y_2: Union[np.ndarray, None],
) -> None:
    n = len(X)
    if X.ndim != 2 or X.shape[1] not in (2, 3):
        raise ValueError("X must have shape (n_samples, 2) or (n_samples, 3).")
    if len(y_1) != n:
        raise ValueError("X and y_1 must have the same number of samples.")
    if y_2 is not None:
        if len(y_2) != n:
            raise ValueError("X and y_2 must have the same number of samples.")
        n_unique_y2 = len(np.unique(y_2))
        if n_unique_y2 > len(_OUTLINE_COLORS):
            raise ValueError(
                f"y_2 has {n_unique_y2} unique values; maximum supported is {len(_OUTLINE_COLORS)}."
            )


def _make_legend_proxy(fill_color, edge_color, label):
    """Create a proxy artist for the legend."""
    return plt.Line2D(
        [],
        [],
        marker="o",
        linestyle="",
        markersize=9,
        markerfacecolor=fill_color,
        markeredgecolor=edge_color,
        markeredgewidth=2.0,
        label=label,
    )


def samples_plot(
    X: np.ndarray,
    y_1: Union[list, np.ndarray],
    y_2: Union[list, np.ndarray, None] = None,
) -> plt.Figure:
    """Plot 2D or 3D samples with fill-color and optional outline-color encoding.

    Args:
        X:   Array of shape (n_samples, 2) or (n_samples, 3).
        y_1: Integer labels mapped to fill color.
        y_2: Optional integer labels mapped to outline/edge color.

    Returns:
        Matplotlib Figure object.
    """
    X = np.asarray(X)
    y_1 = np.asarray(y_1)
    y_2 = np.asarray(y_2) if y_2 is not None else None

    _validate_inputs(X, y_1, y_2)

    is_3d = X.shape[1] == 3
    unique_y1 = np.unique(y_1)
    cmap = _get_fill_cmap(len(unique_y1))
    fill_map = {label: cmap(i) for i, label in enumerate(unique_y1)}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)

    scatter_kwargs = dict(s=120, alpha=0.75, linewidths=1.8)
    legend_proxies = []

    if y_2 is None:
        # --- Single label: only fill color ---
        for label in unique_y1:
            mask = y_1 == label
            coords = (X[mask, 0], X[mask, 1]) + ((X[mask, 2],) if is_3d else ())
            ax.scatter(
                *coords, color=fill_map[label], edgecolors="#222222", **scatter_kwargs
            )
            legend_proxies.append(
                _make_legend_proxy(fill_map[label], "#222222", f"Class {label}")
            )

    else:
        # --- Two labels: fill color + outline color ---
        unique_y2 = np.unique(y_2)
        outline_map = {label: _OUTLINE_COLORS[i] for i, label in enumerate(unique_y2)}

        for label1 in unique_y1:
            for label2 in unique_y2:
                mask = (y_1 == label1) & (y_2 == label2)
                if not np.any(mask):
                    continue
                coords = (X[mask, 0], X[mask, 1]) + ((X[mask, 2],) if is_3d else ())
                ax.scatter(
                    *coords,
                    color=fill_map[label1],
                    edgecolors=outline_map[label2],
                    **scatter_kwargs,
                )
                legend_proxies.append(
                    _make_legend_proxy(
                        fill_map[label1],
                        outline_map[label2],
                        f"Fill {label1} · Edge {label2}",
                    )
                )

        # Separate outline-only legend
        outline_proxies = [
            _make_legend_proxy("lightgray", outline_map[lbl], f"Edge {lbl}")
            for lbl in unique_y2
        ]
        ax.legend(
            handles=outline_proxies,
            loc="upper right",
            fontsize=9,
            title="Edge label",
            framealpha=0.9,
        )

    # Axes labels
    ax.set_xlabel("D1", fontsize=11)
    ax.set_ylabel("D2", fontsize=11)
    if is_3d:
        ax.set_zlabel("D3", fontsize=11)

    # Main legend
    ncol = max(1, len(legend_proxies) // 12)
    ax.legend(
        handles=legend_proxies,
        loc="upper left",
        fontsize=9,
        title="Fill label" if y_2 is None else "Fill · Edge",
        framealpha=0.9,
        ncol=ncol,
    )

    dims = "3D" if is_3d else "2D"
    ax.set_title(f"{dims} Sample Plot", fontsize=14, pad=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
