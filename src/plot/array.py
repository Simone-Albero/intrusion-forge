from typing import Union

import matplotlib.pyplot as plt
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


def samples_plot(
    X: np.ndarray,
    y_1: Union[list, np.ndarray],
    y_2: Union[list, np.ndarray, None] = None,
) -> plt.Figure:
    """Plot 2D samples with color coding and optional outline color encoding.

    Args:
        X: Array of shape (n_samples, 2) for 2D plotting.
        y_1: Integer labels for fill color coding.
        y_2: Optional integer labels for edge/outline color encoding.

    Returns:
        Matplotlib Figure object.
    """
    X = np.asarray(X)
    y_1 = np.asarray(y_1)

    if X.shape[1] != 2:
        raise ValueError("X must have shape (n_samples, 2).")

    if len(X) != len(y_1):
        raise ValueError("X and y_1 must have the same length.")

    if y_2 is not None:
        y_2 = np.asarray(y_2)
        if len(X) != len(y_2):
            raise ValueError("X and y_2 must have the same length.")

    fig, ax = plt.subplots(figsize=(10, 8))

    unique_y1 = np.unique(y_1)

    if len(unique_y1) <= 10:
        cmap = plt.cm.get_cmap("tab10", len(unique_y1))
    elif len(unique_y1) <= 20:
        cmap = plt.cm.get_cmap("tab20", len(unique_y1))
    else:
        cmap = plt.cm.get_cmap("gist_ncar", len(unique_y1))

    # High-contrast outline colors for y_2
    outline_colors = [
        "#FF0000",  # red
        "#000000",  # Black
        "#0000FF",  # blue
        "#FF00FF",  # magenta
        "#00FFFF",  # cyan
        "#FF8800",  # orange
        "#8800FF",  # purple
        "#00FF88",  # spring green
        "#FF0088",  # rose
        "#FFFF00",  # yellow
    ]

    legend_handles = []
    legend_labels = []

    if y_2 is None:
        for idx, label in enumerate(unique_y1):
            mask = y_1 == label
            scatter = ax.scatter(
                X[mask, 0],
                X[mask, 1],
                c=[cmap(idx)],
                label=f"Class {label}",
                edgecolors="black",
                s=150,
                alpha=0.7,
                linewidths=1.5,
            )
            legend_handles.append(scatter)
            legend_labels.append(f"Class {label}")
    else:
        unique_y2 = np.unique(y_2)

        if len(unique_y2) > len(outline_colors):
            raise ValueError(
                f"Too many unique values in y_2 ({len(unique_y2)}). Maximum supported: {len(outline_colors)}"
            )

        outline_map = {
            label: outline_colors[i % len(outline_colors)]
            for i, label in enumerate(unique_y2)
        }

        plotted_combinations = set()

        for idx, color_label in enumerate(unique_y1):
            for shape_label in unique_y2:
                combined_mask = (y_1 == color_label) & (y_2 == shape_label)

                if np.any(combined_mask):
                    scatter = ax.scatter(
                        X[combined_mask, 0],
                        X[combined_mask, 1],
                        c=[cmap(idx)],
                        marker="o",
                        edgecolors=outline_map[shape_label],
                        s=150,
                        alpha=0.7,
                        linewidths=2.0,
                    )

                    combo_key = (color_label, shape_label)
                    if combo_key not in plotted_combinations:
                        plotted_combinations.add(combo_key)
                        legend_handles.append(scatter)
                        legend_labels.append(f"C{color_label}·O{shape_label}")

        # Outline legend entries
        outline_legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                linestyle="",
                markersize=10,
                label=f"Outline {label}",
                markerfacecolor="gray",
                markeredgecolor=outline_map[label],
                markeredgewidth=2.5,
            )
            for label in unique_y2
        ]

    ax.set_xlabel("D1", fontsize=12)
    ax.set_ylabel("D2", fontsize=12)

    if y_2 is None:
        ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="best",
            framealpha=0.9,
            fontsize=10,
        )
    else:
        leg1 = ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper left",
            framealpha=0.9,
            fontsize=9,
            title="Fill·Outline",
            ncol=max(1, len(legend_labels) // 10),
        )
        ax.add_artist(leg1)

        ax.legend(
            handles=outline_legend_elements,
            loc="upper right",
            framealpha=0.9,
            fontsize=9,
            title="Outline Reference",
        )

    ax.set_title("2D Sample Plot", fontsize=14, pad=10)
    ax.grid(True, alpha=0.3)

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
