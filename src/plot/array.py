from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def confusion_matrix_to_plot(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    class_names: list[str] | None = None,
    normalize: str | None = "true",  # one of: None, 'true', 'pred', 'all'
    cmap: str = "Blues",
    figsize: tuple[float, float] = (8, 6),
    show_colorbar: bool = True,
    values_decimals: int | None = None,  # override decimals used in annotations
) -> plt.Figure:
    """Plot a confusion matrix with optional normalization."""
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("`cm` must be a square 2D array (n_classes x n_classes).")

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    elif len(class_names) != n_classes:
        raise ValueError("`class_names` length must match cm.shape[0].")

    cm = np.asarray(cm, dtype=float)

    # Compute normalized matrix if requested (with safe division)
    if normalize is None:
        cm_to_plot = cm.copy()
    elif normalize == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_to_plot = np.divide(cm, np.where(row_sums == 0, 1, row_sums))
    elif normalize == "pred":
        col_sums = cm.sum(axis=0, keepdims=True)
        cm_to_plot = np.divide(cm, np.where(col_sums == 0, 1, col_sums))
    elif normalize == "all":
        total = cm.sum()
        cm_to_plot = cm / (total if total != 0 else 1.0)
    else:
        raise ValueError("`normalize` must be one of {None, 'true', 'pred', 'all'}.")

    # Decide annotation format
    if values_decimals is None:
        values_decimals = 0 if normalize is None else 2
    values_format = f".{values_decimals}f"

    fig, ax = plt.subplots(figsize=figsize)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_to_plot, display_labels=class_names
    )
    disp.plot(
        ax=ax,
        cmap=cmap,
        values_format=values_format,
        colorbar=show_colorbar,
        im_kw={"interpolation": "nearest"},
        xticks_rotation=45,
        include_values=True,
    )

    ax.set_title(title, fontsize=14, pad=16)
    if normalize is not None:
        norm_map = {
            "true": "rows (by true label)",
            "pred": "columns (by predicted)",
            "all": "entire matrix",
        }
        ax.set_xlabel(
            f"Predicted label\nNormalization: {norm_map[normalize]}", labelpad=10
        )
    else:
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
    """Plot 2D samples with color coding and optional shape encoding.

    Args:
        X: Array of shape (n_samples, 2) for 2D plotting.
        y_1: Integer labels for fill color coding.
        y_2: Optional integer labels for marker shape encoding.

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

    # Unique labels
    unique_y1 = np.unique(y_1)

    # Use a colormap that supports many distinct colors
    if len(unique_y1) <= 10:
        cmap = plt.cm.get_cmap("tab10", len(unique_y1))
    elif len(unique_y1) <= 20:
        cmap = plt.cm.get_cmap("tab20", len(unique_y1))
    else:
        cmap = plt.cm.get_cmap("gist_ncar", len(unique_y1))

    # Define marker shapes for y_2
    marker_shapes = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "<",
        ">",
        "p",
        "*",
        "h",
        "H",
        "X",
        "d",
        "P",
    ]

    # Keep track of legend entries to avoid duplicates
    legend_handles = []
    legend_labels = []

    if y_2 is None:
        # Simple case: only color coding
        for idx, label in enumerate(unique_y1):
            mask = y_1 == label
            scatter = ax.scatter(
                X[mask, 0],
                X[mask, 1],
                c=[cmap(idx)],
                label=f"Class {label}",
                edgecolors="black",
                s=80,
                alpha=0.7,
                linewidths=1.5,
            )
            legend_handles.append(scatter)
            legend_labels.append(f"Class {label}")
    else:
        # Complex case: color for y_1, shape for y_2
        unique_y2 = np.unique(y_2)

        if len(unique_y2) > len(marker_shapes):
            raise ValueError(
                f"Too many unique values in y_2 ({len(unique_y2)}). Maximum supported: {len(marker_shapes)}"
            )

        # Create a mapping for shapes
        shape_map = {
            label: marker_shapes[i % len(marker_shapes)]
            for i, label in enumerate(unique_y2)
        }

        # Plot all combinations
        plotted_combinations = set()

        for idx, color_label in enumerate(unique_y1):
            for jdx, shape_label in enumerate(unique_y2):
                combined_mask = (y_1 == color_label) & (y_2 == shape_label)

                if np.any(combined_mask):
                    scatter = ax.scatter(
                        X[combined_mask, 0],
                        X[combined_mask, 1],
                        c=[cmap(idx)],
                        marker=shape_map[shape_label],
                        edgecolors="black",
                        s=100,
                        alpha=0.7,
                        linewidths=1.5,
                    )

                    # Add to legend only if not already added
                    combo_key = (color_label, shape_label)
                    if combo_key not in plotted_combinations:
                        plotted_combinations.add(combo_key)
                        legend_handles.append(scatter)
                        legend_labels.append(f"C{color_label}·S{shape_label}")

        # Add separate legend for shapes if y_2 is used
        # Create dummy scatter plots for shape legend
        shape_legend_elements = []
        for shape_label in unique_y2:
            shape_legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=shape_map[shape_label],
                    color="gray",
                    linestyle="",
                    markersize=8,
                    label=f"Shape {shape_label}",
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                )
            )

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
        # Create two legends: one for color+shape combinations, one for shape reference
        leg1 = ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper left",
            framealpha=0.9,
            fontsize=9,
            title="Color·Shape",
            ncol=max(1, len(legend_labels) // 10),
        )
        ax.add_artist(leg1)

        ax.legend(
            handles=shape_legend_elements,
            loc="upper right",
            framealpha=0.9,
            fontsize=9,
            title="Shape Reference",
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
