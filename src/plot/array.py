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


def vectors_plot(vectors: np.ndarray, colors: list | np.ndarray) -> plt.Figure:
    """Plot 2D or 3D vectors with color coding.

    Args:
        vectors (np.ndarray): Array of shape (n_samples, n_features) where n_features is 2 or 3.
        colors (list | np.ndarray): Array of integer labels for color coding.
    """
    if vectors.ndim != 2 or vectors.shape[1] not in (2, 3):
        raise ValueError(
            "`vectors` must be a 2D array with shape (n_samples, 2) or (n_samples, 3)."
        )

    n_features = vectors.shape[1]
    is_3d = n_features == 3

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d" if is_3d else None)

    # Convert colors to numpy array
    colors_array = np.asarray(colors)
    unique_labels = np.unique(colors_array)
    n_classes = len(unique_labels)

    # Choose appropriate colormap based on number of classes
    if n_classes <= 10:
        cmap = plt.cm.tab10
    elif n_classes <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar  # Good for many distinct colors

    # Map each label to a color index
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    color_indices = np.array([label_to_idx[c] for c in colors_array])

    # Normalize color indices to [0, 1] for colormap
    if n_classes > 1:
        color_values = color_indices / (n_classes - 1)
    else:
        color_values = color_indices * 0

    # Create scatter plot
    if is_3d:
        scatter = ax.scatter(
            vectors[:, 0],
            vectors[:, 1],
            vectors[:, 2],
            c=color_values,
            cmap=cmap,
            s=50,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            vmin=0,
            vmax=1,
        )
        ax.set_zlabel("Z", fontsize=12)
    else:
        scatter = ax.scatter(
            vectors[:, 0],
            vectors[:, 1],
            c=color_values,
            cmap=cmap,
            s=50,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            vmin=0,
            vmax=1,
        )

    # Set labels
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title(f"{n_features}D Vector Plot", fontsize=14, pad=16)

    # Add colorbar or legend based on number of classes
    if n_classes <= 20:
        # Create custom legend with actual class labels
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(i / (n_classes - 1) if n_classes > 1 else 0),
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=f"Class {int(label)}",
            )
            for i, label in enumerate(unique_labels)
        ]
        # Use multiple columns if more than 10 classes
        ncol = 2 if n_classes > 10 else 1
        ax.legend(handles=handles, loc="best", framealpha=0.9, ncol=ncol, fontsize=9)
    else:
        # Use colorbar for many classes
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label("Class", rotation=270, labelpad=20)

        # Set discrete ticks (show up to 15 labels)
        n_ticks = min(n_classes, 15)
        tick_positions = np.linspace(0, 1, n_ticks)
        cbar.set_ticks(tick_positions)
        tick_indices = np.linspace(0, n_classes - 1, n_ticks).astype(int)
        tick_labels = [f"{int(unique_labels[i])}" for i in tick_indices]
        cbar.set_ticklabels(tick_labels)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

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
