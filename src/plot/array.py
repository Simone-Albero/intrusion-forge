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


def vectors_plot(
    vectors: np.ndarray,
    colors: list | np.ndarray,
    outline_colors: list | np.ndarray | None = None,
    use_shapes: bool = True,
) -> plt.Figure:
    """Plot 2D vectors with color coding for fill and optional shape/edge encoding.

    Args:
        vectors (np.ndarray): Array of shape (n_samples, 2) for 2D plotting.
        colors (list | np.ndarray): Integer labels for fill color coding.
        outline_colors (list | np.ndarray | None): Optional integer labels for shape or edge encoding.
            If None, all points use circles with black edges.
        use_shapes (bool): If True and outline_colors provided, uses different marker shapes.
            If False, uses edge colors instead. Shapes provide better visual distinction.
    """
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("`vectors` must be a 2D array with shape (n_samples, 2).")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Process fill colors - use colorblind-friendly colormap
    colors_array = np.asarray(colors)
    unique_labels = np.unique(colors_array)
    n_classes = len(unique_labels)

    # Use colorblind-friendly colormaps
    if n_classes <= 10:
        cmap = plt.cm.tab10
    elif n_classes <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.rainbow  # For many classes

    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    color_indices = np.array([label_to_idx[c] for c in colors_array])
    color_values = color_indices / max(n_classes - 1, 1)

    # Define distinct marker shapes
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
        "H",
        "X",
        "P",
        "d",
        "8",
    ]

    # Process outline colors
    if outline_colors is not None:
        outline_array = np.asarray(outline_colors)
        unique_outline = np.unique(outline_array)
        n_outline = len(unique_outline)

        if use_shapes and n_outline <= len(marker_shapes):
            # Use different shapes for better distinction
            outline_to_shape = {
                label: marker_shapes[idx % len(marker_shapes)]
                for idx, label in enumerate(unique_outline)
            }

            # Plot each shape group separately
            for outline_label in unique_outline:
                mask = outline_array == outline_label
                shape = outline_to_shape[outline_label]

                ax.scatter(
                    vectors[mask, 0],
                    vectors[mask, 1],
                    c=color_values[mask],
                    cmap=cmap,
                    s=100,
                    alpha=0.8,
                    marker=shape,
                    edgecolors="black",
                    linewidths=1.0,
                    vmin=0,
                    vmax=1,
                    label=f"Shape: {int(outline_label)}",
                )
        else:
            # Fall back to edge colors
            if n_outline <= 8:
                outline_cmap = plt.cm.Dark2
            elif n_outline <= 12:
                outline_cmap = plt.cm.Paired
            else:
                outline_cmap = plt.cm.tab20

            outline_to_idx = {label: idx for idx, label in enumerate(unique_outline)}
            outline_indices = np.array([outline_to_idx[c] for c in outline_array])
            outline_values = outline_indices / max(n_outline - 1, 1)
            edge_colors = outline_cmap(outline_values)

            ax.scatter(
                vectors[:, 0],
                vectors[:, 1],
                c=color_values,
                cmap=cmap,
                s=80,
                alpha=0.8,
                edgecolors=edge_colors,
                linewidths=2.0,
                vmin=0,
                vmax=1,
            )
    else:
        # Simple plot without outline encoding
        ax.scatter(
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

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title("2D Vector Plot", fontsize=14, pad=16)

    # Add legend for fill colors
    if n_classes <= 20:
        color_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(i / max(n_classes - 1, 1)),
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=f"Color: {int(label)}",
            )
            for i, label in enumerate(unique_labels)
        ]

        # Add legend for shapes or edge colors if provided
        if outline_colors is not None:
            if use_shapes and n_outline <= len(marker_shapes):
                # Shape legend
                shape_handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker=marker_shapes[i % len(marker_shapes)],
                        color="w",
                        markerfacecolor="gray",
                        markersize=8,
                        markeredgecolor="black",
                        markeredgewidth=1.0,
                        label=f"Shape: {int(label)}",
                    )
                    for i, label in enumerate(unique_outline)
                ]

                # Combine both legends
                all_handles = (
                    color_handles
                    + [plt.Line2D([0], [0], visible=False)]
                    + shape_handles
                )
                ncol = 2 if len(all_handles) > 12 else 1
                ax.legend(
                    handles=all_handles,
                    loc="best",
                    framealpha=0.9,
                    ncol=ncol,
                    fontsize=9,
                )
            elif n_outline <= 20:
                # Edge color legend
                edge_handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="white",
                        markersize=8,
                        markeredgecolor=outline_cmap(i / max(n_outline - 1, 1)),
                        markeredgewidth=2.0,
                        label=f"Edge: {int(label)}",
                    )
                    for i, label in enumerate(unique_outline)
                ]
                all_handles = color_handles + edge_handles
                ncol = 2 if len(all_handles) > 10 else 1
                ax.legend(
                    handles=all_handles,
                    loc="best",
                    framealpha=0.9,
                    ncol=ncol,
                    fontsize=9,
                )
            else:
                ax.legend(handles=color_handles, loc="best", framealpha=0.9, fontsize=9)
        else:
            ncol = 2 if n_classes > 10 else 1
            ax.legend(
                handles=color_handles, loc="best", framealpha=0.9, ncol=ncol, fontsize=9
            )
    else:
        # For many classes, use a colorbar
        scatter = ax.collections[0] if ax.collections else None
        if scatter:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label("Class", rotation=270, labelpad=20)

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
