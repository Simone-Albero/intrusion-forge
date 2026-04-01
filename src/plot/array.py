import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import seaborn as sns
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
    y_2: np.ndarray | None,
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
    y_1: list | np.ndarray,
    y_2: list | np.ndarray | None = None,
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


def strip_box_plot(
    categories: np.ndarray,
    values: np.ndarray,
    color_values: np.ndarray | None = None,
    edge_values: np.ndarray | None = None,
    x_label: str = "",
    y_label: str = "",
    c_label: str = "",
    edge_label: str = "",
    edge_value_labels: dict | None = None,
    title: str = "",
    figsize: tuple[float, float] = (12, 6),
    marker_size: float = 36,
    cmap: str = "RdYlGn_r",
) -> plt.Figure:
    """Strip plot with per-point color encoding and optional discrete edge encoding.

    Points are jittered horizontally per category; y-axis shows *values*.
    Fill color maps *color_values* (or *values* if omitted) through *cmap*.
    NaN fill values render as neutral gray.
    If *edge_values* is given, discrete unique values map to high-contrast edge
    colors; NaN edge values receive a muted gray outline.
    A horizontal bar marks the per-category median.
    """
    categories = np.asarray(categories)
    values = np.asarray(values, dtype=float)
    c = (
        np.asarray(color_values, dtype=float)
        if color_values is not None
        else values.copy()
    )

    # --- fill colors: normalize over finite values, NaN → gray ---
    finite = np.isfinite(c)
    c_min = float(c[finite].min()) if finite.any() else 0.0
    c_max = float(c[finite].max()) if finite.any() else 1.0
    norm = mcolors.Normalize(vmin=c_min, vmax=c_max)
    colormap = plt.get_cmap(cmap)
    c_safe = np.where(finite, c, (c_min + c_max) / 2)
    point_colors = colormap(norm(c_safe))
    point_colors[~finite] = [0.75, 0.75, 0.75, 0.85]

    # --- edge colors: discrete mapping, NaN → gray ---
    mapped_edge_colors: list | str = "white"
    unique_edges: list = []
    edge_color_map: dict = {}
    scatter_lw = 0.4
    if edge_values is not None:
        edge_arr = np.asarray(edge_values)
        try:
            nan_edge = ~np.isfinite(edge_arr.astype(float))
        except (ValueError, TypeError):
            nan_edge = np.zeros(len(edge_arr), dtype=bool)
        unique_edges = sorted(
            dict.fromkeys(v for v, m in zip(edge_arr.tolist(), nan_edge) if not m)
        )
        edge_color_map = {
            v: _OUTLINE_COLORS[i % len(_OUTLINE_COLORS)]
            for i, v in enumerate(unique_edges)
        }
        mapped_edge_colors = [
            "#bbbbbb" if m else edge_color_map[v]
            for v, m in zip(edge_arr.tolist(), nan_edge)
        ]
        scatter_lw = 1.3

    # --- layout ---
    category_order = list(dict.fromkeys(categories))
    cat_to_x = {cat: i for i, cat in enumerate(category_order)}
    rng = np.random.default_rng(seed=42)
    x_positions = np.array(
        [cat_to_x[cat] + rng.uniform(-0.25, 0.25) for cat in categories]
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        x_positions,
        values,
        c=point_colors,
        s=marker_size,
        edgecolors=mapped_edge_colors,
        linewidths=scatter_lw,
        zorder=3,
        alpha=0.85,
    )

    for cat, x in cat_to_x.items():
        median_val = np.median(values[categories == cat])
        ax.plot(
            [x - 0.3, x + 0.3],
            [median_val, median_val],
            color="#444444",
            linewidth=1.5,
            zorder=4,
        )

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03).set_label(c_label, fontsize=10)

    if unique_edges:
        _fmt = lambda v: (
            edge_value_labels[v]
            if edge_value_labels and v in edge_value_labels
            else (str(int(v)) if isinstance(v, float) and v == int(v) else str(v))
        )
        edge_handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor="#dddddd",
                markeredgecolor=edge_color_map[v],
                markeredgewidth=1.5,
                label=_fmt(v),
            )
            for v in unique_edges
        ]
        ax.add_artist(
            ax.legend(
                handles=edge_handles,
                title=edge_label or "edge",
                loc="upper left",
                fontsize=8,
                title_fontsize=9,
                framealpha=0.9,
            )
        )

    ax.set_xticks(range(len(category_order)))
    ax.set_xticklabels(category_order)
    ax.set_title(title, fontsize=14, pad=16)
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def violin_box_plot(
    categories: np.ndarray,
    values: np.ndarray,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    figsize: tuple[float, float] = (6, 5),
    violin_alpha: float = 0.4,
    palette: str = "Set2",
    category_order: list | None = None,
) -> plt.Figure:
    """Split violin plot: each category occupies one half, with a color legend.

    The first category in order of appearance is drawn on the left half,
    the second on the right. Inner quartile lines mark Q1, median, and Q3.
    """
    categories = np.asarray(categories)
    values = np.asarray(values)

    unique_cats = (
        category_order
        if category_order is not None
        else list(dict.fromkeys(categories))
    )
    palette_colors = sns.color_palette(palette, n_colors=len(unique_cats))

    fig, ax = plt.subplots(figsize=figsize)

    for side, (cat, color) in enumerate(zip(unique_cats, palette_colors)):
        mask = categories == cat
        vals = values[mask]
        if len(vals) < 2:
            continue

        parts = ax.violinplot(
            vals, positions=[0], showmedians=False, showextrema=False, widths=0.8
        )
        body = parts["bodies"][0]
        body.set_facecolor(color)
        body.set_alpha(violin_alpha)
        body.set_edgecolor(color)
        body.set_linewidth(1.0)

        # clip to left (side=0) or right (side=1) half
        verts = body.get_paths()[0].vertices
        if side == 0:
            verts[:, 0] = np.minimum(verts[:, 0], 0.0)
        else:
            verts[:, 0] = np.maximum(verts[:, 0], 0.0)

        # inner quartile indicator
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        sign = -1 if side == 0 else 1
        ax.plot([0, sign * 0.06], [med, med], color=color, lw=2.0, zorder=4)
        ax.vlines(sign * 0.04, q1, q3, color=color, lw=1.5, zorder=4)

    ax.axvline(0, color="#aaaaaa", lw=0.8, zorder=2)

    legend_handles = [
        plt.Line2D([], [], color=c, lw=6, alpha=violin_alpha, label=cat)
        for cat, c in zip(unique_cats, palette_colors)
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=9, framealpha=0.9)

    ax.set_xticks([])
    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout()
    return fig


def roc_curve_plot(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str = "ROC Curve",
    figsize: tuple[float, float] = (6, 6),
) -> plt.Figure:
    """ROC curve with AUC annotation and random-classifier baseline."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color="#4E79A7", linewidth=2, label=f"AUC = {auc_score:.4f}")
    ax.plot(
        [0, 1], [0, 1], color="#AAAAAA", linewidth=1, linestyle="--", label="Random"
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate", labelpad=10)
    ax.set_ylabel("True Positive Rate", labelpad=10)
    ax.set_title(title, fontsize=14, pad=16)
    ax.legend(loc="lower right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def feature_importance_plot(
    importances: dict[str, float],
    title: str = "Feature Importances",
    figsize: tuple[float, float] = (12, 8),
) -> plt.Figure:
    """Horizontal bar chart of feature importances, sorted descending."""
    sorted_items = sorted(importances.items(), key=lambda x: x[1])
    features = [item[0] for item in sorted_items]
    values = np.array([item[1] for item in sorted_items])

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(features, values, color="#4E79A7", edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=14, pad=16)
    ax.set_xlabel("Importance", labelpad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout(pad=1.0)
    return fig


def grouped_bar_plot(
    labels: list[str],
    groups: dict[str, list[float]],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    figsize: tuple[float, float] | None = None,
    colors: list[str] | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing multiple series across the same set of labels.

    Args:
        labels: Category labels for the x-axis.
        groups: ``{series_name: [value, ...]}`` — all lists must have the same
            length as *labels*.
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        figsize: Figure size. Defaults to ``(max(8, n * 0.9), 5)``.
        colors: One color per series. Defaults to ``_FILL_COLORS``.
    """
    n = len(labels)
    series = list(groups.items())
    n_series = len(series)
    width = 0.8 / n_series
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width
    palette = colors or _FILL_COLORS

    if figsize is None:
        figsize = (max(8, n * 0.9), 5)

    x = np.arange(n)
    fig, ax = plt.subplots(figsize=figsize)

    for (name, values), offset, color in zip(series, offsets, palette):
        ax.bar(x + offset, values, width, label=name, color=color, alpha=0.85)

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig
