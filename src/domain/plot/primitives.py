import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .base import (
    Plot,
    _apply_labels,
    _ensure_ax,
    _fig_to_plot,
    _finalize,
    _format_value,
    _smart_legend_loc,
)
from .style import (
    HIGHLIGHT_COLOR,
    MUTED_COLOR,
    NEUTRAL_COLOR,
    PALETTE,
    extended_palette,
)


def bar_plot(
    labels: np.ndarray | list[str],
    values: np.ndarray | list[float],
    *,
    orientation: str = "h",
    color: str | list[str] | None = None,
    sort: str | None = "asc",
    top_k: int | None = None,
    annotate_values: bool = True,
    value_format: str = "{:.3f}",
    color_gradient: bool = False,
    x_label: str = "",
    y_label: str = "",
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    bar_positions: np.ndarray | None = None,
    bar_alpha: float = 1.0,
    hide_yticks: bool = False,
    hide_left_spine: bool = False,
) -> Plot | None:
    """Bar chart with optional sorting, top-k filtering and value annotations."""
    if orientation not in ("h", "v"):
        raise ValueError("`orientation` must be 'h' or 'v'.")
    if sort not in ("asc", "desc", None):
        raise ValueError("`sort` must be 'asc', 'desc', or None.")

    items = list(zip(list(labels), list(values)))
    if sort == "asc":
        items.sort(key=lambda kv: kv[1])
        items = items[-top_k:] if top_k is not None else items
    elif sort == "desc":
        items.sort(key=lambda kv: kv[1], reverse=True)
        items = items[:top_k] if top_k is not None else items
    elif top_k is not None:
        items = items[:top_k]

    plot_labels = [str(name) for name, _ in items]
    plot_values = np.array([v for _, v in items], dtype=float)
    n = len(plot_labels)

    if isinstance(color, list):
        bar_color = list(color)
    else:
        base = color if color is not None else PALETTE[0]
        if color_gradient and n > 0:
            base_rgb = np.array(mcolors.to_rgb(base))
            white = np.ones(3)
            ts = 0.25 + 0.75 * np.arange(n) / max(n - 1, 1)
            bar_color = [mcolors.to_hex(t * base_rgb + (1 - t) * white) for t in ts]
        else:
            bar_color = base

    if figsize is None:
        figsize = (
            (12, max(6, n * 0.4 + 1.5))
            if orientation == "h"
            else (max(8, n * 0.5 + 1.5), 6)
        )

    ax, fig = _ensure_ax(ax, figsize)

    positions = bar_positions if bar_positions is not None else plot_labels
    draw = ax.barh if orientation == "h" else ax.bar
    draw(
        positions,
        plot_values,
        color=bar_color,
        alpha=bar_alpha,
        edgecolor="white",
        linewidth=0.5,
    )

    if orientation == "h":
        ax.set_xlabel(x_label or "Value")
        if y_label:
            ax.set_ylabel(y_label)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
    else:
        if x_label:
            ax.set_xlabel(x_label)
        ax.set_ylabel(y_label or "Value")
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if annotate_values and n:
        value_range = plot_values.max() - plot_values.min() or 1.0
        offset = max(value_range * 0.01, 1e-6)
        for i, v in enumerate(plot_values):
            text = value_format.format(v)
            pos = float(bar_positions[i]) if bar_positions is not None else i
            if orientation == "h":
                ax.text(v + offset, pos, text, va="center", ha="left")
            else:
                ax.text(pos, v + offset, text, ha="center", va="bottom")

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if hide_yticks:
        ax.tick_params(axis="y", left=False, labelleft=False)
    if hide_left_spine:
        ax.spines["left"].set_visible(False)

    return _finalize(fig)


def violin_plot(
    categories: np.ndarray,
    values: np.ndarray,
    *,
    category_order: list | None = None,
    colors: tuple[str, ...] | None = None,
    violin_alpha: float = 0.55,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    figsize: tuple[float, float] = (5.4, 3.6),
) -> Plot | None:
    """Violin plot with an inner box (quartiles + median), one per category."""
    categories = np.asarray(categories)
    values = np.asarray(values)

    unique_cats = (
        list(category_order)
        if category_order is not None
        else list(dict.fromkeys(categories.tolist()))
    )
    n_cats = len(unique_cats)

    cat_colors = list(colors) if colors is not None else extended_palette(n_cats)
    if len(cat_colors) < n_cats:
        raise ValueError("`colors` must have at least one entry per category.")

    fig, ax = plt.subplots(figsize=figsize)

    for idx, (cat, color) in enumerate(zip(unique_cats, cat_colors)):
        mask = categories == cat
        vals = values[mask]
        if len(vals) < 2 or np.unique(vals).size < 2:
            continue

        parts = ax.violinplot(
            vals,
            positions=[idx],
            showmedians=False,
            showextrema=False,
            widths=0.85,
        )
        body = parts["bodies"][0]
        body.set_facecolor(color)
        body.set_alpha(violin_alpha)
        body.set_edgecolor(color)
        body.set_linewidth(1.0)

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        whisker_low = float(np.min(vals))
        whisker_high = float(np.max(vals))

        box_half = 0.05
        ax.add_patch(
            plt.Rectangle(
                (idx - box_half, q1),
                box_half * 2,
                q3 - q1,
                facecolor="white",
                edgecolor=MUTED_COLOR,
                linewidth=1.0,
                zorder=4,
            )
        )
        ax.plot(
            [idx - box_half, idx + box_half],
            [med, med],
            color=MUTED_COLOR,
            linewidth=1.8,
            zorder=5,
        )
        for y0, y1 in ((whisker_low, q1), (q3, whisker_high)):
            ax.plot(
                [idx, idx],
                [y0, y1],
                color=MUTED_COLOR,
                linewidth=0.8,
                zorder=4,
            )

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels([str(c) for c in unique_cats])

    _apply_labels(ax, x_label, y_label, title)
    return _fig_to_plot(fig)


def scatter_plot(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    highlight_mask: np.ndarray | None = None,
    names: dict | None = None,
    marker_size: float = 14.0,
    marker_alpha: float | None = None,
    minority_fraction: float = 0.05,
    x_label: str = "Dim 1",
    y_label: str = "Dim 2",
    title: str = "",
    show_legend: bool = True,
    legend_max_items: int = 20,
    legend_on_top: bool = False,
    figsize: tuple[float, float] = (5.6, 4.3),
) -> Plot:
    """2D scatter coloured by label, with an optional highlight mask."""
    X = np.asarray(X)
    labels = np.asarray(labels)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("`X` must have shape (n, 2).")

    highlight = (
        np.asarray(highlight_mask, dtype=bool)
        if highlight_mask is not None
        else np.zeros(len(X), dtype=bool)
    )

    unique_labels = sorted(int(l) for l in np.unique(labels))
    n_labels = max(len(unique_labels), 1)
    color_list = extended_palette(n_labels)
    color_map = {lbl: color_list[i] for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="datalim")

    counts = {lbl: int((labels == lbl).sum()) for lbl in unique_labels}
    total_visible = max(sum(counts.values()), 1)
    draw_order = sorted(unique_labels, key=lambda l: counts[l], reverse=True)

    for lbl in draw_order:
        base = labels == lbl
        if not base.any():
            continue
        color = color_map[lbl]
        n_class = counts[lbl]
        is_minority = n_class / total_visible < minority_fraction

        alpha_val = (
            float(np.clip(2.0 / np.sqrt(max(n_class, 1)), 0.25, 0.85))
            if marker_alpha is None
            else float(marker_alpha)
        )
        size_val = marker_size * (1.4 if is_minority else 1.0)
        if is_minority:
            alpha_val = min(alpha_val + 0.15, 0.85)

        normal = base & ~highlight
        hot = base & highlight
        if normal.any():
            ax.scatter(
                X[normal, 0],
                X[normal, 1],
                c=[color],
                s=size_val,
                alpha=alpha_val,
                marker="o",
                edgecolors="none",
                zorder=3 if is_minority else 2,
                rasterized=True,
            )
        if hot.any():
            ax.scatter(
                X[hot, 0],
                X[hot, 1],
                c=[color],
                s=size_val,
                alpha=min(alpha_val + 0.25, 0.95),
                marker="o",
                edgecolors=HIGHLIGHT_COLOR,
                linewidths=0.9,
                zorder=10,
                rasterized=True,
            )

    if show_legend and names is not None:
        if len(unique_labels) > legend_max_items:
            ax.text(
                0.99,
                0.02,
                f"{len(unique_labels)} groups",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color=MUTED_COLOR,
            )
        else:
            handles = [
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=8,
                    markerfacecolor=color_map[lbl],
                    markeredgecolor="#444444",
                    markeredgewidth=0.5,
                    label=names.get(int(lbl), str(lbl)),
                )
                for lbl in unique_labels
            ]
            if highlight_mask is not None:
                handles.append(
                    plt.Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="",
                        markersize=8,
                        markerfacecolor=NEUTRAL_COLOR,
                        markeredgecolor=HIGHLIGHT_COLOR,
                        markeredgewidth=0.9,
                        label="misclassified",
                    )
                )
            ncol = 1 if len(handles) <= 6 else 2 if len(handles) <= 12 else 3
            loc = _smart_legend_loc(ax, X) if len(X) else "best"
            legend = ax.legend(
                handles=handles,
                loc=loc,
                ncol=ncol,
                framealpha=1.0 if legend_on_top else None,
            )
            if legend_on_top:
                legend.set_zorder(100)

    _apply_labels(ax, x_label, y_label, title)
    return _fig_to_plot(fig)


def numeric_scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    *,
    color_values: np.ndarray | None = None,
    reference_line: bool = False,
    trend_line: bool = False,
    annotations: dict[str, float] | None = None,
    cmap: str = "viridis",
    colorbar_label: str = "",
    alpha: float = 0.7,
    vmin: float | None = None,
    vmax: float | None = None,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    figsize: tuple[float, float] = (5.6, 4.2),
    ax: Axes | None = None,
) -> Plot | None:
    """Numeric scatter with optional colouring, reference line, trend line and metrics box."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax, fig = _ensure_ax(ax, figsize)

    if reference_line and len(x):
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], color=NEUTRAL_COLOR, linewidth=1.0, linestyle=":")

    sc = ax.scatter(
        x,
        y,
        c=color_values if color_values is not None else PALETTE[0],
        cmap=cmap if color_values is not None else None,
        s=22,
        alpha=alpha,
        edgecolors="none",
        vmin=vmin,
        vmax=vmax,
    )
    if color_values is not None and fig is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label)

    if trend_line:
        finite = np.isfinite(x) & np.isfinite(y)
        if int(finite.sum()) >= 2:
            m, b = np.polyfit(x[finite], y[finite], 1)
            x_range = np.array([float(x[finite].min()), float(x[finite].max())])
            ax.plot(
                x_range,
                m * x_range + b,
                color=NEUTRAL_COLOR,
                linewidth=1.0,
                linestyle="--",
                zorder=2,
            )

    if annotations:
        text = "\n".join(
            f"{name} = {_format_value(value, kind='score')}"
            for name, value in annotations.items()
        )
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
        )

    ax.grid(True, alpha=0.15, linewidth=0.5)
    _apply_labels(ax, x_label, y_label, title)
    return _finalize(fig)


def line_plot(
    series: dict[str, list[float] | np.ndarray],
    *,
    x_label: str = "step",
    y_label: str = "value",
    title: str = "",
    colors: list[str] | None = None,
    linewidth: float = 1.2,
    show_legend: bool = True,
    figsize: tuple[float, float] = (8.0, 4.0),
) -> Plot:
    """Multi-series line plot (e.g. training history curves) sharing a step-index x axis."""
    fig, ax = plt.subplots(figsize=figsize)
    keys = list(series.keys())
    palette = colors if colors is not None else extended_palette(max(len(keys), 1))

    for i, (name, values) in enumerate(series.items()):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        ax.plot(
            np.arange(arr.size), arr, color=palette[i], linewidth=linewidth, label=name
        )

    _apply_labels(ax, x_label=x_label, y_label=y_label, title=title)
    ax.grid(True, axis="both", alpha=0.15, linewidth=0.5)
    if show_legend and len(keys) > 1:
        ax.legend(loc="best", fontsize=8)

    return _fig_to_plot(fig)
