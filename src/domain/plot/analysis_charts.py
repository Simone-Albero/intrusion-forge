import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .base import Plot, _apply_labels, _ensure_ax, _fig_to_plot, _finalize
from .primitives import bar_plot, numeric_scatter_plot
from .style import MUTED_COLOR


def _strip_plot(
    categories: np.ndarray,
    values: np.ndarray,
    *,
    fill_values: np.ndarray | None = None,
    fill_cmap: str,
    category_order: list | None = None,
    show_median: bool = True,
    x_label: str = "",
    y_label: str = "",
    ax: Axes | None = None,
) -> Plot | None:
    """Horizontal strip plot with a colormapped fill, one row per category."""
    categories = np.asarray(categories)
    values = np.asarray(values, dtype=float)

    if category_order is None:
        category_order = list(dict.fromkeys(categories.tolist()))
    n_cats = len(category_order)
    cat_to_pos = {cat: i for i, cat in enumerate(category_order)}

    fill_arr = (
        np.asarray(fill_values, dtype=float) if fill_values is not None else values
    )
    finite = np.isfinite(fill_arr)
    cmap_fn = plt.get_cmap(fill_cmap)
    lo = float(np.nanmin(fill_arr)) if finite.any() else 0.0
    hi = float(np.nanmax(fill_arr)) if finite.any() else 1.0
    span = hi - lo if hi > lo else 1.0
    normed = np.where(finite, (fill_arr - lo) / span, 0.5)
    point_colors = np.array([cmap_fn(float(v)) for v in normed], dtype=float)
    point_colors[~finite] = [0.75, 0.75, 0.75, 0.85]

    rng = np.random.default_rng(seed=42)
    base_positions = np.array([cat_to_pos[c] for c in categories], dtype=float)
    positions = base_positions + rng.uniform(-0.25, 0.25, size=len(categories))

    figsize = (11, max(6.0, 0.35 * n_cats + 2.0))
    ax, fig = _ensure_ax(ax, figsize)

    ax.scatter(
        values,
        positions,
        c=point_colors,
        s=36.0,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
        alpha=0.85,
    )

    if show_median:
        for cat in category_order:
            mask = categories == cat
            if not mask.any():
                continue
            pos = cat_to_pos[cat]
            med = float(np.median(values[mask]))
            ax.plot(
                (med, med),
                (pos - 0.3, pos + 0.3),
                color=MUTED_COLOR,
                linewidth=1.5,
                zorder=4,
            )

    cat_labels = [str(c) for c in category_order]
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(cat_labels)
    ax.invert_yaxis()
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    _apply_labels(ax, x_label, y_label)
    return _finalize(fig)


def strip_count_panel_plot(
    categories: np.ndarray,
    values: np.ndarray,
    category_order: list[str],
    counts_by_class: dict[str, int],
    fill_values: np.ndarray,
    fill_cmap: str,
    x_label: str,
    *,
    fill_cmap_label: str = "",
) -> Plot:
    """Strip plot and horizontal count bar side by side, sharing the y-axis."""
    n_cats = len(category_order)
    height = max(3.0, 0.35 * n_cats + 1.5)
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [3.5, 1.0], "wspace": 0.04},
        figsize=(11, height),
        sharey=True,
    )

    _strip_plot(
        categories=categories,
        values=values,
        fill_values=fill_values,
        fill_cmap=fill_cmap,
        category_order=category_order,
        show_median=True,
        x_label=x_label,
        y_label="Class",
        ax=ax_left,
    )

    fill_arr = np.asarray(fill_values, dtype=float)
    finite = np.isfinite(fill_arr)
    lo = float(np.nanmin(fill_arr[finite])) if finite.any() else 0.0
    hi = float(np.nanmax(fill_arr[finite])) if finite.any() else 1.0
    norm = mcolors.Normalize(vmin=lo, vmax=hi)
    sm = plt.cm.ScalarMappable(cmap=fill_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_left, fraction=0.025, pad=0.02, shrink=0.8)
    cbar.set_label(fill_cmap_label, fontsize=8)

    counts = [int(counts_by_class.get(cat, 0)) for cat in category_order]
    y_positions = np.arange(n_cats)
    max_count = max(counts) if counts else 1

    bar_plot(
        labels=list(category_order),
        values=counts,
        orientation="h",
        sort=None,
        bar_positions=y_positions,
        bar_alpha=0.55,
        color=MUTED_COLOR,
        annotate_values=True,
        value_format="{:.0f}",
        x_label="n clusters",
        hide_yticks=True,
        hide_left_spine=True,
        xlim=(0, max_count * 1.18),
        ax=ax_right,
    )

    return _fig_to_plot(fig)


def dual_scatter_plot(
    x: np.ndarray,
    panels: list[tuple[str, np.ndarray, np.ndarray, dict[str, float]]],
    *,
    cmap: str = "viridis",
    colorbar_label: str = "",
    reference_line: bool = False,
    x_label: str = "",
    y_label: str = "",
    figsize: tuple[float, float] = (9.6, 4.2),
) -> Plot | None:
    """Numeric-scatter panels sharing axes and one colorbar, one per `(title, y, color, annotations)`."""
    if not panels:
        return None
    color_all = np.concatenate([np.asarray(c, dtype=float) for _, _, c, _ in panels])
    finite = np.isfinite(color_all)
    vmax = float(np.quantile(color_all[finite], 0.95)) if finite.any() else None

    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for i, (ax, (title, y, color, annotations)) in enumerate(zip(axes, panels)):
        numeric_scatter_plot(
            x,
            y,
            color_values=color,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            reference_line=reference_line,
            annotations=annotations,
            x_label=x_label,
            y_label=y_label if i == 0 else "",
            title=title,
            ax=ax,
        )

    norm = mcolors.Normalize(vmin=0.0, vmax=vmax if vmax is not None else 1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes), fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)

    return _fig_to_plot(fig)
