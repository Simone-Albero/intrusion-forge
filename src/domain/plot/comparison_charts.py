import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from .base import Plot, _apply_labels, _ensure_ax, _finalize
from .style import MUTED_COLOR


def box_strip_plot(
    labels: list[str],
    values: list[np.ndarray],
    *,
    colors: list[str],
    faded: list[bool] | None = None,
    show_points: bool = True,
    x_label: str = "",
    x_lim: tuple[float, float] | None = None,
    axvline: float | None = None,
    legend: dict[str, str] | None = None,
    figsize: tuple[float, float] = (5.4, 3.0),
    ax: Axes | None = None,
) -> Plot | None:
    """Horizontal box plots with an optional jittered strip overlay, one row per group."""
    faded = faded or [False] * len(labels)
    pos = list(range(len(labels), 0, -1))
    ax, fig = _ensure_ax(ax, figsize)
    bp = ax.boxplot(
        values,
        positions=pos,
        vert=False,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        zorder=2,
        medianprops=dict(color="black", linewidth=1.3),
        whiskerprops=dict(color=MUTED_COLOR),
        capprops=dict(color=MUTED_COLOR),
    )
    for patch, color, fade in zip(bp["boxes"], colors, faded):
        patch.set_facecolor(color)
        patch.set_alpha(0.35 if fade else 0.65)
        patch.set_edgecolor("0.3")
    if show_points:
        rng = np.random.default_rng(0)
        for vals, p, color in zip(values, pos, colors):
            vals = np.asarray(vals, dtype=float)
            y = p + (rng.random(vals.size) - 0.5) * 0.32
            ax.scatter(
                vals, y, s=7, color=color, alpha=0.55, edgecolor="none", zorder=3
            )
    if axvline is not None:
        ax.axvline(axvline, color=MUTED_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(pos, labels)
    if x_lim:
        ax.set_xlim(*x_lim)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    if legend:
        handles = [
            Patch(facecolor=c, alpha=0.65, edgecolor="0.3", label=lab)
            for lab, c in legend.items()
        ]
        ax.legend(handles=handles, loc="lower left")
    _apply_labels(ax, x_label=x_label)
    return _finalize(fig)


def line_whisker_plot(
    series: dict[str, tuple[np.ndarray, np.ndarray, str]],
    *,
    n_bins: int = 8,
    log_x: bool = False,
    x_label: str = "",
    y_label: str = "",
    y_lim: tuple[float, float] | None = None,
    vline: float | None = None,
    vline_label: str = "",
    hline: float | None = None,
    figsize: tuple[float, float] = (5.4, 3.2),
    ax: Axes | None = None,
) -> Plot | None:
    """Binned trend line with ±1 std whiskers, one connected line per `(x, y, colour)`."""
    ax, fig = _ensure_ax(ax, figsize)
    if log_x:
        ax.set_xscale("log")

    for name, (x, y, color) in series.items():
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y) & (x > 0 if log_x else True)
        x, y = x[finite], y[finite]
        if x.size == 0:
            continue
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            edges = np.array([lo, lo + 1.0])
        elif log_x:
            edges = np.geomspace(lo, hi, n_bins + 1)
        else:
            edges = np.linspace(lo, hi, n_bins + 1)
        idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
        centers, means, stds = [], [], []
        for b in range(len(edges) - 1):
            sel = idx == b
            if not sel.any():
                continue
            centers.append(
                float(
                    np.sqrt(edges[b] * edges[b + 1])
                    if log_x
                    else 0.5 * (edges[b] + edges[b + 1])
                )
            )
            means.append(float(y[sel].mean()))
            stds.append(float(y[sel].std()))
        if not centers:
            continue
        ax.errorbar(
            centers,
            means,
            yerr=stds,
            color=color,
            marker="o",
            ms=5,
            linewidth=1.6,
            elinewidth=1.0,
            capsize=3,
            label=name,
            zorder=3,
        )

    if y_lim:
        ax.set_ylim(*y_lim)
    if vline is not None:
        ax.axvline(vline, color=MUTED_COLOR, linewidth=0.9, linestyle="--", zorder=1)
        if vline_label:
            ax.text(
                vline,
                ax.get_ylim()[0],
                f" {vline_label}",
                color=MUTED_COLOR,
                ha="left",
                va="bottom",
            )
    if hline is not None:
        ax.axhline(hline, color=MUTED_COLOR, linewidth=0.8, linestyle=":", zorder=1)
    ax.grid(True, axis="both")
    ax.legend(loc="lower right")
    _apply_labels(ax, x_label=x_label, y_label=y_label)
    return _finalize(fig)


def stacked_bar_plot(
    labels: list[str],
    segments: list[tuple[str, list[float], str]],
    *,
    x_label: str = "",
    total_format: str = "{:.1f}",
    sort: str | None = "asc",
    figsize: tuple[float, float] = (5.4, 2.9),
    ax: Axes | None = None,
) -> Plot | None:
    """Horizontal stacked bars with the total annotated at the end of each row."""
    totals = [sum(seg[1][i] for seg in segments) for i in range(len(labels))]
    if sort == "asc":
        order = list(np.argsort(totals))
    elif sort == "desc":
        order = list(np.argsort(totals)[::-1])
    else:
        order = list(range(len(labels)))
    labels = [labels[i] for i in order]
    totals = [totals[i] for i in order]
    segments = [
        (name, [vals[i] for i in order], color) for name, vals, color in segments
    ]

    y = np.arange(len(labels))
    ax, fig = _ensure_ax(ax, figsize)
    left = np.zeros(len(labels))
    for name, vals, color in segments:
        vals = np.asarray(vals, dtype=float)
        ax.barh(
            y,
            vals,
            left=left,
            height=0.55,
            color=color,
            alpha=0.85,
            edgecolor="0.3",
            label=name,
        )
        left = left + vals
    span = max(totals) if totals else 1.0
    for yi, total in zip(y, totals):
        ax.text(total + span * 0.015, yi, total_format.format(total), va="center")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, span * 1.15)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(segments),
        columnspacing=1.2,
        frameon=False,
    )
    _apply_labels(ax, x_label=x_label)
    return _finalize(fig)


def grouped_bar_plot(
    group_labels: list[str],
    series: list[tuple[str, list[float], list[float], list[float], str]],
    *,
    x_label: str = "",
    y_label: str = "",
    y_lim: tuple[float, float] | None = None,
    hline: float | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Plot | None:
    """Vertical grouped bars with asymmetric error bars, one colour per series.

    Each `series` entry is `(name, values, err_low, err_high, color)`, one value per group.
    """
    n_groups = len(group_labels)
    n_series = len(series)
    if figsize is None:
        figsize = (max(8.0, n_groups * 0.9 + 1.5), 4.2)
    ax, fig = _ensure_ax(ax, figsize)

    width = 0.8 / max(n_series, 1)
    x = np.arange(n_groups, dtype=float)
    for i, (name, values, err_low, err_high, color) in enumerate(series):
        values = np.asarray(values, dtype=float)
        offset = (i - (n_series - 1) / 2) * width
        yerr = np.vstack(
            [np.asarray(err_low, dtype=float), np.asarray(err_high, dtype=float)]
        )
        ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=name,
            yerr=yerr,
            error_kw=dict(elinewidth=0.9, capsize=2.0, ecolor="0.25"),
        )

    if hline is not None:
        ax.axhline(hline, color=MUTED_COLOR, linewidth=0.8, linestyle=":", zorder=1)
    ax.set_xticks(x, group_labels)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(n_series, 5),
        columnspacing=1.2,
        frameon=False,
    )
    _apply_labels(ax, x_label=x_label, y_label=y_label)
    return _finalize(fig)
