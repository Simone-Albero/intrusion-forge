import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.domain.plot.base import Plot, _apply_labels, _fig_to_plot
from src.domain.plot.style import MUTED_COLOR


def box_plot(
    labels: list[str],
    values: list[np.ndarray],
    *,
    colors: list[str],
    x_label: str = "",
    x_lim: tuple[float, float] | None = None,
    axvline: float | None = None,
    legend: dict[str, str] | None = None,
    figsize: tuple[float, float] = (5.4, 3.0),
) -> Plot:
    """Horizontal box plots, one row per group."""
    positions = list(range(len(labels), 0, -1))
    fig, ax = plt.subplots(figsize=figsize)
    boxes = ax.boxplot(
        values,
        positions=positions,
        vert=False,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        zorder=2,
        medianprops=dict(color="black", linewidth=1.3),
        whiskerprops=dict(color=MUTED_COLOR),
        capprops=dict(color=MUTED_COLOR),
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor("0.3")
    if axvline is not None:
        ax.axvline(axvline, color=MUTED_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(positions, labels)
    if x_lim:
        ax.set_xlim(*x_lim)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    if legend:
        handles = [
            Patch(facecolor=color, alpha=0.65, edgecolor="0.3", label=name)
            for name, color in legend.items()
        ]
        ax.legend(handles=handles, loc="lower left")
    _apply_labels(ax, x_label=x_label)
    return _fig_to_plot(fig)


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
) -> Plot:
    """Binned trend line with ±1 std whiskers, one connected line per `(x, y, colour)`."""
    fig, ax = plt.subplots(figsize=figsize)
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
        bin_of = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
        centers, means, stds = [], [], []
        for b in range(len(edges) - 1):
            in_bin = bin_of == b
            if not in_bin.any():
                continue
            centers.append(
                float(
                    np.sqrt(edges[b] * edges[b + 1])
                    if log_x
                    else 0.5 * (edges[b] + edges[b + 1])
                )
            )
            means.append(float(y[in_bin].mean()))
            stds.append(float(y[in_bin].std()))
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
    return _fig_to_plot(fig)


def stacked_bar_plot(
    labels: list[str],
    segments: list[tuple[str, list[float], str]],
    *,
    x_label: str = "",
    total_format: str = "{:.1f}",
    sort: str | None = "asc",
    figsize: tuple[float, float] = (5.4, 2.9),
) -> Plot:
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
        (name, [values[i] for i in order], color) for name, values, color in segments
    ]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(len(labels))
    for name, values, color in segments:
        values = np.asarray(values, dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            height=0.55,
            color=color,
            alpha=0.85,
            edgecolor="0.3",
            label=name,
        )
        left = left + values
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
    return _fig_to_plot(fig)


def grouped_bar_plot(
    group_labels: list[str],
    series: list[tuple[str, list[float], list[float] | None, list[float] | None, str]],
    *,
    x_label: str = "",
    y_label: str = "",
    y_lim: tuple[float, float] | None = None,
    hline: float | None = None,
    log_y: bool = False,
    group_notes: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Plot:
    """Vertical grouped bars with asymmetric error bars, one colour per series."""
    n_groups = len(group_labels)
    n_series = len(series)
    if figsize is None:
        figsize = (max(8.0, n_groups * 0.9 + 1.5), 4.2)
    fig, ax = plt.subplots(figsize=figsize)

    width = 0.8 / max(n_series, 1)
    x = np.arange(n_groups, dtype=float)
    for i, (name, values, err_low, err_high, color) in enumerate(series):
        values = np.asarray(values, dtype=float)
        offset = (i - (n_series - 1) / 2) * width
        yerr = (
            None
            if err_low is None
            else np.vstack(
                [np.asarray(err_low, dtype=float), np.asarray(err_high, dtype=float)]
            )
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
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(x, group_labels)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    if group_notes:
        ax.margins(y=0.15)
    for xi, note in zip(x, group_notes or []):
        ax.text(
            xi,
            0.99,
            note,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize="small",
            color=MUTED_COLOR,
        )
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(n_series, 3),
        columnspacing=1.2,
        frameon=False,
    )
    _apply_labels(ax, x_label=x_label, y_label=y_label)
    return _fig_to_plot(fig)


def dual_axis_bar_plot(
    group_labels: list[str],
    left: tuple[str, list[float], str],
    right: tuple[str, list[float], str],
    *,
    x_label: str = "",
    left_lim: tuple[float, float] = (-0.05, 1.05),
    figsize: tuple[float, float] | None = None,
) -> Plot:
    """Two bars per group, `left` on the left y axis and `right` on its own right one.

    The axes are aligned at zero, so both bars of a group start from the same line.
    """
    n_groups = len(group_labels)
    if figsize is None:
        figsize = (max(5.0, n_groups * 1.1 + 1.5), 3.6)
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()

    width = 0.38
    x = np.arange(n_groups, dtype=float)
    handles = []
    for ax, (name, values, color), offset in (
        (ax_left, left, -width / 2),
        (ax_right, right, width / 2),
    ):
        ax.bar(
            x + offset,
            np.asarray(values, dtype=float),
            width=width * 0.92,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_ylabel(name, color=color)
        ax.tick_params(axis="y", colors=color)
        handles.append(Patch(facecolor=color, label=name))

    ax_left.axhline(0.0, color=MUTED_COLOR, linewidth=0.8, linestyle=":", zorder=1)
    ax_left.set_xticks(x, group_labels)
    plt.setp(ax_left.get_xticklabels(), rotation=35, ha="right")
    ax_right.spines["right"].set_visible(True)
    lo, hi = left_lim
    lo = min(lo, float(np.nanmin(np.asarray(left[1], dtype=float))))
    ax_left.set_ylim(lo, hi)
    top = ax_right.get_ylim()[1]
    ax_right.set_ylim(lo / hi * top, top)
    ax_left.grid(True, axis="y")
    ax_left.grid(False, axis="x")
    ax_right.grid(False)
    ax_left.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
    )
    _apply_labels(ax_left, x_label=x_label)
    return _fig_to_plot(fig)
