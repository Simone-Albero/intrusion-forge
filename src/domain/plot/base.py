import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_FIGURE_FORMAT = "pdf"


@dataclass
class Plot:
    """Self-contained figure payload: the rendered bytes and their format."""

    data: bytes
    format: str = "png"


def set_figure_format(figure_format: str) -> None:
    """Set the rendering format ("pdf" or "png") for every Plot created after."""
    global _FIGURE_FORMAT
    if figure_format not in ("pdf", "png"):
        raise ValueError(
            f"Unsupported figure format: {figure_format!r}. Use 'pdf' or 'png'."
        )
    _FIGURE_FORMAT = figure_format


def _fig_to_plot(fig: Figure) -> Plot:
    """Render a figure into a Plot payload and close it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format=_FIGURE_FORMAT, bbox_inches="tight")
    plt.close(fig)
    return Plot(data=buffer.getvalue(), format=_FIGURE_FORMAT)


def _ensure_ax(
    ax: Axes | None,
    figsize: tuple[float, float],
) -> tuple[Axes, Figure | None]:
    """Return the given axes, or a fresh axes and the figure owning it."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return ax, fig
    return ax, None


def _finalize(fig: Figure | None) -> Plot | None:
    """Render an owned figure, or None when drawing onto a caller's axes."""
    if fig is None:
        return None
    return _fig_to_plot(fig)


def _apply_labels(
    ax: Axes,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
) -> None:
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)


def _smart_legend_loc(ax: Axes, X: np.ndarray, max_points: int = 5000) -> str:
    """Pick the legend corner with the fewest data points."""
    if X.size == 0:
        return "best"
    points = X
    if len(points) > max_points:
        sampled = np.random.default_rng(0).choice(
            len(points), max_points, replace=False
        )
        points = points[sampled]

    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    if x_lo == x_hi or y_lo == y_hi:
        x_lo, x_hi = float(points[:, 0].min()), float(points[:, 0].max())
        y_lo, y_hi = float(points[:, 1].min()), float(points[:, 1].max())
        if x_lo == x_hi or y_lo == y_hi:
            return "best"
    x_mid = 0.5 * (x_lo + x_hi)
    y_mid = 0.5 * (y_lo + y_hi)

    counts = {
        "upper left": int(((points[:, 0] < x_mid) & (points[:, 1] >= y_mid)).sum()),
        "upper right": int(((points[:, 0] >= x_mid) & (points[:, 1] >= y_mid)).sum()),
        "lower left": int(((points[:, 0] < x_mid) & (points[:, 1] < y_mid)).sum()),
        "lower right": int(((points[:, 0] >= x_mid) & (points[:, 1] < y_mid)).sum()),
    }
    counts_values = list(counts.values())
    if min(counts_values) > 0 and max(counts_values) / min(counts_values) < 1.3:
        return "best"
    return min(counts, key=counts.get)


def _format_value(value: float, *, kind: str = "auto") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if kind == "score":
        return f"{value:.3f}"
    if kind == "normalized_cm":
        return f"{value:.2f}"
    if kind == "count":
        return f"{int(value):d}"
    magnitude = abs(value)
    if magnitude == 0:
        return "0"
    if magnitude >= 1000:
        return f"{value:.0f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"
