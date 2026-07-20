import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.core.plot import Plot

_FIGURE_FORMAT = "pdf"


def set_figure_format(fmt: str) -> None:
    """Set the rendering format ("pdf" or "png") for every Plot created after."""
    global _FIGURE_FORMAT
    if fmt not in ("pdf", "png"):
        raise ValueError(f"Unsupported figure format: {fmt!r}. Use 'pdf' or 'png'.")
    _FIGURE_FORMAT = fmt


def _fig_to_plot(fig: Figure) -> Plot:
    buf = io.BytesIO()
    fig.savefig(buf, format=_FIGURE_FORMAT, bbox_inches="tight")
    plt.close(fig)
    return Plot(data=buf.getvalue(), format=_FIGURE_FORMAT)


def _ensure_ax(
    ax: Axes | None,
    figsize: tuple[float, float],
) -> tuple[Axes, Figure | None]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return ax, fig
    return ax, None


def _finalize(fig: Figure | None) -> Plot | None:
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
    pts = X
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    if x_lo == x_hi or y_lo == y_hi:
        x_lo, x_hi = float(pts[:, 0].min()), float(pts[:, 0].max())
        y_lo, y_hi = float(pts[:, 1].min()), float(pts[:, 1].max())
        if x_lo == x_hi or y_lo == y_hi:
            return "best"
    x_mid = 0.5 * (x_lo + x_hi)
    y_mid = 0.5 * (y_lo + y_hi)

    counts = {
        "upper left": int(((pts[:, 0] < x_mid) & (pts[:, 1] >= y_mid)).sum()),
        "upper right": int(((pts[:, 0] >= x_mid) & (pts[:, 1] >= y_mid)).sum()),
        "lower left": int(((pts[:, 0] < x_mid) & (pts[:, 1] < y_mid)).sum()),
        "lower right": int(((pts[:, 0] >= x_mid) & (pts[:, 1] < y_mid)).sum()),
    }
    counts_values = list(counts.values())
    if min(counts_values) > 0 and max(counts_values) / min(counts_values) < 1.3:
        return "best"
    return min(counts, key=counts.get)


def _format_value(v: float, *, kind: str = "auto") -> str:
    """Adaptive numeric formatting for plot annotations."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if kind == "score":
        return f"{v:.3f}"
    if kind == "normalized_cm":
        return f"{v:.2f}"
    if kind == "count":
        return f"{int(v):d}"
    abs_v = abs(v)
    if abs_v == 0:
        return "0"
    if abs_v >= 1000:
        return f"{v:.0f}"
    if abs_v >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"
