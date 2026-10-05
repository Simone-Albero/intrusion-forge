import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from src.domain.plot.base import Plot, _fig_to_plot
from src.domain.plot.primitives import numeric_scatter_plot


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
