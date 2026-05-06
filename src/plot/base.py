import io
from dataclasses import dataclass

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .style import FILL_COLORS, OUTLINE_COLORS


@dataclass
class Plot:
    """Immutable, self-contained figure payload.

    Created by _fig_to_plot; the source Figure is closed before this object
    is returned, so no matplotlib state leaks to callers.
    """

    data: bytes
    format: str = "png"


def _fig_to_plot(fig: Figure) -> Plot:
    """Render figure to PNG bytes, close it, and return a Plot."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return Plot(data=buf.getvalue())


def _get_fill_cmap(n: int):
    """Return a ListedColormap with n high-contrast fill colors."""
    colors = (FILL_COLORS * ((n // len(FILL_COLORS)) + 1))[:n]
    return mcolors.ListedColormap(colors)
