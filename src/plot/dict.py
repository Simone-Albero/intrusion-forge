import matplotlib.pyplot as plt
import seaborn as sns

from .base import Plot, _fig_to_plot
from .style import (
    TITLE_FONTSIZE,
    TITLE_PAD,
    LEGEND_FONTSIZE,
    TICK_LABELSIZE,
    LEGEND_FRAMEALPHA,
)


def dict_to_bar_plot(
    d: dict, title: str = "Metrics", ylim: tuple[float, float] | None = (0, 1)
) -> Plot:
    """Plot dictionary as a bar plot."""
    fig, ax = plt.subplots()
    sns.barplot(
        x=list(d.keys()),
        y=list(d.values()),
        hue=list(d.keys()),
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    for i, (key, value) in enumerate(d.items()):
        ax.text(
            i,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=LEGEND_FONTSIZE,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=TICK_LABELSIZE)
    fig.tight_layout()
    return _fig_to_plot(fig)


def dict_to_table(d: dict, title: str = "Metrics") -> Plot:
    """Plot dictionary as a table."""
    fig, ax = plt.subplots()
    ax.axis("off")
    table_data = [
        [k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in d.items()
    ]
    table = ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    fig.tight_layout()
    return _fig_to_plot(fig)
