import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def dict_to_bar_plot(
    d: dict, title: str = "Metrics", ylim: tuple = (0, 1)
) -> plt.Figure:
    """Plot dictionary as a bar plot."""
    fig, ax = plt.subplots()
    bars = sns.barplot(
        x=list(d.keys()),
        y=list(d.values()),
        hue=list(d.keys()),
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title(title, pad=20)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Rotate x-axis labels to prevent overlap
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add value labels on top of each bar
    for i, (key, value) in enumerate(d.items()):
        ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def dict_to_table(d: dict, title: str = "Metrics") -> plt.Figure:
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
    ax.set_title(title)
    fig.tight_layout()
    return fig
