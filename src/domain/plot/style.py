from pathlib import Path

import matplotlib.pyplot as plt

STYLE_PATH: Path = Path(__file__).parent / "default.mplstyle"

PALETTE: list[str] = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#CC78BC",
    "#CA9161",
    "#949494",
    "#ECE133",
    "#56B4E9",
]

HIGHLIGHT_COLOR: str = "#B22222"
MUTED_COLOR: str = "#777777"
NEUTRAL_COLOR: str = "#cccccc"
# The test-set noise floor of the by-size figures; no baseline takes it.
NOISE_COLOR: str = PALETTE[5]


# One entry per baseline variant, in the order of `BASELINE_VARIANTS`.
BASELINE_LABEL: dict[str, str] = {
    "mcp_region": "MCP",
    "atc_region": "ATC",
    "region": "regressor",
    "combo_rankavg": "regressor + MCP",
    "combo_atc_rankavg": "regressor + ATC",
    "train_rate_region": "train rate",
    "val_rate_region": "val rate",
}
BASELINE_COLOR: dict[str, str] = {
    "mcp_region": PALETTE[0],
    "atc_region": PALETTE[1],
    "region": PALETTE[2],
    "combo_rankavg": PALETTE[3],
    "combo_atc_rankavg": PALETTE[4],
    "train_rate_region": PALETTE[7],
    "val_rate_region": HIGHLIGHT_COLOR,
}


def extended_palette(n: int) -> list[str]:
    """Return n distinct hex colors: PALETTE first, then tab20 for overflow."""
    if n <= len(PALETTE):
        return PALETTE[:n]
    tab20_rgb = plt.get_cmap("tab20").colors
    tab20_hex = [
        "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b in tab20_rgb
    ]
    pool = PALETTE + [c for c in tab20_hex if c not in PALETTE]
    return (pool * ((n // len(pool)) + 1))[:n]


def apply_plot_style() -> None:
    """Apply the project's matplotlib style. Idempotent."""
    plt.style.use(str(STYLE_PATH))
