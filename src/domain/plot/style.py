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


# One entry per variant of `VARIANTS`.
BASELINE_LABEL: dict[str, str] = {
    "regressor": "regressor",
    "sample_regressor": "sample regressor",
    "atc": "ATC",
    "mcp": "MCP",
    "train_empiric": "train empiric",
    "val_empiric": "val empiric",
    "atc_cal": "ATC (cal.)",
    "mcp_cal": "MCP (cal.)",
    "train_empiric_cal": "train empiric (cal.)",
    "atc_combo": "regressor + ATC",
    "mcp_combo": "regressor + MCP",
    "train_empiric_combo": "regressor + train empiric",
    "val_empiric_combo": "regressor + val empiric",
    "sample_regressor_combo": "regressor + sample regressor",
}
# A calibrated variant takes its raw one's hue half transparent, so each pair reads as
# one; a combo takes its baseline's.
BASELINE_COLOR: dict[str, str] = {
    "regressor": PALETTE[2],
    "sample_regressor": PALETTE[3],
    "atc": PALETTE[1],
    "mcp": PALETTE[0],
    "train_empiric": PALETTE[7],
    "val_empiric": HIGHLIGHT_COLOR,
    "atc_cal": PALETTE[1] + "80",
    "mcp_cal": PALETTE[0] + "80",
    "train_empiric_cal": PALETTE[7] + "80",
    "atc_combo": PALETTE[1],
    "mcp_combo": PALETTE[0],
    "train_empiric_combo": PALETTE[7],
    "val_empiric_combo": HIGHLIGHT_COLOR,
    "sample_regressor_combo": PALETTE[3],
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
