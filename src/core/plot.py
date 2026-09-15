from dataclasses import dataclass


@dataclass
class Plot:
    """Self-contained figure payload: the rendered bytes and their format."""

    data: bytes
    format: str = "png"
