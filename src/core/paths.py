from dataclasses import dataclass
from pathlib import Path


@dataclass
class OutputPaths:
    """Resolved output layout: dataset-level dirs plus the per-classifier ones."""

    processed_data: Path
    shared: Path
    configs: Path
    outputs: Path
    pickle: Path
    models: Path
    figures: Path
