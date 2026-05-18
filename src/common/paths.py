from dataclasses import dataclass
from pathlib import Path


@dataclass
class OutputPaths:
    """Output paths for the analysis pipeline."""

    json_logs: Path
    tb_logs: Path
    processed_data: Path
    configs: Path
    pickle: Path
