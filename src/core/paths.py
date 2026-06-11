from dataclasses import dataclass
from pathlib import Path


@dataclass
class OutputPaths:
    """Output paths for the pipeline.

    Shared across classifiers (dataset-level):
      - `processed_data`: train/val/test files written by prepare_data
      - `shared`:         complexity, config snapshot, timing; `metadata/` holds df_meta, df_info, clusters_meta

    Per-classifier (resolved against `${classifier.name}`):
      - `configs`:  composed Hydra configs snapshot
      - `outputs`:  training/testing/analysis JSON outputs
      - `pickle`:   binary side artifacts (confusion matrices, etc.)
      - `models`:   serialized estimators / checkpoints
      - `figures`:  rendered figure files (PNG)
    """

    processed_data: Path
    shared: Path
    configs: Path
    outputs: Path
    pickle: Path
    models: Path
    figures: Path
