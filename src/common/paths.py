from dataclasses import dataclass
from pathlib import Path


@dataclass
class OutputPaths:
    """Output paths for the pipeline.

    Shared across classifiers (dataset-level):
      - `processed_data`: train/val/test files written by prepare_data
      - `data_logs`:      df_meta, clusters_meta, complexity metrics
      - `tb_logs`:        TensorBoard log root (sub-runs use sub-directories)

    Per-classifier (resolved against `${classifier.name}`):
      - `configs`:        composed Hydra configs snapshot
      - `json_logs`:      training/testing/analysis JSON outputs
      - `pickle`:         binary side artifacts (confusion matrices, etc.)
      - `models`:         serialized estimators / checkpoints
    """

    processed_data: Path
    data_logs: Path
    tb_logs: Path
    configs: Path
    json_logs: Path
    pickle: Path
    models: Path
