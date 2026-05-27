from dataclasses import dataclass, replace
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


def with_variant(paths: OutputPaths, variant: str) -> OutputPaths:
    """Nest per-classifier dirs under `variant`; dataset-level paths unchanged.

    e.g. ``models  .../tabular/models  ->  .../tabular/{variant}/models``. The
    shared dataset-level paths (``processed_data``, ``shared``) are kept as-is.
    """
    def nest(p: Path) -> Path:
        return p.parent / variant / p.name

    return replace(
        paths,
        configs=nest(paths.configs),
        outputs=nest(paths.outputs),
        pickle=nest(paths.pickle),
        models=nest(paths.models),
        figures=nest(paths.figures),
    )
