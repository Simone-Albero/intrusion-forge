from pathlib import Path

from src.core.paths import OutputPaths
from src.core.utils import load_from_json


def paths_from_cfg(cfg) -> OutputPaths:
    """Build the resolved output layout from the hydra `cfg.path` block."""
    return OutputPaths(
        processed_data=Path(cfg.path.processed_data),
        shared=Path(cfg.path.shared),
        configs=Path(cfg.path.configs),
        outputs=Path(cfg.path.outputs),
        pickle=Path(cfg.path.pickle),
        models=Path(cfg.path.models),
        figures=Path(cfg.path.figures),
    )


def load_prepared_metadata(path: Path) -> dict:
    """Load one of prepare's metadata files, refusing one without a `classes` table."""
    metadata = load_from_json(path)
    if "classes" not in metadata:
        raise ValueError(
            f"{path} predates the current artifact format: "
            "re-run `make prepare FORCE=1`."
        )
    return metadata
