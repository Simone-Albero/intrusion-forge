from pathlib import Path

from src.core.paths import OutputPaths


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
