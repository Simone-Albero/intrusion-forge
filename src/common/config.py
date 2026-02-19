import os
from pathlib import Path
from typing import List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from omegaconf import OmegaConf


from src.common.utils import save_to_json


def load_config(
    config_path: str = "configs",
    config_name: str = "config",
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    Load and return a DictConfig using Hydra's Compose API.

    Args:
        config_path: Path to the configuration folder. Can be:
            - Relative path from the caller (e.g., "configs", "../configs")
            - Absolute path (e.g., "/Users/user/project/configs")
        config_name: Name of the main configuration file (without .yaml extension)
        overrides: List of overrides like ["db.user=admin", "db.port=1234"]

    Returns:
        DictConfig: The composed configuration
    """
    overrides = overrides or []

    # Convert to absolute path if it's relative
    config_dir = Path(config_path)
    if not config_dir.is_absolute():
        # Get the caller's directory (typically project root)
        caller_frame = os.path.dirname(os.path.abspath(__file__))
        # Go up to project root (adjust levels as needed)
        project_root = Path(caller_frame).parent.parent.parent
        config_dir = project_root / config_path

    config_dir = config_dir.resolve()

    if not config_dir.exists():
        raise ValueError(
            f"Configuration directory does not exist: {config_dir}\n"
            f"Current working directory: {os.getcwd()}"
        )

    # Use initialize_config_dir for absolute paths
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)

    save_to_json(
        OmegaConf.to_container(cfg, resolve=True),
        Path(cfg.path.configs) / f"{config_name}_composed.json",
    )

    cfg = DictConfig(cfg)
    return cfg
