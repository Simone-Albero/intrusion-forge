from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config(
    *,
    config_path: str | Path = "configs",
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Compose a DictConfig via Hydra from `config_path` and dotlist `overrides`."""
    overrides = overrides or []

    config_dir = Path(config_path)
    if not config_dir.is_absolute():
        config_dir = Path.cwd() / config_path
    config_dir = config_dir.resolve()

    if not config_dir.exists():
        raise ValueError(
            f"Configuration directory does not exist: {config_dir}\n"
            f"Current working directory: {Path.cwd()}"
        )

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return DictConfig(cfg)


def to_container(cfg) -> dict:
    """Convert an OmegaConf config (or sub-node) to plain Python types."""
    return OmegaConf.to_container(cfg, resolve=True)


def select_config(cfg, keys: tuple[str, ...]) -> dict:
    """The slice of `cfg` named by dotted `keys`, as plain Python values under those keys."""
    selected = {}
    for key in keys:
        node = cfg
        for part in key.split("."):
            node = node[part]
        selected[key] = (
            to_container(node) if isinstance(node, (DictConfig, ListConfig)) else node
        )
    return selected
