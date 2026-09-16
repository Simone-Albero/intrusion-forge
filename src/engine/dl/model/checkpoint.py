import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _parse_loss(path: Path) -> float:
    """Validation loss encoded in a checkpoint filename, inf when absent."""
    try:
        return float(path.stem.split("loss=")[1])
    except (IndexError, ValueError):
        return float("inf")


def _load(
    path: Path, model: nn.Module, device: torch.device, weights_only: bool
) -> None:
    """Load a checkpoint's state dict into `model`."""
    logger.info("Loading checkpoint from %s", path)
    checkpoint = torch.load(path, map_location=device, weights_only=weights_only)
    model.load_state_dict(checkpoint)


def load_best_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    device: torch.device,
    *,
    weights_only: bool = True,
) -> None:
    """Load the checkpoint with the lowest recorded loss, falling back to the newest."""
    files = list(checkpoint_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}; refusing to continue "
            "with randomly initialized weights."
        )
    best = min(files, key=_parse_loss)
    if _parse_loss(best) == float("inf"):
        logger.warning(
            "Could not parse loss from any checkpoint filename in %s; "
            "falling back to the most recent file by mtime",
            checkpoint_dir,
        )
        best = max(files, key=lambda p: p.stat().st_mtime)
    _load(best, model, device, weights_only)
