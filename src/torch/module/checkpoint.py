import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _parse_loss(path: Path) -> float:
    try:
        return float(path.stem.split("loss=")[1])
    except (IndexError, ValueError):
        return float("inf")


def _load(
    path: Path, model: nn.Module, device: torch.device, weights_only: bool
) -> None:
    logger.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=weights_only)
    model.load_state_dict(checkpoint)


def load_best_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    device: torch.device,
    weights_only: bool = True,
) -> None:
    files = list(checkpoint_dir.glob("*.pt"))
    if not files:
        logger.warning("No checkpoint found, using current model state")
        return
    best = min(files, key=_parse_loss)
    if _parse_loss(best) == float("inf"):
        logger.warning(
            "Could not parse loss from any checkpoint filename, using most recent"
        )
        best = max(files, key=lambda p: p.stat().st_mtime)
    _load(best, model, device, weights_only)


def load_latest_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    device: torch.device,
    weights_only: bool = True,
) -> None:
    files = list(checkpoint_dir.glob("*.pt"))
    if not files:
        logger.warning("No checkpoint found, using current model state")
        return
    _load(max(files, key=lambda p: p.stat().st_mtime), model, device, weights_only)
