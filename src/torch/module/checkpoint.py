from pathlib import Path
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_best_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    device: torch.device,
    weights_only: bool = True,
) -> None:
    """Load the best model checkpoint if available.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        model: Model to load checkpoint into
        device: Device to map checkpoint to
        weights_only: Whether to load only weights (safer)

    Raises:
        Exception: If checkpoint loading fails
    """
    checkpoint_files = list(checkpoint_dir.glob("best_model_*.pt"))

    if not checkpoint_files:
        logger.warning("No checkpoint found, using current model state")
        return

    # Sort by validation loss in filename (lower is better)
    def extract_loss_value(path: Path) -> float:
        """Extract loss value from checkpoint filename, handling negative values."""
        try:
            # Split by 'val_loss=' and get the part after it
            loss_part = path.stem.split("val_loss=")[1]
            # Remove any trailing suffixes and convert to float
            # This handles formats like: "val_loss=-0.5_epoch=10" or "val_loss=0.5"
            loss_str = loss_part.split("_")[0] if "_" in loss_part else loss_part
            return float(loss_str)
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to parse loss from {path.name}: {e}")
            return float("inf")  # Return infinity so it won't be selected as best

    try:
        best_checkpoint = min(checkpoint_files, key=extract_loss_value)
    except (IndexError, ValueError):
        # Fallback to modification time if parsing fails
        logger.warning(
            "Could not parse validation loss from filename, using most recent"
        )
        best_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)

    logger.info(f"Loading best model from {best_checkpoint}")
    try:
        checkpoint = torch.load(
            best_checkpoint, map_location=device, weights_only=weights_only
        )
        model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise
