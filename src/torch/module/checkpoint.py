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
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))

    if not checkpoint_files:
        logger.warning("No checkpoint found, using current model state")
        return

    # Sort by loss value in filename (lower is better)
    def extract_loss_value(path: Path) -> float:
        """Extract loss value from checkpoint filename.

        Handles format: best_model_N_loss=X.XXXX where N is epoch and X.XXXX is loss value.
        Supports negative loss values.
        """
        try:
            # Split by 'loss=' and extract the value after it
            loss_part = path.stem.split("loss=")[1]
            # The loss value is everything after 'loss=' until end of filename
            return float(loss_part)
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to parse loss from {path.name}: {e}")
            return float("inf")  # Return infinity so it won't be selected as best

    try:
        best_checkpoint = min(checkpoint_files, key=extract_loss_value)
    except ValueError:
        # Fallback to most recent if parsing fails for all files
        logger.warning(
            "Could not parse loss from any checkpoint filename, using most recent"
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


def load_latest_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    device: torch.device,
    weights_only: bool = True,
) -> None:
    """Load the latest model checkpoint if available.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        model: Model to load checkpoint into
        device: Device to map checkpoint to
        weights_only: Whether to load only weights (safer)
    Raises:
        Exception: If checkpoint loading fails
    """
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))

    if not checkpoint_files:
        logger.warning("No checkpoint found, using current model state")
        return

    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)

    logger.info(f"Loading latest model from {latest_checkpoint}")
    try:
        checkpoint = torch.load(
            latest_checkpoint, map_location=device, weights_only=weights_only
        )
        model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise
