from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from scipy.special import softmax

from .model.base import BaseModel, ModelOutput


def df_to_tensors(
    df,
    col_groups: list[list[str]],
    dtypes: list[torch.dtype] | None = None,
) -> list[torch.Tensor]:
    """Build one tensor per column group from a DataFrame.

    Args:
        df: Source DataFrame.
        col_groups: Each inner list is a group of column names for one tensor.
        dtypes: Dtype per group. Defaults to float32 for all groups.

    Returns:
        List of tensors, one per group. Empty groups produce zero-width tensors.
    """
    dtypes = dtypes or [torch.float32] * len(col_groups)
    result = []
    for cols, dtype in zip(col_groups, dtypes):
        if cols:
            result.append(torch.as_tensor(df[cols].to_numpy(), dtype=dtype))
        else:
            result.append(torch.empty(len(df), 0, dtype=dtype))
    return result


def run_model(
    model: BaseModel,
    inputs: list[torch.Tensor],
    device: torch.device,
) -> ModelOutput:
    """Move inputs to device and run a forward pass, skipping zero-width tensors."""
    model.eval()
    with torch.no_grad():
        return model(*[t.to(device) for t in inputs if t.size(1) > 0])


def _default_pred_fn(
    output: ModelOutput,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Default prediction extractor: softmax over logits, optional z embedding."""
    probs = softmax(output["logits"].cpu().numpy(), axis=1)
    z = output["z"].cpu().numpy() if "z" in output else None
    return probs.argmax(axis=1), z, probs.max(axis=1)


def get_predictions(
    model: BaseModel,
    inputs: list[torch.Tensor],
    y: torch.Tensor,
    device: torch.device,
    pred_fn: (
        Callable[[ModelOutput], tuple[np.ndarray, np.ndarray | None, np.ndarray]] | None
    ) = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Run inference and return (y_true, y_pred, z, confidences).

    Args:
        model: The model to run.
        inputs: Feature tensors (zero-width tensors are skipped).
        y: Ground-truth labels tensor.
        device: Target device.
        pred_fn: Optional callable ``(ModelOutput) -> (y_pred, z, confidences)``.
                 Defaults to softmax over logits with optional z embedding.

    Returns:
        (y_true, y_pred, z, confidences) as numpy arrays (z may be None).
    """
    output = run_model(model, inputs, device)
    y_pred, z, confidences = (pred_fn or _default_pred_fn)(output)
    return y.numpy(), y_pred, z, confidences
