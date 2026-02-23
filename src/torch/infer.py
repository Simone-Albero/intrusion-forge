from collections.abc import Callable

import torch
import torch.nn.functional as F

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
            result.append(torch.tensor(df[cols].to_numpy(), dtype=dtype))
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
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Default prediction extractor: softmax over logits, optional z embedding."""
    probs = F.softmax(output["logits"].cpu(), dim=1)
    z = output["z"].cpu() if "z" in output else None
    return probs.argmax(dim=1), z, probs.max(dim=1).values


def get_predictions(
    model: BaseModel,
    inputs: list[torch.Tensor],
    y: torch.Tensor,
    device: torch.device,
    pred_fn: (
        Callable[[ModelOutput], tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]]
        | None
    ) = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run inference and return (y_true, y_pred, z, confidences).

    Args:
        model: The model to run.
        inputs: Feature tensors (zero-width tensors are skipped).
        y: Ground-truth labels tensor.
        device: Target device.
        pred_fn: Optional callable ``(ModelOutput) -> (y_pred, z, confidences)``.
                 Defaults to softmax over logits with optional z embedding.

    Returns:
        (y_true, y_pred, z, confidences) as torch Tensors (z may be None).
    """
    output = run_model(model, inputs, device)
    y_pred, z, confidences = (pred_fn or _default_pred_fn)(output)

    return y.flatten().cpu(), y_pred.flatten(), z, confidences.flatten()
