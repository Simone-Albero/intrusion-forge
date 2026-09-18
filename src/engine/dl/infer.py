import pandas as pd
import torch

from .model.base import BaseModel, ModelOutput


def df_to_tensors(
    df: pd.DataFrame,
    col_groups: list[list[str]],
    *,
    dtypes: list[torch.dtype] | None = None,
) -> list[torch.Tensor]:
    """One tensor per column group; empty groups produce zero-width tensors."""
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
    """Move inputs to device and run a forward pass."""
    model.eval()
    with torch.no_grad():
        return model(*[t.to(device) for t in inputs])
