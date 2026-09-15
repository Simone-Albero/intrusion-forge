import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data.batch import default_collate
from .data.dataset import TabularDataset
from .loss import LossFactory


def create_dataset(
    df: pd.DataFrame,
    num_cols: list,
    cat_cols: list,
    *,
    label_col: str | None = None,
) -> TabularDataset:
    """Wrap a DataFrame in a TabularDataset over the given feature columns."""
    return TabularDataset(df, num_cols=num_cols, cat_cols=cat_cols, label_col=label_col)


def create_dataloader(dataset: Dataset, params: dict) -> DataLoader:
    """Build a DataLoader with the project's collate function."""
    return DataLoader(dataset, collate_fn=default_collate, **params)


def create_loss(name: str, params: dict, device: torch.device) -> nn.Module:
    """Instantiate a registered loss on `device`."""
    return LossFactory.create(name, params).to(device)


def create_optimizer(
    name: str,
    params: dict,
    model: nn.Module,
    *,
    loss_fn: nn.Module | None = None,
) -> torch.optim.Optimizer:
    """Build a torch optimizer over the model's and the loss's trainable parameters."""
    trainable = list(model.parameters())
    if loss_fn is not None and list(loss_fn.parameters()):
        trainable += list(loss_fn.parameters())
    return getattr(torch.optim, name)(trainable, **params)


def create_scheduler(
    name: str | None,
    params: dict,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Build a torch LR scheduler, resolving an "auto" steps_per_epoch from the loader."""
    if name is None:
        return None
    if params.get("steps_per_epoch") == "auto":
        params = dict(params)
        params["steps_per_epoch"] = len(dataloader)
    return getattr(torch.optim.lr_scheduler, name)(optimizer, **params)
