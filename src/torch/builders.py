from typing import Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data.dataset import TabularDataset
from .data.batch import default_collate
from .model import ModelFactory
from .loss import LossFactory


def create_dataset(
    df: pd.DataFrame,
    num_cols: list,
    cat_cols: list,
    label_col: Optional[str] = None,
) -> TabularDataset:
    return TabularDataset(
        df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=label_col,
    )


def create_dataloader(dataset: Dataset, params: dict) -> DataLoader:
    return DataLoader(dataset, collate_fn=default_collate, **params)


def create_model(name: str, params: dict, device: torch.device) -> nn.Module:
    model = ModelFactory.create(name, params).to(device)
    return model


def create_loss(name: str, params: dict, device: torch.device) -> nn.Module:
    loss_fn = LossFactory.create(name, params).to(device)
    return loss_fn


def create_optimizer(
    name: str, params: dict, model: nn.Module, loss_fn: Optional[nn.Module] = None
) -> torch.optim.Optimizer:
    params_to_optimize = model.parameters()
    if loss_fn is not None and len(list(loss_fn.parameters())) > 0:
        params_to_optimize = list(model.parameters()) + list(loss_fn.parameters())

    optimizer = torch.optim.__dict__[name](params_to_optimize, **params)
    return optimizer


def create_scheduler(
    name: str, params: dict, optimizer: torch.optim.Optimizer, dataloader: DataLoader
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if name is None:
        return None

    if params.get("steps_per_epoch") == "auto":
        params["steps_per_epoch"] = len(dataloader)

    scheduler = torch.optim.lr_scheduler.__dict__[name](optimizer, **params)

    return scheduler
