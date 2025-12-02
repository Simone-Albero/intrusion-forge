from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader

from .dataset import MixedTabularDataset
from .batch import default_collate


def create_dataset(
    df: pd.DataFrame,
    num_cols: list,
    cat_cols: list,
    label_col: Optional[str] = None,
) -> MixedTabularDataset:
    """Create a MixedTabularDataset from a dataframe.

    Args:
        df: Input dataframe
        num_cols: List of numerical column names
        cat_cols: List of categorical column names
        label_col: Name of the label column (optional for unsupervised)

    Returns:
        MixedTabularDataset instance
    """
    return MixedTabularDataset(
        df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=label_col,
    )


def create_dataloader(
    dataset: MixedTabularDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Create a DataLoader with specified configuration.

    Args:
        dataset: Dataset to load from
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Configured DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=default_collate,
    )
