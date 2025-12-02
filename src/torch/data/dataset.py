from typing import Tuple, Optional, Sequence, List, Union

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


TensorGroup = Union[List[torch.Tensor], torch.Tensor]
Sample = Tuple[TensorGroup, Optional[TensorGroup]]


class TensorDataset(Dataset):
    """
    Wraps a pandas DataFrame (and optional pandas Series) into torch tensors.
    """

    def __init__(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        labels: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.long,
    ) -> None:
        self.features: torch.Tensor = torch.as_tensor(
            features.values if isinstance(features, pd.DataFrame) else features,
            dtype=feature_dtype,
        )
        self.labels: Optional[torch.Tensor] = (
            torch.as_tensor(
                labels.values if isinstance(labels, pd.Series) else labels,
                dtype=label_dtype,
            )
            if labels is not None
            else None
        )

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int) -> Sample:
        features = self.features[index]
        labels = self.labels[index] if self.labels is not None else features
        return features, labels


class NumericalTensorDataset(TensorDataset):
    """
    Dataset wrapper for numerical features in a DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        num_cols: Sequence[str],
        label_col: Optional[str] = None,
    ) -> None:
        features = df[num_cols]
        labels = df[label_col] if label_col else None
        super().__init__(
            features,
            labels,
            feature_dtype=torch.float32,
            label_dtype=torch.long,
        )


class CategoricalTensorDataset(TensorDataset):
    """
    Dataset wrapper for categorical features in a DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: Sequence[str],
        label_col: Optional[str] = None,
    ) -> None:
        features = df[cat_cols]
        labels = df[label_col] if label_col else None

        super().__init__(
            features,
            labels,
            feature_dtype=torch.long,
            label_dtype=torch.long,
        )


class MixedTabularDataset(Dataset):
    """
    Combines numerical and categorical datasets into a single dataset yielding TabularSample.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        num_cols: Sequence[str],
        cat_cols: Sequence[str],
        label_col: Optional[str] = None,
    ) -> None:
        self.numerical_ds = NumericalTensorDataset(df, num_cols, label_col=None)
        self.categorical_ds = CategoricalTensorDataset(df, cat_cols, label_col=None)

        self.labels: Optional[torch.Tensor] = (
            torch.as_tensor(df[label_col].values, dtype=torch.long)
            if label_col
            else None
        )

    def __len__(self) -> int:
        return len(self.numerical_ds)

    def __getitem__(self, index: int) -> Sample:
        num_sample = self.numerical_ds[index][0]
        cat_sample = self.categorical_ds[index][0]
        features = [num_sample, cat_sample]
        labels = self.labels[index] if self.labels is not None else features
        return features, labels
