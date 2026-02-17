from typing import Tuple, Optional, Sequence, List, Union

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


Sample = Tuple[List[torch.Tensor], List[torch.Tensor]]


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
        features = [self.features[index]]
        labels = [self.labels[index]] if self.labels is not None else features
        return features, labels


class TabularDataset(Dataset):
    """
    Dataset for tabular data supporting numerical, categorical, or mixed features.

    Args:
        df: DataFrame containing the data
        num_cols: List of numerical column names (optional)
        cat_cols: List of categorical column names (optional)
        label_col: Name or list of label column names (optional)

    Returns:
        Sample tuple where:
        - features is a list of tensors [numerical, categorical] or single tensor
        - labels is a list of tensors (one per label column) or same as features if no labels
    """

    def __init__(
        self,
        df: pd.DataFrame,
        num_cols: Optional[Sequence[str]] = None,
        cat_cols: Optional[Sequence[str]] = None,
        label_col: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        self.has_numerical = num_cols is not None and len(num_cols) > 0
        self.has_categorical = cat_cols is not None and len(cat_cols) > 0

        if not self.has_numerical and not self.has_categorical:
            raise ValueError("At least one of num_cols or cat_cols must be provided")

        self.numerical_features: Optional[torch.Tensor] = None
        self.categorical_features: Optional[torch.Tensor] = None

        if self.has_numerical:
            self.numerical_features = torch.as_tensor(
                df[num_cols].values.copy(), dtype=torch.float32
            )

        if self.has_categorical:
            self.categorical_features = torch.as_tensor(
                df[cat_cols].values.copy(), dtype=torch.long
            )

        # Support single or multiple label columns
        self.labels: Optional[List[torch.Tensor]] = None
        if label_col is not None:
            label_cols = [label_col] if isinstance(label_col, str) else list(label_col)
            self.labels = [
                torch.as_tensor(df[col].values.copy(), dtype=torch.long)
                for col in label_cols
            ]

        self._length = len(df)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Sample:
        if self.has_numerical and self.has_categorical:
            features = [
                self.numerical_features[index],
                self.categorical_features[index],
            ]
        elif self.has_numerical:
            features = [self.numerical_features[index]]
        else:
            features = [self.categorical_features[index]]

        labels = (
            [label_tensor[index] for label_tensor in self.labels]
            if self.labels is not None
            else features
        )
        return features, labels
