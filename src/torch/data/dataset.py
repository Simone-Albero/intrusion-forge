import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


Sample = tuple[list[torch.Tensor], list[torch.Tensor]]


def _to_tensor(
    data: pd.DataFrame | pd.Series | np.ndarray, dtype: torch.dtype
) -> torch.Tensor:
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values
    return torch.as_tensor(data, dtype=dtype)


class TensorDataset(Dataset):
    """Wraps a DataFrame/array (and optional labels) into a torch Dataset."""

    def __init__(
        self,
        features: pd.DataFrame | np.ndarray,
        labels: pd.Series | np.ndarray | None = None,
        feature_dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.long,
    ) -> None:
        self.features = _to_tensor(features, feature_dtype)
        self.labels = _to_tensor(labels, label_dtype) if labels is not None else None

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int) -> Sample:
        features = [self.features[index]]
        labels = [self.labels[index]] if self.labels is not None else features
        return features, labels


class TabularDataset(Dataset):
    """Dataset for tabular data with mixed numerical and categorical features.

    Args:
        df: Source DataFrame.
        num_cols: Numerical column names.
        cat_cols: Categorical column names.
        label_col: Single label column name or list of label column names.

    Returns:
        ``(features, labels)`` where each is a list of tensors. When no labels
        are provided, ``labels`` is the same list as ``features``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        num_cols: list[str] | None = None,
        cat_cols: list[str] | None = None,
        label_col: str | list[str] | None = None,
    ) -> None:
        if not num_cols and not cat_cols:
            raise ValueError("At least one of num_cols or cat_cols must be provided.")

        self.numerical_features = (
            torch.as_tensor(df[num_cols].values.copy(), dtype=torch.float32)
            if num_cols
            else None
        )
        self.categorical_features = (
            torch.as_tensor(df[cat_cols].values.copy(), dtype=torch.long)
            if cat_cols
            else None
        )

        if label_col is not None:
            label_cols = [label_col] if isinstance(label_col, str) else list(label_col)
            self.labels: list[torch.Tensor] | None = [
                torch.as_tensor(df[col].values.copy(), dtype=torch.long)
                for col in label_cols
            ]
        else:
            self.labels = None

        self._length = len(df)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Sample:
        features = [
            t[index]
            for t in (self.numerical_features, self.categorical_features)
            if t is not None
        ]
        labels = (
            [t[index] for t in self.labels] if self.labels is not None else features
        )
        return features, labels
