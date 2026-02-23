import hashlib

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def drop_nans(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Drop rows with NaN or infinite values in specified columns."""
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)


def query_filter(df: pd.DataFrame, query: str | None = None) -> pd.DataFrame:
    """Filter DataFrame using a query string."""
    return df.query(query) if query else df


def rare_category_filter(
    df: pd.DataFrame, cat_cols: list, min_count: int = 3000
) -> pd.DataFrame:
    """Remove rows with rare categories in specified categorical columns."""
    if not min_count or min_count <= 0:
        return df
    df = df.copy()
    for col in cat_cols:
        counts = df[col].value_counts()
        df = df[~df[col].isin(counts[counts < min_count].index)]
    return df


def subsample_df(
    df: pd.DataFrame,
    n_samples: int,
    random_state: int | None = None,
    label_col: str | None = None,
) -> pd.DataFrame:
    """Subsample a DataFrame with optional stratification by label column."""
    if label_col is None:
        return df.sample(n=min(n_samples, len(df)), random_state=random_state)
    per_class = n_samples // df[label_col].nunique()
    _key = "__groupby_col__"
    df = df.copy()
    df[_key] = df[label_col]
    return (
        df.groupby(_key, group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), per_class), random_state=random_state))
        .drop(columns=[_key])
        .reset_index(drop=True)
    )


def random_undersample_df(
    df: pd.DataFrame, label_col: str, random_state: int | None = None
) -> pd.DataFrame:
    """Undersample to balance classes."""
    min_count = df[label_col].value_counts().min()
    _key = "__groupby_col__"
    df = df.copy()
    df[_key] = df[label_col]
    return (
        df.groupby(_key, group_keys=False)
        .apply(lambda g: g.sample(n=min_count, random_state=random_state))
        .reset_index(drop=True)
    )


def random_oversample_df(
    df: pd.DataFrame, label_col: str, random_state: int | None = None
) -> pd.DataFrame:
    """Oversample to balance classes."""
    max_count = df[label_col].value_counts().max()

    def oversample(g):
        n = max_count - len(g)
        if n > 0:
            g = pd.concat([g, g.sample(n=n, replace=True, random_state=random_state)])
        return g

    _key = "__groupby_col__"
    df = df.copy()
    df[_key] = df[label_col]
    return df.groupby(_key, group_keys=False).apply(oversample).reset_index(drop=True)


def ml_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int | None = None,
    label_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train, validation, and test sets with optional stratification."""
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac, val_frac, and test_frac must sum to 1.0.")

    stratify = df[label_col] if label_col else None
    train_df, rest = train_test_split(
        df, train_size=train_frac, random_state=random_state, stratify=stratify
    )

    stratify_rest = rest[label_col] if label_col else None
    val_df, test_df = train_test_split(
        rest,
        train_size=val_frac / (val_frac + test_frac),
        random_state=random_state,
        stratify=stratify_rest,
    )
    return train_df, val_df, test_df


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numerical features to specified quantiles to reduce outlier impact."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.lower_bounds_ = X.quantile(self.lower_quantile)
        self.upper_bounds_ = X.quantile(self.upper_quantile)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(X).clip(
            lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1
        )


class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p transformation to handle skewed data with zeros."""

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(np.maximum(X, 0) + self.epsilon)


class TopNCategoryEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features by mapping top-N frequent categories to integers."""

    def __init__(self, top_n: int = 256, default_value: int = 0):
        self.top_n = top_n
        self.default_value = default_value

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.category_maps_ = {
            col: {
                cat: i + 1
                for i, cat in enumerate(
                    X[col].value_counts().nlargest(self.top_n - 1).index
                )
            }
            for col in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = (
                X[col]
                .map(self.category_maps_.get(col, {}))
                .fillna(self.default_value)
                .astype(int)
            )
        return X


class TopNHashEncoder(BaseEstimator, TransformerMixin):
    """Hybrid categorical encoder: top-N categories + hash buckets for rare/OOV values.

    Encoding scheme:
      0              — missing / NaN
      1 … top_n      — top-N most frequent categories
      top_n+1 …      — hash buckets for rare/OOV categories
    """

    def __init__(
        self,
        top_n: int = 256,
        hash_buckets: int = 1024,
        missing_token: int = 0,
        hash_key: str = "cat-encoder-v1",
        dtype: type = np.int32,
    ):
        self.top_n = top_n
        self.hash_buckets = hash_buckets
        self.missing_token = missing_token
        self.hash_key = hash_key
        self.dtype = dtype

    def _hash_bucket(self, col: str, value, n: int) -> int:
        s = "NA" if pd.isna(value) else str(value)
        digest = hashlib.blake2b(
            f"{self.hash_key}|{col}|{s}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, byteorder="little") % n

    def fit(self, X: pd.DataFrame, y=None):
        if self.top_n < 0 or self.hash_buckets < 0:
            raise ValueError("top_n and hash_buckets must be non-negative.")
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.category_maps_ = {
            col: {
                cat: i + 1
                for i, cat in enumerate(
                    X[col].value_counts(dropna=True).nlargest(self.top_n).index
                )
            }
            for col in self.columns_
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        hashed_start = 1 + self.top_n
        out = {}
        for col in (c for c in self.columns_ if c in X.columns):
            cmap = self.category_maps_[col]
            ids = []
            for val in X[col]:
                if pd.isna(val):
                    ids.append(self.missing_token)
                elif val in cmap:
                    ids.append(cmap[val])
                elif self.hash_buckets > 0:
                    ids.append(
                        hashed_start + self._hash_bucket(col, val, self.hash_buckets)
                    )
                else:
                    ids.append(self.missing_token)
            out[col] = np.asarray(ids, dtype=self.dtype)
        return pd.DataFrame(out, index=X.index)


def encode_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    src_label_col: str,
    dst_label_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Encode string labels to integers using a fitted LabelEncoder.

    Returns: (train_df, val_df, test_df, label_mapping)
    """
    le = LabelEncoder()
    dst = dst_label_col or f"encoded_{src_label_col}"
    train_df[dst] = le.fit_transform(train_df[src_label_col])
    val_df[dst] = le.transform(val_df[src_label_col])
    test_df[dst] = le.transform(test_df[src_label_col])
    label_mapping = {int(i): str(name) for i, name in enumerate(le.classes_)}
    return train_df, val_df, test_df, label_mapping
