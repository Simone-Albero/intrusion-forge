from typing import Optional, Tuple
import hashlib

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


def drop_nans(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)


def query_filter(df: pd.DataFrame, query: Optional[str] = None) -> pd.DataFrame:
    if query:
        return df.query(query)
    return df


def rare_category_filter(
    df: pd.DataFrame, cat_cols: list, min_count: int = 3000
) -> pd.DataFrame:
    df = df.copy()

    if min_count is None or min_count <= 0:
        return df

    for col in cat_cols:
        value_counts = df[col].value_counts()
        rare_categories = value_counts[value_counts < min_count].index
        df = df[~df[col].isin(rare_categories)]
    return df


def subsample_df(
    df: pd.DataFrame,
    n_samples: int,
    random_state: Optional[int] = None,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    if label_col is None:
        if n_samples > len(df):
            return df.sample(n=len(df), random_state=random_state)
        return df.sample(n=n_samples, random_state=random_state)
    else:
        return (
            df.groupby(label_col, group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=min(len(x), n_samples // df[label_col].nunique()),
                    random_state=random_state,
                )
            )
            .reset_index(drop=True)
        )


def random_undersample_df(
    df: pd.DataFrame, label_col: str, random_state: Optional[int] = None
) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    min_count = counts.min()
    sampled_groups = []
    for label, group in df.groupby(label_col):
        sampled = group.sample(n=min_count, random_state=random_state)
        sampled_groups.append(sampled)
    return pd.concat(sampled_groups, ignore_index=True)


def random_oversample_df(
    df: pd.DataFrame, label_col: str, random_state: Optional[int] = None
) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    max_count = counts.max()
    sampled_groups = []
    for label, group in df.groupby(label_col):
        n_samples = max_count - len(group)
        if n_samples > 0:
            sampled = group.sample(n=n_samples, replace=True, random_state=random_state)
            group = pd.concat([group, sampled], ignore_index=True)
        sampled_groups.append(group)
    return pd.concat(sampled_groups, ignore_index=True)


def ml_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: Optional[int] = None,
    label_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac, val_frac, and test_frac must sum to 1.0")

    train_df, remaining_df = train_test_split(
        df, train_size=train_frac, random_state=random_state, stratify=df[label_col]
    )

    val_size = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        remaining_df,
        train_size=val_size,
        random_state=random_state,
        stratify=remaining_df[label_col],
    )

    return train_df, val_df, test_df


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        X = pd.DataFrame(X).copy()
        self.lower_bounds_ = X.quantile(self.lower_quantile)
        self.upper_bounds_ = X.quantile(self.upper_quantile)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        return X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)


class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p transformation to handle skewed data with zeros."""

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        return np.log1p(np.maximum(X_transformed, 0) + self.epsilon)

    def inverse_transform(self, X):
        return np.expm1(X) - self.epsilon


class TopNCategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, top_n: int = 256, default_value: int = 0):
        self.top_n = top_n
        self.default_value = default_value

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        X = pd.DataFrame(X).copy()
        self.category_maps_ = {}
        n_categories = self.top_n - 1  # Reserve 0 for default
        for col in X.columns:
            top_categories = X[col].value_counts().nlargest(n_categories).index.tolist()

            category_map = {
                category: idx + 1 for idx, category in enumerate(top_categories)
            }
            self.category_maps_[col] = category_map
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            category_map = self.category_maps_.get(col, {})
            X[col] = X[col].map(category_map).fillna(self.default_value).astype(int)
        return X


class TopNHashEncoder(BaseEstimator, TransformerMixin):
    """
    Hybrid categorical encoder with top-N frequent categories and hash buckets for rare values.

    Encoding scheme:
      - 0: Missing/NaN values (padding_idx compatible)
      - 1 to top_n: Top-N most frequent categories
      - (top_n + 1) onwards: Hashed buckets for rare/OOV categories

    Optional features:
      - log_freq: log1p of category frequency from training
      - is_unk: binary flag indicating unknown/hashed category
    """

    def __init__(
        self,
        top_n: int = 256,
        hash_buckets: int = 1024,
        add_log_freq: bool = True,
        add_is_unk: bool = True,
        missing_token: int = 0,  # Changed from 1 to 0
        hash_key: str = "cat-encoder-v1",
        dtype: type = np.int32,
    ):
        self.top_n = top_n
        self.hash_buckets = hash_buckets
        self.add_log_freq = add_log_freq
        self.add_is_unk = add_is_unk
        self.missing_token = missing_token
        self.hash_key = hash_key
        self.dtype = dtype

    def _stable_hash_bucket(self, col: str, value, num_buckets: int) -> int:
        """Deterministic hash bucket [0, num_buckets-1] using column-specific hashing."""
        value_str = "NA" if pd.isna(value) else str(value)
        payload = f"{self.hash_key}|{col}|{value_str}".encode("utf-8")
        hash_digest = hashlib.blake2b(payload, digest_size=8).digest()
        return (
            int.from_bytes(hash_digest, byteorder="little", signed=False) % num_buckets
        )

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        if self.top_n < 0 or self.hash_buckets < 0:
            raise ValueError("top_n and hash_buckets must be non-negative")

        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.category_maps_ = {}
        self.freq_maps_ = {}

        for col in self.columns_:
            counts = X[col].value_counts(dropna=True)
            self.freq_maps_[col] = counts.to_dict()

            # Map top-N categories to contiguous IDs starting at 1 (0 reserved for missing)
            top_categories = counts.nlargest(self.top_n).index if self.top_n > 0 else []
            self.category_maps_[col] = {
                cat: 1 + idx  # Start from 1, reserve 0 for missing
                for idx, cat in enumerate(top_categories)
            }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        cols = [c for c in self.columns_ if c in X.columns]
        hashed_start = 1 + self.top_n  # Hash buckets start after top-N categories
        out = {}

        for col in cols:
            category_map = self.category_maps_[col]
            freq_map = self.freq_maps_[col]
            values = X[col].values

            # Encode IDs and track unknown categories
            ids = []
            is_unk = []

            for val in values:
                if pd.isna(val):
                    ids.append(self.missing_token)  # 0 for missing
                    is_unk.append(1)
                elif val in category_map:
                    ids.append(category_map[val])  # 1 to top_n
                    is_unk.append(0)
                else:
                    # Unknown category: use hash bucket or default to missing
                    if self.hash_buckets > 0:
                        bucket = self._stable_hash_bucket(col, val, self.hash_buckets)
                        ids.append(hashed_start + bucket)  # top_n+1 onwards
                    else:
                        ids.append(self.missing_token)
                    is_unk.append(1)

            out[col] = np.asarray(ids, dtype=self.dtype)

            if self.add_is_unk:
                out[f"{col}__is_unk"] = np.asarray(is_unk, dtype=np.int8)

            if self.add_log_freq:
                log_freqs = [
                    np.log1p(1.0) if pd.isna(val) else np.log1p(freq_map.get(val, 1.0))
                    for val in values
                ]
                out[f"{col}__log_freq"] = np.asarray(log_freqs, dtype=np.float32)

        return pd.DataFrame(out, index=X.index)
