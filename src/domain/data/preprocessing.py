import hashlib

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


def drop_nans(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop rows with NaN or infinite values in specified columns."""
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)


def query_filter(df: pd.DataFrame, *, query: str | None) -> pd.DataFrame:
    """Filter DataFrame using a query string."""
    return df.query(query) if query else df


def drop_rare_classes(
    df: pd.DataFrame, label_col: str, *, min_count: int
) -> pd.DataFrame:
    """Remove the rows of every class with fewer than `min_count` rows."""
    counts = df[label_col].value_counts()
    return df[~df[label_col].isin(counts[counts < min_count].index)]


def _stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    per_group: int,
    *,
    random_state: int,
) -> pd.DataFrame:
    """Sample up to `per_group` rows from every label group, keeping their index."""
    return df.groupby(df[label_col].values, group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), per_group), random_state=random_state)
    )


def subsample_df(
    df: pd.DataFrame, n_samples: int, *, random_state: int, label_col: str
) -> pd.DataFrame:
    """Up to `n_samples // n_classes` rows from every class."""
    per_class = n_samples // df[label_col].nunique()
    return _stratified_sample(df, label_col, per_class, random_state=random_state)


def random_undersample_df(
    df: pd.DataFrame, label_col: str, *, random_state: int
) -> pd.DataFrame:
    """Undersample to balance classes."""
    min_count = df[label_col].value_counts().min()
    return _stratified_sample(df, label_col, min_count, random_state=random_state)


def ml_split(
    df: pd.DataFrame,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_state: int,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train, validation and test sets, stratified by label."""
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac, val_frac, and test_frac must sum to 1.0.")

    train_df, rest = train_test_split(
        df, train_size=train_frac, random_state=random_state, stratify=df[label_col]
    )
    val_df, test_df = train_test_split(
        rest,
        train_size=val_frac / (val_frac + test_frac),
        random_state=random_state,
        stratify=rest[label_col],
    )
    return train_df, val_df, test_df


class LogTransformer(BaseEstimator, TransformerMixin):
    """Signed log1p: compresses a skewed column and keeps the order of its negative values."""

    def fit(self, X, *, y=None) -> "LogTransformer":
        """Stateless fit."""
        return self

    def transform(self, X):
        """Apply sign(x) * log1p(|x|)."""
        return np.sign(X) * np.log1p(np.abs(X))


class TopNHashEncoder(BaseEstimator, TransformerMixin):
    """Categorical encoder: 0 for missing, 1…top_n for frequent categories, then hash buckets."""

    def __init__(
        self,
        *,
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
        """Stable bucket index of a category value."""
        s = "NA" if pd.isna(value) else str(value)
        digest = hashlib.blake2b(
            f"{self.hash_key}|{col}|{s}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, byteorder="little") % n

    def fit(self, X: pd.DataFrame, *, y=None) -> "TopNHashEncoder":
        """Learn the top-N categories of every column."""
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
        """Encode every column into category ids, hash buckets or the missing token."""
        X = pd.DataFrame(X)
        hashed_start = 1 + self.top_n
        out = {}
        for col in (c for c in self.columns_ if c in X.columns):
            cmap = self.category_maps_[col]
            s = X[col]
            ids = s.map(cmap)
            if self.hash_buckets > 0:
                oov = ids.isna() & s.notna()
                hash_map = {
                    v: hashed_start + self._hash_bucket(col, v, self.hash_buckets)
                    for v in s[oov].unique()
                }
                ids = ids.where(~oov, s.map(hash_map))
            ids = ids.fillna(self.missing_token)
            out[col] = ids.to_numpy(dtype=self.dtype)
        return pd.DataFrame(out, index=X.index)


def encode_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    src_label_col: str,
    dst_label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Encode string labels to integers using a LabelEncoder fitted on train."""
    le = LabelEncoder()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[dst_label_col] = le.fit_transform(train_df[src_label_col])
    val_df[dst_label_col] = le.transform(val_df[src_label_col])
    test_df[dst_label_col] = le.transform(test_df[src_label_col])
    label_mapping = {int(i): str(name) for i, name in enumerate(le.classes_)}
    return train_df, val_df, test_df, label_mapping


def build_preprocessor(
    *,
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    num_steps: list[tuple[str, BaseEstimator]] | None = None,
    cat_steps: list[tuple[str, BaseEstimator]] | None = None,
) -> ColumnTransformer:
    """Assemble a ColumnTransformer from per-type (name, transformer) steps."""
    set_config(transform_output="pandas")
    transformers = []
    if num_cols and num_steps:
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols and cat_steps:
        transformers.append(("cat", Pipeline(cat_steps), cat_cols))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
