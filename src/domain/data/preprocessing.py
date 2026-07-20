import hashlib

import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler


def drop_nans(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop rows with NaN or infinite values in specified columns."""
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)


def query_filter(df: pd.DataFrame, *, query: str | None = None) -> pd.DataFrame:
    """Filter DataFrame using a query string."""
    return df.query(query) if query else df


def rare_category_filter(
    df: pd.DataFrame, cat_cols: list[str], *, min_count: int = 3000
) -> pd.DataFrame:
    """Remove rows with rare categories in specified categorical columns."""
    if not min_count or min_count <= 0:
        return df
    df = df.copy()
    for col in cat_cols:
        counts = df[col].value_counts()
        df = df[~df[col].isin(counts[counts < min_count].index)]
    return df


def _stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    per_group: int,
    *,
    random_state: int | None = None,
) -> pd.DataFrame:
    return (
        df.groupby(df[label_col].values, group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), per_group), random_state=random_state))
        .reset_index(drop=True)
    )


def subsample_df(
    df: pd.DataFrame,
    n_samples: int,
    *,
    random_state: int | None = None,
    label_col: str | None = None,
) -> pd.DataFrame:
    """Subsample a DataFrame with optional stratification by label column."""
    if label_col is None:
        return df.sample(n=min(n_samples, len(df)), random_state=random_state)
    per_class = n_samples // df[label_col].nunique()
    return _stratified_sample(df, label_col, per_class, random_state=random_state)


def random_undersample_df(
    df: pd.DataFrame, label_col: str, *, random_state: int | None = None
) -> pd.DataFrame:
    """Undersample to balance classes."""
    min_count = df[label_col].value_counts().min()
    return _stratified_sample(df, label_col, min_count, random_state=random_state)


def ml_split(
    df: pd.DataFrame,
    *,
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


class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p transformation to handle skewed data with zeros."""

    def __init__(self, *, epsilon: float = 1e-10):
        self.epsilon = epsilon

    def fit(self, X, *, y=None):
        return self

    def transform(self, X):
        return np.log1p(np.maximum(X, 0) + self.epsilon)


class TopNHashEncoder(BaseEstimator, TransformerMixin):
    """Hybrid categorical encoder: top-N categories + hash buckets for rare/OOV values.

    Encoding scheme:
      0              — missing / NaN
      1 … top_n      — top-N most frequent categories
      top_n+1 …      — hash buckets for rare/OOV categories

    With hash_buckets=0 there are no OOV slots, so rare/OOV values fold into the
    missing token (0).
    """

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
        s = "NA" if pd.isna(value) else str(value)
        digest = hashlib.blake2b(
            f"{self.hash_key}|{col}|{s}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, byteorder="little") % n

    def fit(self, X: pd.DataFrame, *, y=None):
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
            s = X[col]
            ids = s.map(cmap)  # top-N -> id, NaN for missing/OOV
            if self.hash_buckets > 0:
                oov = ids.isna() & s.notna()
                # hash only the distinct OOV values, then broadcast
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
    src_label_col: str,
    *,
    dst_label_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Encode string labels to integers using a LabelEncoder fitted on train."""
    le = LabelEncoder()
    dst = dst_label_col or f"encoded_{src_label_col}"
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[dst] = le.fit_transform(train_df[src_label_col])
    val_df[dst] = le.transform(val_df[src_label_col])
    test_df[dst] = le.transform(test_df[src_label_col])
    label_mapping = {int(i): str(name) for i, name in enumerate(le.classes_)}
    return train_df, val_df, test_df, label_mapping


def build_preprocessor(
    *,
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    num_steps: list[tuple[str, BaseEstimator]] | None = None,
    cat_steps: list[tuple[str, BaseEstimator]] | None = None,
    remainder: str = "passthrough",
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
        remainder=remainder,
        verbose_feature_names_out=False,
    )


def cluster_feature_columns(
    cluster_features: dict[str, dict[str, float | None]],
    *,
    exclude: tuple[str, ...] = ("cluster_class",),
) -> list[str]:
    """Sorted union of measure names across cluster rows, minus `exclude`."""
    names: set[str] = set()
    for row in cluster_features.values():
        names.update(row.keys())
    return sorted(names - set(exclude))


def attach_cluster_features(
    df: pd.DataFrame,
    cluster_features: dict[str, dict[str, float | None]],
    *,
    cluster_col: str = "cluster",
    exclude: tuple[str, ...] = ("cluster_class",),
) -> pd.DataFrame:
    """Left-join per-cluster complexity rows onto `df` by `cluster_col`.

    Target columns already present are dropped first (idempotent), `exclude` keys
    are never attached (leakage-prone), and unmatched rows get NaN.
    """
    columns = cluster_feature_columns(cluster_features, exclude=exclude)
    feature_df = pd.DataFrame.from_dict(cluster_features, orient="index").reindex(
        columns=columns
    )
    feature_df.index = feature_df.index.astype(df[cluster_col].dtype)
    df = df.drop(columns=[c for c in columns if c in df.columns])
    return df.merge(feature_df, left_on=cluster_col, right_index=True, how="left")


def scale_columns_on_train(
    splits: dict[str, pd.DataFrame],
    columns: list[str],
    *,
    train_key: str = "train",
) -> dict[str, pd.DataFrame]:
    """Median-impute then RobustScale `columns`, fitting both on the train split.

    Residual NaN are median-filled first (RobustScaler cannot fit on NaN); stats
    come only from train, so val/test are transformed without leakage.
    """
    if not columns:
        return splits
    train = splits[train_key][columns]
    medians = train.median()
    scaler = RobustScaler().fit(train.fillna(medians))
    scaled: dict[str, pd.DataFrame] = {}
    for name, df in splits.items():
        df = df.copy()
        df[columns] = scaler.transform(df[columns].fillna(medians))
        scaled[name] = df
    return scaled
