from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .model import MLClassifierFactory

_HISTGB_MAX_CARDINALITY = 255

CLASSIFIER_PREPROCESS: dict[str, str] = {
    "logistic_regression": "onehot",
    "lda": "onehot",
    "linear_svc": "onehot",
    "knn": "onehot",
    "decision_tree": "passthrough",
    "random_forest": "passthrough",
    "hist_gradient_boosting": "native_sklearn",
    "xgboost": "native_xgb",
    "naive_bayes": "drop_cat",
}


class CappedCategoryEncoder(BaseEstimator, TransformerMixin):
    """Cast columns to pandas Categorical, keeping only the top-`max_cardinality` values."""

    def __init__(self, max_cardinality: int | None = 255):
        self.max_cardinality = max_cardinality

    def fit(self, X: pd.DataFrame, y=None) -> "CappedCategoryEncoder":
        """Learn the retained categories of every column."""
        self.categories_: dict[str, pd.Index] = {}
        for col in X.columns:
            counts = X[col].value_counts()
            keep = (
                len(counts)
                if self.max_cardinality is None
                else min(self.max_cardinality, len(counts))
            )
            self.categories_[col] = counts.nlargest(keep).index
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cast every column to its learned category set, mapping the rest to NaN."""
        result = X.copy()
        for col in result.columns:
            cats = self.categories_[col]
            result[col] = pd.Categorical(
                result[col].where(result[col].isin(cats)), categories=cats
            )
        return result

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Output feature names, unchanged from the input."""
        return self.feature_names_in_.copy()


_CAT_ENCODERS: dict[str, Callable[[], TransformerMixin] | None] = {
    "drop_cat": None,
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "native_sklearn": lambda: CappedCategoryEncoder(
        max_cardinality=_HISTGB_MAX_CARDINALITY
    ),
    "native_xgb": lambda: CappedCategoryEncoder(max_cardinality=None),
}
# native_* strategies feed a classifier that reads pandas categorical dtype directly, so
# their ColumnTransformer keeps plain column names and emits a DataFrame.
_NATIVE_CATEGORICAL_STRATEGIES = ("native_sklearn", "native_xgb")


def _build_preprocess(
    strategy: str,
    num_cols: list[str],
    cat_cols: list[str],
) -> ColumnTransformer | str:
    """Build the preprocessor a strategy calls for."""
    if strategy == "passthrough":
        return "passthrough"
    if strategy not in _CAT_ENCODERS:
        raise ValueError(f"Unknown preprocessing strategy: {strategy!r}")

    transformers = [("num", "passthrough", num_cols)]
    encoder_factory = _CAT_ENCODERS[strategy]
    if encoder_factory is not None:
        transformers.append(("cat", encoder_factory(), cat_cols))

    native = strategy in _NATIVE_CATEGORICAL_STRATEGIES
    pre = ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=not native
    )
    return pre.set_output(transform="pandas") if native else pre


def _augment_params_for_strategy(strategy: str, params: dict) -> dict:
    """Add the classifier params a native-categorical strategy requires."""
    params = dict(params)
    if strategy == "native_sklearn":
        params.setdefault("categorical_features", "from_dtype")
    elif strategy == "native_xgb":
        params.setdefault("enable_categorical", True)
        params.setdefault("tree_method", "hist")
    return params


def build_pipeline(
    name: str,
    params: dict,
    num_cols: list[str],
    cat_cols: list[str],
) -> Pipeline:
    """Build a preprocess-plus-classifier Pipeline, per the `CLASSIFIER_PREPROCESS` table."""
    strategy = CLASSIFIER_PREPROCESS.get(name, "passthrough")
    pre = _build_preprocess(strategy, num_cols, cat_cols)
    full_params = _augment_params_for_strategy(strategy, params)
    clf = MLClassifierFactory.create(name, full_params)
    return Pipeline([("pre", pre), ("clf", clf)])
