import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .model import MLClassifierFactory

# HistGradientBoosting raises above this native-categorical cardinality
_HISTGB_MAX_CARDINALITY = 255

CLASSIFIER_PREPROCESS: dict[str, str] = {
    "logistic_regression": "onehot",
    "lda": "onehot",
    "svm_rbf": "onehot",
    "linear_svc": "onehot",
    "knn": "onehot",
    "decision_tree": "passthrough",
    "random_forest": "passthrough",
    "hist_gradient_boosting": "native_sklearn",
    "xgboost": "native_xgb",
    "naive_bayes": "drop_cat",
}


class CappedCategoryEncoder(BaseEstimator, TransformerMixin):
    """Cast columns to pandas Categorical, keeping only the top-`max_cardinality` training-fold values (rest become NaN) so HistGB's 255-category limit is never exceeded."""

    def __init__(self, max_cardinality: int | None = 255):
        self.max_cardinality = max_cardinality

    def fit(self, X: pd.DataFrame, y=None):
        self.categories_: dict[str, pd.Index] = {}
        for col in X.columns:
            counts = X[col].value_counts()
            keep = len(counts) if self.max_cardinality is None else min(self.max_cardinality, len(counts))
            self.categories_[col] = counts.nlargest(keep).index
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        for col in result.columns:
            cats = self.categories_[col]
            result[col] = pd.Categorical(
                result[col].where(result[col].isin(cats)), categories=cats
            )
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_.copy()



def _build_preprocess(
    strategy: str,
    num_cols: list[str],
    cat_cols: list[str],
):
    if strategy == "passthrough":
        return "passthrough"
    if strategy == "drop_cat":
        return ColumnTransformer([("num", "passthrough", num_cols)], remainder="drop")
    if strategy == "onehot":
        return ColumnTransformer(
            [
                ("num", "passthrough", num_cols),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )
    if strategy == "native_sklearn":
        return ColumnTransformer(
            [
                ("num", "passthrough", num_cols),
                ("cat", CappedCategoryEncoder(max_cardinality=_HISTGB_MAX_CARDINALITY), cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
    if strategy == "native_xgb":
        # max_cardinality=None maps unseen test values to NaN; XGBoost raises on
        # out-of-set categories otherwise.
        return ColumnTransformer(
            [
                ("num", "passthrough", num_cols),
                ("cat", CappedCategoryEncoder(max_cardinality=None), cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
    raise ValueError(f"Unknown preprocessing strategy: {strategy!r}")


def _augment_params_for_strategy(strategy: str, params: dict) -> dict:
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
    """Build a `Pipeline([("pre", ...), ("clf", ...)])`, choosing the preprocessor via the `CLASSIFIER_PREPROCESS` table."""
    strategy = CLASSIFIER_PREPROCESS.get(name, "passthrough")
    pre = _build_preprocess(strategy, num_cols, cat_cols)
    full_params = _augment_params_for_strategy(strategy, params)
    clf = MLClassifierFactory.create(name, full_params)
    return Pipeline([("pre", pre), ("clf", clf)])
