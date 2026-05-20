from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from .model import MLClassifierFactory

"""Each classifier name maps to one preprocessing strategy that determines how
categorical columns are fed to the estimator:

  - "onehot":         num passthrough + OneHotEncoder on cat (linear/distance models).
  - "passthrough":    integer-encoded cat passed as-is (tree models).
  - "native_sklearn": cat dtype + classifier `categorical_features` flag (HistGB).
  - "native_xgb":     cat dtype + classifier `enable_categorical=True` (XGBoost).
  - "drop_cat":       cat columns dropped, only num reach the estimator (GaussianNB).

`build_pipeline(name, params, num_cols, cat_cols)` returns a fitted-ready
sklearn `Pipeline` wrapping the appropriate preprocessing step plus the
classifier built via `MLClassifierFactory`.
"""

CLASSIFIER_PREPROCESS: dict[str, str] = {
    "logistic_regression": "onehot",
    "lda": "onehot",
    "svm_rbf": "onehot",
    "knn": "onehot",
    "decision_tree": "passthrough",
    "random_forest": "passthrough",
    "hist_gradient_boosting": "native_sklearn",
    "xgboost": "native_xgb",
    "naive_bayes": "drop_cat",
}


def _to_category_dtype(X):
    """Cast every column to pandas Categorical dtype (used by XGBoost native mode)."""
    return X.astype("category")


def _build_preprocess(
    strategy: str,
    num_cols: list[str],
    cat_cols: list[str],
):
    """Return the sklearn step that prepares features for the given strategy.

    For "passthrough" returns the string ``"passthrough"`` (sklearn convention).
    For "native_*" strategies returns a column-wise cast to ``pd.Categorical``,
    which makes both HistGB (`categorical_features="from_dtype"`) and XGBoost
    (`enable_categorical=True`) detect categorical features automatically.
    """
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
    if strategy in ("native_sklearn", "native_xgb"):
        # Cast only cat columns to pandas Categorical so the classifier picks
        # them up via dtype detection. Num columns are kept verbatim.
        return ColumnTransformer(
            [
                ("num", "passthrough", num_cols),
                (
                    "cat",
                    FunctionTransformer(_to_category_dtype, validate=False),
                    cat_cols,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
    raise ValueError(f"Unknown preprocessing strategy: {strategy!r}")


def _augment_params_for_strategy(
    strategy: str,
    params: dict,
    num_cols: list[str],
    cat_cols: list[str],
) -> dict:
    """Add classifier kwargs required by the native strategies."""
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
    """Build a sklearn `Pipeline([("pre", <preprocessor>), ("clf", <classifier>)])`.

    The preprocessing step is chosen via the `CLASSIFIER_PREPROCESS` table.
    """
    strategy = CLASSIFIER_PREPROCESS.get(name, "passthrough")
    pre = _build_preprocess(strategy, num_cols, cat_cols)
    full_params = _augment_params_for_strategy(strategy, params, num_cols, cat_cols)
    clf = MLClassifierFactory.create(name, full_params)
    return Pipeline([("pre", pre), ("clf", clf)])
