from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.core.utils import load_from_joblib, save_to_joblib
from src.engine.ml.preprocessing import build_pipeline


def _check_context(context: dict | None) -> tuple[list[str], list[str]]:
    if context is None:
        raise ValueError(
            "ML training requires `context` with `num_cols` and `cat_cols`."
        )
    return context["num_cols"], context["cat_cols"]


def fit_classifier(
    name: str,
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    context: dict | None = None,
) -> tuple[Pipeline, dict]:
    """Build an sklearn pipeline (preprocess + classifier) and fit on (X, y).

    `X` is a DataFrame whose columns include both numerical and categorical
    features. The categorical preprocessing strategy is chosen per classifier
    in `CLASSIFIER_PREPROCESS`.

    `X_val`/`y_val` are accepted for interface parity with DL but ignored.

    Returns ``(pipeline, {})``.
    """
    num_cols, cat_cols = _check_context(context)
    pipeline = build_pipeline(name, params, num_cols, cat_cols)
    pipeline.fit(X, y)
    return pipeline, {}


def grid_search_classifier(
    name: str,
    params: dict,
    grid: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    scoring: str = "f1_macro",
    cv: int = 5,
    n_jobs: int = -1,
    context: dict | None = None,
) -> tuple[Pipeline, dict]:
    """Cross-validated grid search over the classifier step of the pipeline.

    Grid keys (e.g. ``n_estimators``) are remapped to ``clf__<key>`` to target
    the classifier inside the Pipeline. Returns ``(best_pipeline, summary)``.
    """
    num_cols, cat_cols = _check_context(context)
    base = build_pipeline(name, params, num_cols, cat_cols)
    clf_grid = {f"clf__{k}": v for k, v in grid.items()}

    search = GridSearchCV(
        base,
        param_grid=clf_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y)

    cv_results = [
        {
            "params": {k.replace("clf__", "", 1): v for k, v in p.items()},
            "mean_test_score": float(s),
            "std_test_score": float(std),
        }
        for p, s, std in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        )
    ]
    summary = {
        "best_params": {
            k.replace("clf__", "", 1): v for k, v in search.best_params_.items()
        },
        "best_score": float(search.best_score_),
        "scoring": scoring,
        "cv": cv,
        "cv_results": cv_results,
    }
    return search.best_estimator_, summary


def predict_with_proba(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    context: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(predicted labels, class probability matrix)`` for the pipeline."""
    return pipeline.predict(X), pipeline.predict_proba(X)


def save_model(
    pipeline: Pipeline,
    path: Path,
    *,
    name: str = "",
    params: dict | None = None,
) -> None:
    """Save the full sklearn Pipeline to ``path / 'model.joblib'``."""
    save_to_joblib(pipeline, Path(path) / "model.joblib")


def load_model(path: Path, *, context: dict | None = None) -> Pipeline:
    """Load the sklearn Pipeline from ``path / 'model.joblib'``."""
    return load_from_joblib(Path(path) / "model.joblib")
