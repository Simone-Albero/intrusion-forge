import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from .classifier import MLClassifierFactory


def fit_classifier(
    name: str,
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
) -> BaseEstimator:
    """Instantiate a classifier via factory and fit on (X, y)."""
    model = MLClassifierFactory.create(name, params)
    model.fit(X, y)
    return model


def grid_search_classifier(
    name: str,
    params: dict,
    grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    scoring: str = "f1_macro",
    cv: int = 5,
    n_jobs: int = -1,
) -> tuple[BaseEstimator, dict]:
    """Cross-validated grid search starting from a base classifier built with `params`.

    Returns (best_estimator, summary). `summary` keys: `best_params`, `best_score`,
    `scoring`, `cv`, `cv_results` (slim: param combos + mean/std test scores).
    """
    base = MLClassifierFactory.create(name, params)
    search = GridSearchCV(
        base,
        param_grid=grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y)

    cv_results = [
        {
            "params": dict(p),
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
        "best_params": dict(search.best_params_),
        "best_score": float(search.best_score_),
        "scoring": scoring,
        "cv": cv,
        "cv_results": cv_results,
    }
    return search.best_estimator_, summary


def predict_with_proba(
    model: BaseEstimator,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted labels, class probability matrix)."""
    return model.predict(X), model.predict_proba(X)
