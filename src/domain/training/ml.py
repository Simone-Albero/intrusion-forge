import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.core.utils import load_from_joblib, save_to_joblib
from src.engine.ml.preprocessing import build_pipeline


def _strip_clf_prefix(params: dict) -> dict:
    """Drop the pipeline step prefix from grid-search parameter names."""
    return {k.replace("clf__", "", 1): v for k, v in params.items()}


class MLTrainer:
    """Fits scikit-learn pipelines of preprocessing plus classifier."""

    def __init__(self, num_cols: list[str], cat_cols: list[str]) -> None:
        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def features(self, df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
        """The feature slice the sklearn pipeline consumes."""
        return df[feat_cols]

    def prepare(
        self, df: pd.DataFrame, feat_cols: list[str], label_col: str
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """A feature slice plus the label array."""
        return self.features(df, feat_cols), df[label_col].to_numpy()

    def fit(
        self,
        name: str,
        params: dict,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        X_val: pd.DataFrame,
        save_dir: Path,
    ) -> tuple[Pipeline, dict]:
        """Build an sklearn pipeline of preprocessing plus classifier and fit it on (X, y)."""
        pipeline = build_pipeline(name, params, self.num_cols, self.cat_cols)
        pipeline.fit(X, y)
        return pipeline, {}

    def grid_search(
        self,
        name: str,
        params: dict,
        grid: dict,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        scoring: str = "f1_macro",
        cv: int = 5,
        max_samples: int | None = None,
        random_state: int = 42,
    ) -> tuple[Pipeline, dict]:
        """Cross-validated grid search over the classifier step, refitting the winner on all data."""
        clf_grid = {f"clf__{k}": v for k, v in grid.items()}

        subsampled = max_samples is not None and len(X) > max_samples
        if subsampled:
            _, X_search, _, y_search = train_test_split(
                X, y, test_size=max_samples, stratify=y, random_state=random_state
            )
        else:
            X_search, y_search = X, y

        with tempfile.TemporaryDirectory() as cache_dir:
            base = build_pipeline(name, params, self.num_cols, self.cat_cols)
            base.memory = cache_dir
            search = GridSearchCV(
                base,
                param_grid=clf_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=not subsampled,
            )
            search.fit(X_search, y_search)

        if subsampled:
            best_clf_params = _strip_clf_prefix(search.best_params_)
            best_pipeline = build_pipeline(
                name, {**params, **best_clf_params}, self.num_cols, self.cat_cols
            )
            best_pipeline.fit(X, y)
        else:
            best_pipeline = search.best_estimator_

        cv_results = [
            {
                "params": _strip_clf_prefix(p),
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
            "best_params": _strip_clf_prefix(search.best_params_),
            "best_score": float(search.best_score_),
            "scoring": scoring,
            "cv": cv,
            "cv_results": cv_results,
        }
        return best_pipeline, summary

    def predict(
        self, model: Pipeline, X: pd.DataFrame, *, return_embedding: bool = False
    ) -> tuple:
        """Predict a DataFrame → (y_pred, y_proba); ML pipelines have no embedding, so z is None."""
        y_pred, y_proba = model.predict(X), model.predict_proba(X)
        if return_embedding:
            return y_pred, y_proba, None
        return y_pred, y_proba

    def save(
        self,
        model: Pipeline,
        path: Path,
        *,
        name: str = "",
        params: dict | None = None,
    ) -> None:
        """Save the full sklearn Pipeline to `path / model.joblib`."""
        save_to_joblib(model, Path(path) / "model.joblib")

    def load(self, path: Path) -> Pipeline:
        """Load the sklearn Pipeline from `path / model.joblib`."""
        return load_from_joblib(Path(path) / "model.joblib")

    def has_model(self, path: Path) -> bool:
        """True when `path` holds a saved sklearn Pipeline."""
        return (Path(path) / "model.joblib").exists()
