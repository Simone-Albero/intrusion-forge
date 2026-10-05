import numpy as np
import pandas as pd

from src.engine.ml.preprocessing import build_regressor_pipeline


def fit_sample_regressor(
    X: pd.DataFrame,
    failure: np.ndarray,
    region: np.ndarray,
    *,
    fold_of_region: pd.Series,
    name: str,
    params: dict,
    max_rows: int,
    random_state: int,
) -> np.ndarray:
    """Each row's chance of failure, from a fit on other folds' regions only."""
    fold_of_row = fold_of_region.loc[region].to_numpy()
    rng = np.random.default_rng(random_state)
    predicted = np.empty(len(X))
    for fold in np.unique(fold_of_row):
        held_out = fold_of_row == fold
        fit_rows = np.flatnonzero(~held_out)
        if len(fit_rows) > max_rows:
            fit_rows = np.sort(rng.choice(fit_rows, size=max_rows, replace=False))
        model = build_regressor_pipeline(name, {**params, "random_state": random_state})
        model.fit(X.iloc[fit_rows], failure[fit_rows])
        predicted[held_out] = np.clip(model.predict(X[held_out]), 0.0, 1.0)
    return predicted
