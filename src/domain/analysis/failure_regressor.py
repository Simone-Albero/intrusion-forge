import logging
import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import (
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.core.utils import timed
from src.engine.ml.model import MLRegressorFactory
from src.engine.ml.preprocessing import REGRESSOR_PREPROCESS, build_regressor_pipeline

logger = logging.getLogger(__name__)


def _usable_n_splits(n_smallest: int, n_splits: int) -> int:
    """Largest split count up to `n_splits` the smallest group can fill, 0 below two."""
    usable = min(n_splits, n_smallest)
    return usable if usable >= 2 else 0


def _check_models(models: dict[str, dict], primary: str) -> None:
    """Raise before any fit on a model, grid key or primary the run could not finish with."""
    if primary not in models:
        raise ValueError(
            f"The primary failure regressor {primary!r} is not among the models "
            f"{sorted(models)}."
        )
    for name, spec in models.items():
        if name not in REGRESSOR_PREPROCESS:
            raise ValueError(
                f"No preprocessing for failure regressor {name!r}: "
                f"add it to REGRESSOR_PREPROCESS ({sorted(REGRESSOR_PREPROCESS)})."
            )
        accepted = build_regressor_pipeline(name, spec["params"]).get_params()
        unknown = [k for k in spec["param_grid"] if f"model__{k}" not in accepted]
        if unknown:
            raise TypeError(f"Failure regressor {name!r} takes no parameter {unknown}.")
    if not hasattr(MLRegressorFactory.get(primary), "feature_importances_"):
        raise TypeError(
            f"The primary failure regressor {primary!r} has no feature_importances_."
        )


def _fit_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    name: str,
    params: dict,
    param_grid: dict,
    inner_cv: KFold | None,
    random_state: int,
    n_iter: int,
    search_seed: int,
) -> tuple[Pipeline, dict, float | None]:
    """Fit one model; without inner_cv, use its defaults, no search."""
    params = {**params, "random_state": random_state}
    if inner_cv is None:
        fitted = build_regressor_pipeline(name, params)
        fitted.fit(X, y)
        model_params = fitted.named_steps["model"].get_params()
        return fitted, {k: model_params[k] for k in param_grid}, None

    search = RandomizedSearchCV(
        estimator=build_regressor_pipeline(name, params),
        param_distributions={f"model__{k}": v for k, v in param_grid.items()},
        # A grid smaller than the budget is searched whole: asking for more draws than
        # it holds makes sklearn warn on every fold and run the whole grid anyway.
        n_iter=min(n_iter, len(ParameterGrid(param_grid))),
        cv=inner_cv,
        # Scored as reported, clipped: a model is tuned for the predictor it is
        # judged as.
        scoring=make_scorer(
            lambda y_true, y_pred: r2_score(y_true, np.clip(y_pred, 0.0, 1.0))
        ),
        n_jobs=-1,
        # At a fixed random_state RandomizedSearchCV draws the same combinations
        # whatever the data, so each fit gets its own seed to search its own slice.
        random_state=search_seed,
        verbose=0,
    )
    search.fit(X, y)
    best_params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    return search.best_estimator_, best_params, float(search.best_score_)


def _fit_outer_fold(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_held_out: pd.DataFrame,
    y_held_out: pd.Series,
    *,
    name: str,
    params: dict,
    param_grid: dict,
    inner_cv: KFold | None,
    random_state: int,
    n_iter: int,
    fold: int,
) -> dict:
    """Fit one outer fold and score it on its held-out regions."""
    fitted, best_params, best_score = _fit_model(
        X_fit,
        y_fit,
        name=name,
        params=params,
        param_grid=param_grid,
        inner_cv=inner_cv,
        random_state=random_state,
        n_iter=n_iter,
        search_seed=random_state + fold,
    )

    # A rate lies in [0, 1] and a forest's mean of rates always does; the other models
    # are not bound to it.
    y_pred = np.clip(fitted.predict(X_held_out), 0.0, 1.0)
    has_variance = len(y_held_out) > 1 and np.std(y_held_out) > 0 and np.std(y_pred) > 0
    return {
        "r2": (
            float(r2_score(y_held_out, y_pred)) if len(y_held_out) > 1 else float("nan")
        ),
        "mae": float(mean_absolute_error(y_held_out, y_pred)),
        "spearman": (
            float(spearmanr(y_pred, y_held_out).statistic)
            if has_variance
            else float("nan")
        ),
        "importances": getattr(
            fitted.named_steps["model"], "feature_importances_", None
        ),
        "y_pred": y_pred.tolist(),
        "indices": X_held_out.index.tolist(),
        "best_params": best_params,
        "best_score": best_score,
    }


def _quantile_strata(y: pd.Series, q: int) -> pd.Series | None:
    """Quantile-bin codes of the continuous target, or None if it yields fewer than two bins."""
    bins = pd.qcut(y, q=min(q, len(y)), duplicates="drop")
    if bins.nunique() < 2:
        return None
    return bins.cat.codes


def join_region_summary(
    regions: pd.DataFrame, classes: pd.DataFrame, failures: pd.DataFrame
) -> pd.DataFrame:
    """One row per region: its descriptors, its class's, and the failures observed on it."""
    measures = [c for c in regions.columns if c not in ("region", "class_id")]
    class_measures = [c for c in classes.columns if c != "class_id"]
    summary = pd.DataFrame({"region": regions["region"].to_numpy()})
    for measure in measures:
        summary[f"region_{measure}"] = regions[measure].to_numpy()
    of_class = classes.set_index("class_id").loc[regions["class_id"], class_measures]
    for measure in class_measures:
        summary[f"class_{measure}"] = of_class[measure].to_numpy()
    summary["class_id"] = regions["class_id"].to_numpy()
    observed = failures.set_index("region").reindex(summary["region"])
    summary["n_eval"] = observed["n_eval"].fillna(0).astype(int).to_numpy()
    summary["failure_rate"] = observed["failure_rate"].to_numpy()
    summary["mcp_risk"] = observed["mcp_risk"].to_numpy()
    return summary.set_index("region")


def _fit_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    name: str,
    params: dict,
    param_grid: dict,
    outer_cv: StratifiedKFold | KFold,
    n_outer_folds: int,
    split_labels: pd.Series | None,
    inner_cv: KFold | None,
    random_state: int,
    n_iter: int,
) -> dict:
    """Fit every outer fold; collect per-fold scores and out-of-fold predictions."""
    folds = []
    for fold_id, (fit_idx, held_out_idx) in enumerate(
        tqdm(
            outer_cv.split(X, split_labels),
            total=n_outer_folds,
            desc=f"Outer CV {name}",
        )
    ):
        scores = _fit_outer_fold(
            X.iloc[fit_idx],
            y.iloc[fit_idx],
            X.iloc[held_out_idx],
            y.iloc[held_out_idx],
            name=name,
            params=params,
            param_grid=param_grid,
            inner_cv=inner_cv,
            random_state=random_state,
            n_iter=n_iter,
            fold=fold_id,
        )
        folds.append(
            {**scores, "y_true": y.iloc[held_out_idx].tolist(), "fold_id": fold_id}
        )

    return {
        "folds": folds,
        "y_true": np.array([v for fold in folds for v in fold["y_true"]]),
        "y_pred": np.array([v for fold in folds for v in fold["y_pred"]]),
        "indices": [i for fold in folds for i in fold["indices"]],
        "fold": [fold["fold_id"] for fold in folds for _ in fold["indices"]],
    }


def _pooled_metrics(cv_result: dict) -> dict:
    y_true, y_pred = cv_result["y_true"], cv_result["y_pred"]
    folds = cv_result["folds"]
    rho = spearmanr(y_pred, y_true)
    return {
        "spearman": float(rho.statistic),
        "spearman_pvalue": float(rho.pvalue),
        "r2": float(r2_score(y_true, y_pred)),
        "r2_std": float(np.nanstd([fold["r2"] for fold in folds])),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mae_std": float(np.std([fold["mae"] for fold in folds])),
        "mse": float(np.mean((y_pred - y_true) ** 2)),
    }


def _primary_details(cv_result: dict, feature_cols: list[str]) -> dict:
    folds = cv_result["folds"]
    mean_importances = np.mean([fold["importances"] for fold in folds], axis=0)
    return {
        "per_fold": [
            {
                "fold": fold["fold_id"],
                "spearman": fold["spearman"],
                "r2": fold["r2"],
                "mae": fold["mae"],
                **{f"param_{k}": v for k, v in fold["best_params"].items()},
                "best_score": fold["best_score"],
            }
            for fold in folds
        ],
        "feature_importances": [
            {"feature": feature, "importance": importance}
            for feature, importance in zip(feature_cols, mean_importances.tolist())
        ],
    }


def _failure_rate_distribution(rates: pd.Series) -> dict:
    """Summary stats of the failure-rate distribution over the used regions."""
    if rates.empty:
        return {}
    quantiles = rates.quantile([0.25, 0.5, 0.75, 0.9])
    return {
        "min": float(rates.min()),
        "p25": float(quantiles[0.25]),
        "median": float(quantiles[0.5]),
        "p75": float(quantiles[0.75]),
        "p90": float(quantiles[0.9]),
        "max": float(rates.max()),
        "mean": float(rates.mean()),
        "pct_zero": float((rates == 0).mean()),
    }


# Numeric summary columns that are not descriptors: the target and its support, the
# classifier's own risk (a baseline, not a descriptor), and `class_id`, which names a
# class and measures nothing.
_NOT_FEATURES = ("failure_rate", "n_eval", "mcp_risk", "class_id")


@timed
def fit_failure_regressor(
    summary: pd.DataFrame,
    *,
    models: dict[str, dict],
    primary: str,
    n_outer_splits: int,
    n_inner_splits: int,
    n_iter: int,
    random_state: int,
    min_eval_support: int,
) -> tuple[dict, pd.DataFrame, Pipeline | None]:
    """Nested CV of every model on the same outer folds: the primary's results, its
    held-out `predicted_rate` and `fold` per region, and the primary refitted on every
    used region (None when the target is degenerate)."""
    _check_models(models, primary)
    logger.info("Running failure regressor ...")

    # A region can still end up with no routed row, e.g. a small one in a single split.
    no_eval = summary["failure_rate"].isna()
    low_support = ~no_eval & (summary["n_eval"] < min_eval_support)
    n_excluded_no_eval = int(no_eval.sum())
    n_excluded_low_support = int(low_support.sum())
    used = summary[~no_eval & ~low_support]

    rates = used["failure_rate"].astype(float)
    n_eval = used["n_eval"].astype(float)
    global_error_rate = (
        float((rates * n_eval).sum() / n_eval.sum()) if n_eval.sum() else 0.0
    )
    exclusions = {
        "n_regions_total": int(no_eval.size),
        "n_regions_used": int(len(used)),
        "n_excluded_no_eval": n_excluded_no_eval,
        "n_excluded_low_support": n_excluded_low_support,
        "min_eval_support": min_eval_support,
        "global_error_rate": global_error_rate,
    }
    total_excluded = n_excluded_no_eval + n_excluded_low_support
    if total_excluded:
        # Losing more than a fifth of the regions earns a warning: routine on a single
        # split, whose test rows alone starve per-region support.
        excluded_frac = total_excluded / no_eval.size if no_eval.size else 0.0
        log = logger.warning if excluded_frac > 0.2 else logger.info
        log(
            "Excluded regions — no evaluated rows: %d, support < %d: %d; %d/%d used",
            n_excluded_no_eval,
            min_eval_support,
            n_excluded_low_support,
            len(used),
            no_eval.size,
        )

    # A descriptor no region has a value for carries nothing to learn from.
    feature_cols = [
        c
        for c in used.select_dtypes("number").columns
        if c not in _NOT_FEATURES and used[c].notna().any()
    ]
    X = used[feature_cols].copy()
    y = used["failure_rate"].astype(float)

    rate_distribution = {
        "failure_rate_distribution": _failure_rate_distribution(rates),
    }
    n_used = len(used)
    if n_used < 2 or float(y.std()) < 1e-9:
        message = (
            f"Failure regressor skipped: {n_used} usable region(s), "
            f"failure-rate std={float(y.std()):.4g}. Need >=2 regions with variance."
        )
        logger.warning("[STAGE-SKIP] %s", message)
        skipped = {
            "skipped": True,
            "reason": "degenerate_target",
            "message": message,
            **exclusions,
            **rate_distribution,
        }
        no_held_out = pd.DataFrame(
            {
                "predicted_rate": pd.Series(dtype=float),
                "fold": pd.Series(dtype=int),
            },
            index=pd.Index([], name="region"),
        )
        return skipped, no_held_out, None

    strata = _quantile_strata(y, n_outer_splits)
    if strata is not None:
        n_outer = _usable_n_splits(int(strata.value_counts().min()), n_outer_splits)
    else:
        n_outer = 0
    if n_outer >= 2:
        outer_cv = StratifiedKFold(
            n_splits=n_outer, shuffle=True, random_state=random_state
        )
        split_labels = strata
    else:
        n_outer = _usable_n_splits(n_used, n_outer_splits)
        outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)
        split_labels = None

    n_fit_worst = n_used - math.ceil(n_used / n_outer)
    n_inner = _usable_n_splits(n_fit_worst, n_inner_splits)
    if n_outer < n_outer_splits or n_inner < n_inner_splits:
        logger.warning(
            "[CV-ADAPT] Adapting CV (regions=%d): outer %d→%d, inner %d→%d%s",
            n_used,
            n_outer_splits,
            n_outer,
            n_inner_splits,
            n_inner or 0,
            " (no search — using model defaults)" if n_inner == 0 else "",
        )

    inner_cv = (
        KFold(n_splits=n_inner, shuffle=True, random_state=random_state)
        if n_inner > 0
        else None
    )

    cv_results = {
        name: _fit_nested_cv(
            X,
            y,
            name=name,
            params=spec["params"],
            param_grid=spec["param_grid"],
            outer_cv=outer_cv,
            n_outer_folds=n_outer,
            split_labels=split_labels,
            inner_cv=inner_cv,
            random_state=random_state,
            n_iter=n_iter,
        )
        for name, spec in models.items()
    }
    metrics = {name: _pooled_metrics(result) for name, result in cv_results.items()}

    logger.info("Refitting %s on the %d used regions ...", primary, n_used)
    primary_spec = models[primary]
    refit_model, refit_params, refit_score = _fit_model(
        X,
        y,
        name=primary,
        params=primary_spec["params"],
        param_grid=primary_spec["param_grid"],
        inner_cv=inner_cv,
        random_state=random_state,
        n_iter=n_iter,
        # The seed after the outer folds' own, so the refit searches a slice of its own.
        search_seed=random_state + n_outer,
    )

    results = {
        **exclusions,
        **rate_distribution,
        **metrics[primary],
        **_primary_details(cv_results[primary], feature_cols),
        "model": primary,
        "refit": [
            {
                "model": primary,
                "best_score": refit_score,
                **{f"param_{k}": v for k, v in refit_params.items()},
            }
        ],
        "models": [{"model": name, **scores} for name, scores in metrics.items()],
        "model_folds": [
            {
                "model": name,
                "fold": fold["fold_id"],
                "spearman": fold["spearman"],
                "r2": fold["r2"],
                "mae": fold["mae"],
                "best_score": fold["best_score"],
            }
            for name, result in cv_results.items()
            for fold in result["folds"]
        ],
    }
    for name, scores in metrics.items():
        logger.info(
            "Failure regressor %s — Spearman: %.4f, R²: %.4f, MAE: %.4f, MSE: %.4f",
            name,
            scores["spearman"],
            scores["r2"],
            scores["mae"],
            scores["mse"],
        )
    primary_result = cv_results[primary]
    held_out = pd.DataFrame(
        {"predicted_rate": primary_result["y_pred"], "fold": primary_result["fold"]},
        index=pd.Index(primary_result["indices"], name="region"),
    )
    return results, held_out, refit_model
