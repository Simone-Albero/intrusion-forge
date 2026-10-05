import logging
import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import (
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
)
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.confidence import atc_region_risk
from src.domain.analysis.failure import is_failure
from src.domain.analysis.grouping import RowsBy
from src.domain.analysis.risk_coverage import oracle_benefit_recovered
from src.engine.ml.model import MLRegressorFactory
from src.engine.ml.preprocessing import REGRESSOR_PREPROCESS, build_regressor_pipeline

logger = logging.getLogger(__name__)


def _max_safe_splits(n_minority: int, n_splits_cfg: int) -> int:
    """Largest k <= n_splits_cfg such that StratifiedKFold(k) won't degenerate."""
    k = min(n_splits_cfg, n_minority)
    return k if k >= 2 else 0


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


def _fit_outer_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    name: str,
    params: dict,
    param_grid: dict,
    inner_cv: KFold | None,
    random_state: int,
    n_iter: int,
    fold: int,
) -> dict:
    """Fit and score one outer fold; without inner_cv, use the model's defaults, no search."""
    params = {**params, "random_state": random_state}
    if inner_cv is None:
        best = build_regressor_pipeline(name, params)
        best.fit(X_train, y_train)
        model_params = best.named_steps["model"].get_params()
        best_params = {k: model_params[k] for k in param_grid}
        best_score = None
    else:
        search = RandomizedSearchCV(
            estimator=build_regressor_pipeline(name, params),
            param_distributions={f"model__{k}": v for k, v in param_grid.items()},
            # A grid smaller than the budget is searched whole: asking for more draws than
            # it holds makes sklearn warn on every fold and run the whole grid anyway.
            n_iter=min(n_iter, len(ParameterGrid(param_grid))),
            cv=inner_cv,
            # Scored as reported, clipped: a model is tuned for the predictor it is judged as.
            scoring=make_scorer(lambda y, p: r2_score(y, np.clip(p, 0.0, 1.0))),
            n_jobs=-1,
            # Offset by fold: at a fixed random_state RandomizedSearchCV draws the same
            # combinations whatever the data, so every outer fold would search the same
            # slice of the grid.
            random_state=random_state + fold,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        best_params = {
            k.removeprefix("model__"): v for k, v in search.best_params_.items()
        }
        best_score = float(search.best_score_)

    # A rate lies in [0, 1] and a forest's mean of rates always does; the other models
    # are not bound to it.
    y_pred = np.clip(best.predict(X_test), 0.0, 1.0)
    has_variance = len(y_test) > 1 and np.std(y_test) > 0 and np.std(y_pred) > 0
    return {
        "r2": float(r2_score(y_test, y_pred)) if len(y_test) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "spearman": (
            float(spearmanr(y_pred, y_test).statistic) if has_variance else float("nan")
        ),
        "importances": getattr(best.named_steps["model"], "feature_importances_", None),
        "y_pred": y_pred.tolist(),
        "indices": X_test.index.tolist(),
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
    outer_k: int,
    split_labels: pd.Series | None,
    inner_cv: KFold | None,
    random_state: int,
    n_iter: int,
) -> dict:
    """Fit every outer fold; collect per-fold scores and out-of-fold predictions."""
    folds = []
    for f, (train_idx, test_idx) in enumerate(
        tqdm(outer_cv.split(X, split_labels), total=outer_k, desc=f"Outer CV {name}")
    ):
        fold = _fit_outer_fold(
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[test_idx],
            name=name,
            params=params,
            param_grid=param_grid,
            inner_cv=inner_cv,
            random_state=random_state,
            n_iter=n_iter,
            fold=f,
        )
        folds.append({**fold, "y_true": y.iloc[test_idx].tolist(), "fold_id": f})

    return {
        "folds": folds,
        "y_true": np.array([v for fold in folds for v in fold["y_true"]]),
        "y_pred": np.array([v for fold in folds for v in fold["y_pred"]]),
        "indices": [i for fold in folds for i in fold["indices"]],
    }


def _oof_metrics(oof: dict) -> dict:
    y_true, y_pred = oof["y_true"], oof["y_pred"]
    folds = oof["folds"]
    rho = spearmanr(y_pred, y_true)
    return {
        "spearman": float(rho.statistic),
        "spearman_pvalue": float(rho.pvalue),
        "r2": float(r2_score(y_true, y_pred)),
        "r2_std": float(np.nanstd([f["r2"] for f in folds])),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mae_std": float(np.std([f["mae"] for f in folds])),
        "mse": float(np.mean((y_pred - y_true) ** 2)),
    }


def _primary_details(oof: dict, feature_cols: list[str]) -> dict:
    folds = oof["folds"]
    mean_importances = np.mean([f["importances"] for f in folds], axis=0)
    return {
        "per_fold": [
            {
                "fold": f["fold_id"],
                "spearman": f["spearman"],
                "r2": f["r2"],
                "mae": f["mae"],
                **{f"param_{k}": v for k, v in f["best_params"].items()},
                "best_score": f["best_score"],
            }
            for f in folds
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
) -> tuple[dict, pd.Series]:
    """Fit every model by nested CV on the same outer folds; return the primary's results and held-out predictions."""
    _check_models(models, primary)
    logger.info("Running failure regressor ...")
    df = summary

    # A region can still end up with no routed row, e.g. a small one in a single split.
    no_eval = df["failure_rate"].isna()
    low_support = ~no_eval & (df["n_eval"] < min_eval_support)
    n_excluded_no_eval = int(no_eval.sum())
    n_excluded_low_support = int(low_support.sum())
    df = df[~no_eval & ~low_support]

    rates = df["failure_rate"].astype(float)
    n_eval = df["n_eval"].astype(float)
    global_error_rate = (
        float((rates * n_eval).sum() / n_eval.sum()) if n_eval.sum() else 0.0
    )
    exclusions = {
        "n_regions_total": int(no_eval.size),
        "n_regions_used": int(len(df)),
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
            len(df),
            no_eval.size,
        )

    # A descriptor no region has a value for carries nothing to learn from.
    feature_cols = [
        c
        for c in df.select_dtypes("number").columns
        if c not in _NOT_FEATURES and df[c].notna().any()
    ]
    X = df[feature_cols].copy()
    y = df["failure_rate"].astype(float)

    context_metrics = {
        "failure_rate_distribution": _failure_rate_distribution(rates),
    }
    n_used = len(df)
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
            **context_metrics,
        }
        return skipped, pd.Series(dtype=float, name="predicted_rate")

    strata = _quantile_strata(y, n_outer_splits)
    if strata is not None:
        outer_k = _max_safe_splits(int(strata.value_counts().min()), n_outer_splits)
    else:
        outer_k = 0
    if outer_k >= 2:
        outer_cv = StratifiedKFold(
            n_splits=outer_k, shuffle=True, random_state=random_state
        )
        split_labels = strata
    else:
        outer_k = _max_safe_splits(n_used, n_outer_splits)
        outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=random_state)
        split_labels = None

    m_train_worst = n_used - math.ceil(n_used / outer_k)
    inner_k = _max_safe_splits(m_train_worst, n_inner_splits)
    if outer_k < n_outer_splits or inner_k < n_inner_splits:
        logger.warning(
            "[CV-ADAPT] Adapting CV (regions=%d): outer %d→%d, inner %d→%d%s",
            n_used,
            n_outer_splits,
            outer_k,
            n_inner_splits,
            inner_k or 0,
            " (no search — using model defaults)" if inner_k == 0 else "",
        )

    inner_cv = (
        KFold(n_splits=inner_k, shuffle=True, random_state=random_state)
        if inner_k > 0
        else None
    )

    oofs = {
        name: _fit_nested_cv(
            X,
            y,
            name=name,
            params=spec["params"],
            param_grid=spec["param_grid"],
            outer_cv=outer_cv,
            outer_k=outer_k,
            split_labels=split_labels,
            inner_cv=inner_cv,
            random_state=random_state,
            n_iter=n_iter,
        )
        for name, spec in models.items()
    }
    metrics = {name: _oof_metrics(oof) for name, oof in oofs.items()}

    results = {
        **exclusions,
        **context_metrics,
        **metrics[primary],
        **_primary_details(oofs[primary], feature_cols),
        "model": primary,
        "models": [{"model": name, **m} for name, m in metrics.items()],
        "model_folds": [
            {
                "model": name,
                "fold": f["fold_id"],
                "spearman": f["spearman"],
                "r2": f["r2"],
                "mae": f["mae"],
                "best_score": f["best_score"],
            }
            for name, oof in oofs.items()
            for f in oof["folds"]
        ],
    }
    for name, m in metrics.items():
        logger.info(
            "Failure regressor %s — Spearman: %.4f, R²: %.4f, MAE: %.4f, MSE: %.4f",
            name,
            m["spearman"],
            m["r2"],
            m["mae"],
            m["mse"],
        )
    oof = oofs[primary]
    predicted_rate = pd.Series(
        oof["y_pred"],
        index=pd.Index(oof["indices"], name="region"),
        name="predicted_rate",
    )
    return results, predicted_rate


# Each calibrated variant and the raw one it maps onto the failure rate's scale.
CALIBRATED_VARIANTS = {
    "mcp_region_cal": "mcp_region",
    "atc_region_cal": "atc_region",
    "train_rate_region_cal": "train_rate_region",
}
BASELINE_VARIANTS = (
    "mcp_region",
    "atc_region",
    "region",
    "combo_rankavg",
    "combo_atc_rankavg",
    "train_rate_region",
    "val_rate_region",
    *CALIBRATED_VARIANTS,
)
RATE_BASELINE_VARIANTS = (
    "region",
    "mcp_region",
    "atc_region",
    "train_rate_region",
    "val_rate_region",
    *CALIBRATED_VARIANTS,
)


def instance_baselines(
    samples: pd.DataFrame,
    predicted_rate: pd.Series,
    *,
    atc_threshold: float,
    train_rate: pd.Series,
    val_rate: pd.Series,
    calibrated: dict[str, pd.Series],
) -> dict:
    """Region rho, region-rate MSE and oracle benefit of every baseline variant."""
    # `atc_threshold` is the confidence cut, chosen on rows other than `samples`;
    # `train_rate` and `val_rate` are the failure rates the classifier made on other rows
    # of each region, `calibrated` the rate of each `CALIBRATED_VARIANTS` entry, all
    # indexed by region.
    # Only the regions the regressor scored, so every variant ranks the same regions.
    samples = samples[samples["region"].isin(predicted_rate.index)]
    region_of_row = samples["region"].to_numpy()
    by_region = RowsBy(region_of_row)
    failure = is_failure(
        samples["y_true"].to_numpy(), samples["y_pred"].to_numpy()
    ).astype(float)
    mcp = samples["mcp_risk"].to_numpy(dtype=float)
    confidence = 1.0 - mcp
    region = by_region.spread(predicted_rate.loc[by_region.ids].to_numpy(dtype=float))
    mcp_region = by_region.spread(by_region.reduce(mcp))
    train_rate_region = by_region.spread(
        train_rate.loc[by_region.ids].to_numpy(dtype=float)
    )
    val_rate_region = by_region.spread(
        val_rate.loc[by_region.ids].to_numpy(dtype=float)
    )
    observed = by_region.reduce(failure)
    atc_region = atc_region_risk(confidence, region_of_row, threshold=atc_threshold)

    n = failure.size
    region_rank = rankdata(region) / (n + 1)
    combo_rankavg = region_rank + rankdata(mcp) / (n + 1)
    combo_atc_rankavg = region_rank + rankdata(atc_region) / (n + 1)

    scores = {
        "mcp_region": mcp_region,
        "atc_region": atc_region,
        "region": region,
        "combo_rankavg": combo_rankavg,
        "combo_atc_rankavg": combo_atc_rankavg,
        "train_rate_region": train_rate_region,
        "val_rate_region": val_rate_region,
        **{
            name: by_region.spread(rate.loc[by_region.ids].to_numpy(dtype=float))
            for name, rate in calibrated.items()
        },
    }

    # A rate variant holds one value per region, so that value is the prediction:
    # averaging its copies moves the last ulp and breaks ties, and `region` would drift
    # from the regressor's own rho. The rank averages differ row to row and are averaged
    # with numpy, whose pairwise sum a pandas groupby would not reproduce to the ulp.
    predicted_by_name = {
        name: (
            by_region.first(scores[name])
            if name in RATE_BASELINE_VARIANTS
            else by_region.reduce(scores[name])
        )
        for name in BASELINE_VARIANTS
    }

    support = np.ones(failure.size)
    baselines = []
    for name in BASELINE_VARIANTS:
        sc = scores[name]
        predicted = predicted_by_name[name]
        rho = (
            float(spearmanr(predicted, observed).statistic)
            if np.std(predicted) > 1e-12 and np.std(observed) > 1e-12
            else float("nan")
        )
        baselines.append(
            {
                "variant": name,
                "spearman": rho,
                # A NaN score, a baseline val could not calibrate, ranks no row.
                "oracle_benefit_recovered": (
                    oracle_benefit_recovered(sc, failure, support)
                    if np.isfinite(sc).all()
                    else float("nan")
                ),
                # Null rather than absent: the rank-average variants have no rate to
                # compare, and a uniform row shape is what makes this a table.
                "region_rate_mse": (
                    float(np.mean((predicted - observed) ** 2))
                    if name in RATE_BASELINE_VARIANTS
                    else None
                ),
            }
        )

    return {
        "n_eval": int(len(samples)),
        "n_regions": int(by_region.ids.size),
        "baselines": baselines,
    }


# The predictions the by-size tables and figures compare, in the order of their bars.
SIZE_VARIANTS = (
    "region",
    "train_rate_region",
    "val_rate_region",
    "mcp_region",
    "atc_region",
)
# The by-size squared and signed errors add the calibrated variants, each beside its raw
# one; a positive slope keeps the raw one's rho up to float resolution, so the rho
# figures leave them out.
SIZE_ERROR_VARIANTS = (
    "region",
    "train_rate_region",
    "train_rate_region_cal",
    "val_rate_region",
    "mcp_region",
    "mcp_region_cal",
    "atc_region",
    "atc_region_cal",
)


def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
    se = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else np.nan
    return float(values.mean()), se


def error_by_region_size(
    observed: pd.Series,
    predictions: dict[str, pd.Series],
    *,
    size: pd.Series,
    n_bins: int,
) -> list[dict]:
    """Per bin of region size, each prediction's squared and signed error and its rho."""
    regions = observed.index
    if len(regions) < n_bins:
        return []
    order = np.lexsort((regions.to_numpy(), size.loc[regions].to_numpy()))
    bin_of = np.empty(len(regions), dtype=int)
    bin_of[order] = np.arange(len(regions)) * n_bins // len(regions)

    rate = observed.to_numpy(dtype=float)
    rows = []
    for b in range(n_bins):
        in_bin = bin_of == b
        sizes = size.loc[regions[in_bin]]
        for variant, predicted in predictions.items():
            value = predicted.loc[regions[in_bin]].to_numpy(dtype=float)
            error = value - rate[in_bin]
            mse, mse_se = _mean_and_se(error**2)
            bias, bias_se = _mean_and_se(error)
            rows.append(
                {
                    "size_bin": b,
                    "variant": variant,
                    "n_regions": int(in_bin.sum()),
                    "size_min": int(sizes.min()),
                    "size_max": int(sizes.max()),
                    "spearman": (
                        float(spearmanr(value, rate[in_bin]).statistic)
                        if np.std(value) > 1e-12 and np.std(rate[in_bin]) > 1e-12
                        else float("nan")
                    ),
                    "mse": mse,
                    "mse_se": mse_se,
                    "bias": bias,
                    "bias_se": bias_se,
                }
            )
    return rows
