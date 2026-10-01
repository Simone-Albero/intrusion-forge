import logging
import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.confidence import atc_region_risk
from src.domain.analysis.failure import is_failure
from src.domain.analysis.grouping import RowGroups
from src.domain.analysis.risk_coverage import oracle_benefit_recovered

logger = logging.getLogger(__name__)


def _max_safe_splits(n_minority: int, n_splits_cfg: int) -> int:
    """Largest k <= n_splits_cfg such that StratifiedKFold(k) won't degenerate."""
    k = min(n_splits_cfg, n_minority)
    return k if k >= 2 else 0


def _fit_outer_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    inner_cv: KFold | None,
    param_grid: dict,
    random_state: int,
    n_iter: int,
    fold: int,
) -> dict:
    """Fit and score one outer fold; without inner_cv, use a default RF, no search."""
    if inner_cv is None:
        best = RandomForestRegressor(random_state=random_state)
        best.fit(X_train, y_train)
        best_params = {k: best.get_params()[k] for k in param_grid}
        best_score = None
    else:
        search = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=random_state),
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=inner_cv,
            scoring="r2",
            n_jobs=-1,
            # Offset by fold: at a fixed random_state RandomizedSearchCV draws the same
            # combinations whatever the data, so every outer fold would search the same
            # slice of the grid.
            random_state=random_state + fold,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        best_params = search.best_params_
        best_score = float(search.best_score_)

    y_pred = best.predict(X_test)
    has_variance = len(y_test) > 1 and np.std(y_test) > 0 and np.std(y_pred) > 0
    return {
        "r2": float(r2_score(y_test, y_pred)) if len(y_test) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "spearman": (
            float(spearmanr(y_pred, y_test).statistic) if has_variance else float("nan")
        ),
        "importances": best.feature_importances_,
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
    outer_cv: StratifiedKFold | KFold,
    outer_k: int,
    split_labels: pd.Series | None,
    inner_cv: KFold | None,
    param_grid: dict,
    random_state: int,
    n_iter: int,
) -> dict:
    """Fit every outer fold; collect per-fold scores and out-of-fold predictions."""
    folds = []
    for f, (train_idx, test_idx) in enumerate(
        tqdm(outer_cv.split(X, split_labels), total=outer_k, desc="Outer CV")
    ):
        fold = _fit_outer_fold(
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[test_idx],
            inner_cv=inner_cv,
            param_grid=param_grid,
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


def _aggregate_oof_results(oof: dict, feature_cols: list[str]) -> dict:
    y_true, y_pred = oof["y_true"], oof["y_pred"]
    folds = oof["folds"]
    mean_importances = np.mean([f["importances"] for f in folds], axis=0)
    rho = spearmanr(y_pred, y_true)

    return {
        "spearman": float(rho.statistic),
        "spearman_pvalue": float(rho.pvalue),
        "r2": float(r2_score(y_true, y_pred)),
        "r2_std": float(np.nanstd([f["r2"] for f in folds])),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mae_std": float(np.std([f["mae"] for f in folds])),
        "mse": float(np.mean((y_pred - y_true) ** 2)),
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
    param_grid: dict,
    n_outer_splits: int,
    n_inner_splits: int,
    n_iter: int,
    random_state: int,
    min_eval_support: int,
) -> tuple[dict, pd.Series]:
    """Fit a nested-CV Random Forest predicting each region's failure rate; return the
    results and each scored region's prediction made while it was held out."""
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
            " (no search — using RF defaults)" if inner_k == 0 else "",
        )

    inner_cv = (
        KFold(n_splits=inner_k, shuffle=True, random_state=random_state)
        if inner_k > 0
        else None
    )

    oof = _fit_nested_cv(
        X,
        y,
        outer_cv=outer_cv,
        outer_k=outer_k,
        split_labels=split_labels,
        inner_cv=inner_cv,
        param_grid=param_grid,
        random_state=random_state,
        n_iter=n_iter,
    )

    results = {
        **exclusions,
        **context_metrics,
        **_aggregate_oof_results(oof, feature_cols),
    }
    logger.info(
        "Failure regressor results — Spearman: %.4f, R²: %.4f, MAE: %.4f, MSE: %.4f",
        results["spearman"],
        results["r2"],
        results["mae"],
        results["mse"],
    )
    predicted_rate = pd.Series(
        oof["y_pred"],
        index=pd.Index(oof["indices"], name="region"),
        name="predicted_rate",
    )
    return results, predicted_rate


BASELINE_VARIANTS = (
    "mcp_region",
    "atc_region",
    "region",
    "combo_rankavg",
    "combo_atc_rankavg",
    "train_rate_region",
    "val_rate_region",
)
RATE_BASELINE_VARIANTS = (
    "region",
    "mcp_region",
    "atc_region",
    "train_rate_region",
    "val_rate_region",
)


def instance_baselines(
    samples: pd.DataFrame,
    predicted_rate: pd.Series,
    *,
    atc_threshold: float,
    train_rate: pd.Series,
    val_rate: pd.Series,
) -> dict:
    """Region rho, region-rate MSE and oracle benefit of every baseline variant."""
    # `atc_threshold` is the confidence cut, chosen on rows other than `samples`;
    # `train_rate` and `val_rate` are the failure rates the classifier made on other rows
    # of each region, indexed by region.
    # Only the regions the regressor scored, so every variant ranks the same regions.
    samples = samples[samples["region"].isin(predicted_rate.index)]
    region_of_row = samples["region"].to_numpy()
    groups = RowGroups(region_of_row)
    failure = is_failure(
        samples["y_true"].to_numpy(), samples["y_pred"].to_numpy()
    ).astype(float)
    mcp = samples["mcp_risk"].to_numpy(dtype=float)
    confidence = 1.0 - mcp
    region = groups.spread(predicted_rate.loc[groups.ids].to_numpy(dtype=float))
    mcp_region = groups.spread(groups.reduce(mcp))
    train_rate_region = groups.spread(train_rate.loc[groups.ids].to_numpy(dtype=float))
    val_rate_region = groups.spread(val_rate.loc[groups.ids].to_numpy(dtype=float))
    observed = groups.reduce(failure)
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
    }

    # A rate variant holds one value per region, so that value is the prediction:
    # averaging its copies moves the last ulp and breaks ties, and `region` would drift
    # from the regressor's own rho. The rank averages differ row to row and are averaged
    # with numpy, whose pairwise sum a pandas groupby would not reproduce to the ulp.
    predicted_by_name = {
        name: (
            groups.first(scores[name])
            if name in RATE_BASELINE_VARIANTS
            else groups.reduce(scores[name])
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
                "oracle_benefit_recovered": oracle_benefit_recovered(
                    sc, failure, support
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
        "n_regions": int(groups.ids.size),
        "baselines": baselines,
    }


def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
    se = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else np.nan
    return float(values.mean()), se


def error_by_region_size(
    observed: pd.Series,
    predictions: dict[str, pd.Series],
    *,
    size: pd.Series,
    n_eval: pd.Series,
    n_bins: int,
) -> list[dict]:
    """Per bin of region size, each prediction's squared and signed error and the test noise."""
    regions = observed.index
    if len(regions) < n_bins:
        return []
    order = np.lexsort((regions.to_numpy(), size.loc[regions].to_numpy()))
    bin_of = np.empty(len(regions), dtype=int)
    bin_of[order] = np.arange(len(regions)) * n_bins // len(regions)

    rate = observed.to_numpy(dtype=float)
    support = n_eval.loc[regions].to_numpy(dtype=float)
    noise = np.where(
        support > 1, rate * (1 - rate) / np.maximum(support - 1, 1), np.nan
    )
    rows = []
    for b in range(n_bins):
        in_bin = bin_of == b
        sizes = size.loc[regions[in_bin]]
        for variant, predicted in predictions.items():
            error = predicted.loc[regions[in_bin]].to_numpy(dtype=float) - rate[in_bin]
            mse, mse_se = _mean_and_se(error**2)
            bias, bias_se = _mean_and_se(error)
            rows.append(
                {
                    "size_bin": b,
                    "variant": variant,
                    "n_regions": int(in_bin.sum()),
                    "size_min": int(sizes.min()),
                    "size_max": int(sizes.max()),
                    "test_noise": float(np.nanmean(noise[in_bin])),
                    "mse": mse,
                    "mse_se": mse_se,
                    "bias": bias,
                    "bias_se": bias_se,
                }
            )
    return rows
