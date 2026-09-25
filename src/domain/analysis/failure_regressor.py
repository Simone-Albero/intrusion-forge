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
from src.domain.analysis.confidence import atc_cluster_risk
from src.domain.analysis.failure import is_failure
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
    inner_cv: KFold | None,
    param_grid: dict,
    *,
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
    try:
        bins = pd.qcut(y, q=min(q, len(y)), duplicates="drop")
    except (ValueError, IndexError):
        return None
    if bins.nunique() < 2:
        return None
    return bins.cat.codes


def build_cluster_summary(
    complexity: list[dict],
    class_complexity: list[dict],
    predictions: dict,
) -> list[dict]:
    """Merge `cluster_`/`class_`-prefixed complexity with the observed failure rates."""
    by_class = {rec["class_id"]: rec for rec in class_complexity}
    errors = {rec["cluster_id"]: rec for rec in predictions["clusters"]}

    summary = []
    for cluster_measures in complexity:
        cluster_id = cluster_measures["cluster_id"]
        class_id = cluster_measures.get("cluster_class")
        class_measures = by_class.get(class_id, {}) if class_id is not None else {}
        cluster_feats = {
            f"cluster_{k}": v
            for k, v in cluster_measures.items()
            if k not in ("cluster_id", "cluster_class", "is_noise_cluster")
        }
        class_feats = {
            f"class_{k}": v
            for k, v in class_measures.items()
            if k not in ("class_id", "is_noise_cluster")
        }
        error = errors.get(cluster_id, {})
        summary.append(
            {
                "cluster_id": cluster_id,
                **cluster_feats,
                **class_feats,
                "cluster_class": class_id,
                "is_noise_cluster": int(
                    cluster_measures.get("is_noise_cluster", False)
                ),
                "n_eval": error["n_rows"] if error else 0,
                "failure_rate": error.get("error_rate"),
                "mcp_risk": error.get("mcp_risk"),
            }
        )
    return summary


def _fit_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv: StratifiedKFold | KFold,
    outer_k: int,
    split_labels: pd.Series | None,
    inner_cv: KFold | None,
    param_grid: dict,
    *,
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
            inner_cv,
            param_grid,
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
        "oof_predicted_rate": [
            {"cluster_id": cid, "predicted_rate": float(pred)}
            for cid, pred in zip(oof["indices"], y_pred)
        ],
    }


def _failure_rate_distribution(rates: pd.Series) -> dict:
    """Summary stats of the failure-rate distribution over the used clusters."""
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


@timed
def fit_failure_regressor(
    cluster_stats: dict,
    param_grid: dict,
    *,
    feature_cols: list[str] | None = None,
    n_outer_splits: int = 5,
    n_inner_splits: int = 5,
    n_iter: int = 40,
    random_state: int = 42,
    min_eval_support: int = 5,
) -> dict:
    """Fit a nested-CV Random Forest predicting each cluster's failure rate from its features."""
    logger.info("Running failure regressor ...")
    df = pd.DataFrame(cluster_stats).set_index("cluster_id")

    is_noise = (
        df["is_noise_cluster"].fillna(0).astype(bool)
        if "is_noise_cluster" in df
        else pd.Series(False, index=df.index)
    )
    no_eval = df["failure_rate"].isna()
    low_support = ~no_eval & (df["n_eval"].fillna(0) < min_eval_support)
    n_excluded_no_eval = int(no_eval.sum())
    n_excluded_low_support = int((low_support & ~is_noise).sum())
    n_excluded_noise = int((is_noise & ~no_eval).sum())
    noise_eval_share = (
        float(df.loc[is_noise & ~no_eval, "n_eval"].fillna(0).sum())
        / float(df.loc[~no_eval, "n_eval"].fillna(0).sum())
        if df.loc[~no_eval, "n_eval"].fillna(0).sum()
        else 0.0
    )
    df = df[~no_eval & ~low_support & ~is_noise]

    rates = df["failure_rate"].astype(float)
    n_eval = df["n_eval"].astype(float)
    global_error_rate = (
        float((rates * n_eval).sum() / n_eval.sum()) if n_eval.sum() else 0.0
    )
    exclusions = {
        "n_clusters_total": int(no_eval.size),
        "n_clusters_used": int(len(df)),
        "n_excluded_no_eval": n_excluded_no_eval,
        "n_excluded_low_support": n_excluded_low_support,
        "n_excluded_noise": n_excluded_noise,
        "noise_eval_share": noise_eval_share,
        "min_eval_support": min_eval_support,
        "global_error_rate": global_error_rate,
    }
    total_excluded = n_excluded_no_eval + n_excluded_low_support + n_excluded_noise
    if total_excluded:
        # Losing more than a fifth of the clusters earns a warning: routine on a single
        # split, whose test rows alone starve per-cluster support.
        excluded_frac = total_excluded / no_eval.size if no_eval.size else 0.0
        log = logger.warning if excluded_frac > 0.2 else logger.info
        log(
            "Excluded clusters — no evaluated rows: %d, support < %d: %d, noise "
            "pseudo-clusters: %d (%.1f%% of evaluated rows); %d/%d used",
            n_excluded_no_eval,
            min_eval_support,
            n_excluded_low_support,
            n_excluded_noise,
            100.0 * noise_eval_share,
            len(df),
            no_eval.size,
        )

    if feature_cols is None:
        feature_cols = [
            c
            for c in df.select_dtypes("number").columns
            if c not in ("failure_rate", "n_eval", "is_noise_cluster", "mcp_risk")
        ]
    X = df[feature_cols].copy()
    y = df["failure_rate"].astype(float)

    context_metrics = {
        "failure_rate_distribution": _failure_rate_distribution(rates),
    }
    n_used = len(df)
    if n_used < 2 or float(y.std()) < 1e-9:
        message = (
            f"Failure regressor skipped: {n_used} usable cluster(s), "
            f"failure-rate std={float(y.std()):.4g}. Need >=2 clusters with variance."
        )
        logger.warning("[STAGE-SKIP] %s", message)
        return {
            "skipped": True,
            "reason": "degenerate_target",
            "message": message,
            **exclusions,
            **context_metrics,
        }

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
            "[CV-ADAPT] Adapting CV (clusters=%d): outer %d→%d, inner %d→%d%s",
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
        outer_cv,
        outer_k,
        split_labels,
        inner_cv,
        param_grid,
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
    return results


RATE_BASELINE_NAMES = ("region", "mcp_cluster", "atc_cluster")


def instance_baselines(samples: pd.DataFrame, predicted_rate: list[dict]) -> dict:
    """Compare the 5 baseline variants against observed failure: cluster rho, cluster-rate MSE
    (rate variants only) and per-sample oracle benefit recovered."""
    rate_by_cluster = {r["cluster_id"]: r["predicted_rate"] for r in predicted_rate}
    cluster = samples["cluster"].to_numpy()
    failure = is_failure(
        samples["y_true"].to_numpy(), samples["y_pred"].to_numpy()
    ).astype(float)
    correct = 1.0 - failure
    mcp = samples["mcp_risk"].to_numpy(dtype=float)
    confidence = 1.0 - mcp
    fallback = (
        float(np.mean(list(rate_by_cluster.values()))) if rate_by_cluster else 0.0
    )
    region = np.array([rate_by_cluster.get(c, fallback) for c in cluster], dtype=float)

    clusters = np.unique(cluster)
    mcp_cluster = np.empty_like(mcp)
    observed = np.empty(clusters.size)
    for i, c in enumerate(clusters):
        m = cluster == c
        mcp_cluster[m] = mcp[m].mean()
        observed[i] = failure[m].mean()
    atc_cluster = atc_cluster_risk(confidence, correct, cluster)

    n = failure.size
    region_rank = rankdata(region) / (n + 1)
    combo_rankavg = region_rank + rankdata(mcp) / (n + 1)
    combo_atc_rankavg = region_rank + rankdata(atc_cluster) / (n + 1)

    scores = {
        "mcp_cluster": mcp_cluster,
        "atc_cluster": atc_cluster,
        "region": region,
        "combo_rankavg": combo_rankavg,
        "combo_atc_rankavg": combo_atc_rankavg,
    }

    # Not a pandas groupby: its Cython mean accumulates in a different order from
    # numpy's pairwise sum, so the two disagree in the last ulp on any cluster with
    # enough rows — enough to move the published spearman in its fifth decimal.
    predicted_by_name = {name: np.empty(clusters.size) for name in scores}
    for i, c in enumerate(clusters):
        m = cluster == c
        for name, sc in scores.items():
            predicted_by_name[name][i] = sc[m].mean()

    support = np.ones(failure.size)
    baselines = []
    for name, sc in scores.items():
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
                "cluster_rate_mse": (
                    float(np.mean((predicted - observed) ** 2))
                    if name in RATE_BASELINE_NAMES
                    else None
                ),
            }
        )

    return {
        "n_eval": int(len(samples)),
        "n_clusters": int(clusters.size),
        "baselines": baselines,
    }
