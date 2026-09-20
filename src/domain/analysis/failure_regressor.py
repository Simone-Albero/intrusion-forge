import logging
import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
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


def _run_outer_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    inner_cv: KFold | None,
    param_grid: dict,
    random_state: int,
) -> dict:
    """Run one outer CV fold. When inner_cv is None, skip GridSearchCV and use default RF."""
    if inner_cv is None:
        best = RandomForestRegressor(random_state=random_state)
        best.fit(X_train, y_train)
    else:
        grid = GridSearchCV(
            estimator=RandomForestRegressor(random_state=random_state),
            param_grid=param_grid,
            cv=inner_cv,
            scoring="r2",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

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
    """Merge per-cluster and class-level complexity with the observed failure rates.

    Cluster and class share the same measure names, so each side keeps its prefix: they
    are two different numbers about the same quantity, not a naming accident.
    """
    by_class = {rec["class_id"]: rec for rec in class_complexity}
    errors = {rec["cluster_id"]: rec for rec in predictions.get("clusters", [])}

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
                "n_test": error.get("n_total", 0),
                "failure_rate": error.get("error_rate"),
                "mcp_risk": error.get("mcp_risk"),
            }
        )
    return summary


def _run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv: StratifiedKFold | KFold,
    outer_k: int,
    split_labels: pd.Series | None,
    inner_cv: KFold | None,
    param_grid: dict,
    random_state: int,
) -> dict:
    """Run the outer CV loop and collect per-fold scores + out-of-fold predictions."""
    fold_r2s: list[float] = []
    fold_maes: list[float] = []
    fold_spearmans: list[float] = []
    fold_importances: list[np.ndarray] = []
    oof_y_true: list[float] = []
    oof_y_pred: list[float] = []
    oof_indices: list = []
    oof_fold_ids: list[int] = []

    for f, (train_idx, test_idx) in enumerate(
        tqdm(outer_cv.split(X, split_labels), total=outer_k, desc="Outer CV")
    ):
        fold = _run_outer_fold(
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[test_idx],
            inner_cv,
            param_grid,
            random_state,
        )
        fold_r2s.append(fold["r2"])
        fold_maes.append(fold["mae"])
        fold_spearmans.append(fold["spearman"])
        fold_importances.append(fold["importances"])
        oof_y_true.extend(y.iloc[test_idx].tolist())
        oof_y_pred.extend(fold["y_pred"])
        oof_indices.extend(fold["indices"])
        oof_fold_ids.extend([f] * len(fold["y_pred"]))

    return {
        "fold_r2s": fold_r2s,
        "fold_maes": fold_maes,
        "fold_spearmans": fold_spearmans,
        "fold_importances": fold_importances,
        "y_true": np.array(oof_y_true),
        "y_pred": np.array(oof_y_pred),
        "indices": oof_indices,
        "fold_ids": np.array(oof_fold_ids),
    }


def _aggregate_oof_results(oof: dict, feature_cols: list[str]) -> dict:
    """Aggregate out-of-fold predictions into the published regression-metrics block."""
    y_true, y_pred = oof["y_true"], oof["y_pred"]
    mean_importances = np.mean(oof["fold_importances"], axis=0)
    rho = spearmanr(y_pred, y_true)
    fold_spearmans = np.array(oof["fold_spearmans"], dtype=float)

    return {
        "spearman": float(rho.statistic),
        "spearman_pvalue": float(rho.pvalue),
        "r2": float(r2_score(y_true, y_pred)),
        "r2_std": float(np.nanstd(oof["fold_r2s"])),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mae_std": float(np.std(oof["fold_maes"])),
        "mse": float(np.mean((y_pred - y_true) ** 2)),
        "per_fold": [
            {"fold": f, "spearman": spearman, "r2": r2, "mae": mae}
            for f, (spearman, r2, mae) in enumerate(
                zip(fold_spearmans.tolist(), oof["fold_r2s"], oof["fold_maes"])
            )
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
    random_state: int = 42,
    min_test_support: int = 5,
) -> dict:
    """Fit a nested-CV Random Forest predicting each cluster's failure rate from its features."""
    logger.info("Running failure regressor ...")
    df = pd.DataFrame(cluster_stats).set_index("cluster_id")

    is_noise = (
        df["is_noise_cluster"].fillna(0).astype(bool)
        if "is_noise_cluster" in df
        else pd.Series(False, index=df.index)
    )
    no_test = df["failure_rate"].isna()
    low_support = ~no_test & (df["n_test"].fillna(0) < min_test_support)
    n_excluded_no_test = int(no_test.sum())
    n_excluded_low_support = int((low_support & ~is_noise).sum())
    n_excluded_noise = int((is_noise & ~no_test).sum())
    noise_test_share = (
        float(df.loc[is_noise & ~no_test, "n_test"].fillna(0).sum())
        / float(df.loc[~no_test, "n_test"].fillna(0).sum())
        if df.loc[~no_test, "n_test"].fillna(0).sum()
        else 0.0
    )
    df = df[~no_test & ~low_support & ~is_noise]

    rates = df["failure_rate"].astype(float)
    n_test = df["n_test"].astype(float)
    global_error_rate = (
        float((rates * n_test).sum() / n_test.sum()) if n_test.sum() else 0.0
    )
    exclusions = {
        "n_clusters_total": int(no_test.size),
        "n_clusters_used": int(len(df)),
        "n_excluded_no_test": n_excluded_no_test,
        "n_excluded_low_support": n_excluded_low_support,
        "n_excluded_noise": n_excluded_noise,
        "noise_test_share": noise_test_share,
        "min_test_support": min_test_support,
        "global_error_rate": global_error_rate,
    }
    if n_excluded_no_test or n_excluded_low_support or n_excluded_noise:
        logger.info(
            "Excluded clusters — no test: %d, support < %d: %d, noise pseudo-clusters: %d "
            "(%.1f%% of test support); %d/%d used",
            n_excluded_no_test,
            min_test_support,
            n_excluded_low_support,
            n_excluded_noise,
            100.0 * noise_test_share,
            len(df),
            no_test.size,
        )

    if feature_cols is None:
        feature_cols = [
            c
            for c in df.select_dtypes("number").columns
            if c not in ("failure_rate", "n_test", "is_noise_cluster", "mcp_risk")
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
            " (no GridSearchCV — using RF defaults)" if inner_k == 0 else "",
        )

    inner_cv = (
        KFold(n_splits=inner_k, shuffle=True, random_state=random_state)
        if inner_k > 0
        else None
    )

    oof = _run_nested_cv(
        X, y, outer_cv, outer_k, split_labels, inner_cv, param_grid, random_state
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


_RATE_BASELINE_NAMES = ("region", "mcp_cluster", "atc_cluster")


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

    mcp_cluster = np.empty_like(mcp)
    for c in np.unique(cluster):
        m = cluster == c
        mcp_cluster[m] = mcp[m].mean()
    atc_cluster = atc_cluster_risk(confidence, correct, cluster)

    n = failure.size
    combo_rankavg = rankdata(region) / (n + 1) + rankdata(mcp) / (n + 1)
    combo_atc_rankavg = rankdata(region) / (n + 1) + rankdata(atc_cluster) / (n + 1)

    scores = {
        "mcp_cluster": mcp_cluster,
        "atc_cluster": atc_cluster,
        "region": region,
        "combo_rankavg": combo_rankavg,
        "combo_atc_rankavg": combo_atc_rankavg,
    }

    support = np.ones(failure.size)
    clusters = np.unique(cluster)
    observed = np.array([failure[cluster == c].mean() for c in clusters], dtype=float)

    baselines = []
    for name, sc in scores.items():
        predicted = np.array([sc[cluster == c].mean() for c in clusters], dtype=float)
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
                    if name in _RATE_BASELINE_NAMES
                    else None
                ),
            }
        )

    return {
        "n_test": int(len(samples)),
        "n_clusters": int(clusters.size),
        "baselines": baselines,
    }
