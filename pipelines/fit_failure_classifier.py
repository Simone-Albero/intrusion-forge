import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm import tqdm

from src.core.config import load_config, save_config, to_container
from src.core.log import (
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from pipelines.common import paths_from_cfg
from src.core.utils import flush_timing, load_from_json, timed

setup_logger(log_file="resources/logs.txt")
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
    inner_cv: StratifiedKFold | None,
    param_grid: dict,
    random_state: int,
) -> dict:
    """Run one outer CV fold. When inner_cv is None, skip GridSearchCV and use default RF."""
    if inner_cv is None:
        best = RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced",
        )
        best.fit(X_train, y_train)
    else:
        grid = GridSearchCV(
            estimator=RandomForestClassifier(
                random_state=random_state,
                class_weight="balanced",
            ),
            param_grid=param_grid,
            cv=inner_cv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")

    return {
        "f1": f1_score(y_test, y_pred),
        "auc": auc,
        "importances": best.feature_importances_,
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
        "indices": X_test.index.tolist(),
    }


def _global_error_rate(cluster_errors: dict) -> float:
    """Pooled test error rate over every cluster with test samples."""
    n_err = sum(v["n_error"] for v in cluster_errors.values())
    n_tot = sum(v["n_total"] for v in cluster_errors.values())
    return n_err / n_tot if n_tot else 0.0


def _label_failed(
    n_error: int,
    n_total: int,
    *,
    labeling: str,
    alpha: float,
    threshold: float,
    global_error_rate: float,
) -> bool:
    """Failure label for one cluster.

    binomial:  failed when the cluster's error count is significantly above the
               classifier's global error rate (one-sided binomial test). Scale-
               aware: large clusters need a real elevation, small ones a strong
               excess — a single slip among tens of thousands of test samples
               no longer counts as failure.
    threshold: failed when failure_rate > threshold (legacy semantics).
    """
    if n_total == 0:
        return False
    if labeling == "binomial":
        return n_error > 0 and float(
            binom.sf(n_error - 1, n_total, global_error_rate)
        ) < alpha
    return (n_error / n_total) > threshold


def build_cluster_summary(
    complexity: dict,
    class_complexity: dict,
    predictions: dict,
    *,
    labeling: str = "binomial",
    alpha: float = 0.05,
    failure_threshold: float = 0.0,
) -> dict:
    """Merge per-cluster complexity with class-level complexity (joined on the
    cluster's class via `cluster_class`) and per-classifier failure rates.

    Output schema per cluster:
        cluster_<measure>  — cluster-level complexity (vs top-K adversarial clusters)
        class_<measure>    — class-level complexity of the cluster's class
        cluster_class, is_noise_cluster, n_test, failure_rate, is_failed
    """
    if labeling not in ("binomial", "threshold"):
        raise ValueError(
            f"Unknown labeling: {labeling!r}. Valid: 'binomial', 'threshold'."
        )
    cluster_errors = predictions.get("clusters", {}).get("global", {}) or {}
    global_error_rate = _global_error_rate(cluster_errors)
    summary: dict[str, dict] = {}
    for cid, cluster_measures in complexity.items():
        class_id = cluster_measures.get("cluster_class")
        class_measures = (
            class_complexity.get(str(class_id), {}) if class_id is not None else {}
        )
        cluster_feats = {
            f"cluster_{k}": v
            for k, v in cluster_measures.items()
            if k not in ("cluster_class", "is_noise_cluster")
        }
        class_feats = {
            f"class_{k}": v
            for k, v in class_measures.items()
            if k != "is_noise_cluster"
        }
        error_entry = cluster_errors.get(str(cid), {})
        failure_rate = error_entry.get("error_rate")
        summary[str(cid)] = {
            **cluster_feats,
            **class_feats,
            "cluster_class": class_id,
            "is_noise_cluster": cluster_measures.get("is_noise_cluster", False),
            "n_test": error_entry.get("n_total", 0),
            "failure_rate": failure_rate,
            "is_failed": failure_rate is not None
            and _label_failed(
                error_entry.get("n_error", 0),
                error_entry.get("n_total", 0),
                labeling=labeling,
                alpha=alpha,
                threshold=failure_threshold,
                global_error_rate=global_error_rate,
            ),
        }
    return summary


def _run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv: StratifiedKFold,
    outer_k: int,
    inner_cv: StratifiedKFold | None,
    param_grid: dict,
    random_state: int,
) -> dict:
    """Run the outer CV loop and collect per-fold scores + out-of-fold predictions."""
    fold_f1s: list[float] = []
    fold_aucs: list[float] = []
    fold_importances: list[np.ndarray] = []
    oof_y_true: list[int] = []
    oof_y_pred: list[int] = []
    oof_y_proba: list[float] = []
    oof_indices: list = []

    for train_idx, test_idx in tqdm(
        outer_cv.split(X, y), total=outer_k, desc="Outer CV"
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
        fold_f1s.append(fold["f1"])
        fold_aucs.append(fold["auc"])
        fold_importances.append(fold["importances"])
        oof_y_true.extend(y.iloc[test_idx].tolist())
        oof_y_pred.extend(fold["y_pred"])
        oof_y_proba.extend(fold["y_proba"])
        oof_indices.extend(fold["indices"])

    return {
        "fold_f1s": fold_f1s,
        "fold_aucs": fold_aucs,
        "fold_importances": fold_importances,
        "y_true": np.array(oof_y_true),
        "y_pred": np.array(oof_y_pred),
        "y_proba": np.array(oof_y_proba),
        "indices": oof_indices,
    }


def _aggregate_oof_results(oof: dict, feature_cols: list[str]) -> dict:
    """Aggregate out-of-fold predictions into the published metrics block."""
    y_true, y_pred, y_proba = oof["y_true"], oof["y_pred"], oof["y_proba"]
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    mean_importances = np.mean(oof["fold_importances"], axis=0)
    try:
        oof_auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        oof_auc = float("nan")

    return {
        "f1_score": float(f1_score(y_true, y_pred)),
        "f1_score_std": float(np.std(oof["fold_f1s"])),
        "f1_scores_per_fold": oof["fold_f1s"],
        "roc_auc": oof_auc,
        "roc_auc_std": float(np.nanstd(oof["fold_aucs"])),
        "roc_auc_per_fold": oof["fold_aucs"],
        "roc_curve_data": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            digits=4,
            output_dict=True,
        ),
        "feature_importances": dict(zip(feature_cols, mean_importances.tolist())),
        "oof_predictions": {
            str(cid): int(pred == true)
            for cid, pred, true in zip(oof["indices"], y_pred, y_true)
        },
        "oof_risk_proba": {
            str(cid): float(proba) for cid, proba in zip(oof["indices"], y_proba)
        },
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
    }


@timed
def fit_failure_classifier(
    cluster_stats: dict,
    param_grid: dict,
    *,
    feature_cols: list[str] | None = None,
    n_outer_splits: int = 5,
    n_inner_splits: int = 5,
    random_state: int = 42,
    labeling: str = "binomial",
    alpha: float = 0.05,
    failure_threshold: float = 0.0,
    min_test_support: int = 5,
    analysis_bus: LogDispatcher | None = None,
) -> dict:
    """Train a Random Forest with nested CV to predict cluster failure from separability features.

    The failure label comes from the `is_failed` column of `cluster_stats`
    (computed by `build_cluster_summary` under the configured labeling);
    `labeling`/`alpha`/`failure_threshold` are reported for transparency.
    Clusters without a failure rate (no test samples) or with fewer than
    `min_test_support` test samples are excluded — their failure label would
    be fabricated or near-random noise. Uses nested cross-validation: outer
    StratifiedKFold for unbiased evaluation, inner GridSearchCV for
    hyperparameter selection. Metrics are aggregated over out-of-fold (OOF)
    predictions.
    """
    logger.info("Running failure classifier ...")
    df = pd.DataFrame.from_dict(cluster_stats, orient="index")

    no_test = df["failure_rate"].isna()
    low_support = ~no_test & (df["n_test"].fillna(0) < min_test_support)
    n_excluded_no_test = int(no_test.sum())
    n_excluded_low_support = int(low_support.sum())
    df = df[~no_test & ~low_support]

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
        "min_test_support": min_test_support,
        "labeling": labeling,
        "alpha": alpha,
        "threshold": failure_threshold,
        "global_error_rate": global_error_rate,
    }
    if n_excluded_no_test or n_excluded_low_support:
        logger.info(
            "Excluded clusters — no test samples: %d, test support < %d: %d (%d/%d used)",
            n_excluded_no_test,
            min_test_support,
            n_excluded_low_support,
            len(df),
            no_test.size,
        )

    if feature_cols is None:
        feature_cols = [
            c
            for c in df.select_dtypes("number").columns
            if c not in ("is_failed", "failure_rate", "n_test")
        ]
    X = df[feature_cols].copy()

    y = df["is_failed"].astype(int)

    n_positives = int(y.sum())
    n_negatives = int((1 - y).sum())
    prevalence = n_positives / max(n_positives + n_negatives, 1)
    context_metrics = {
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "prevalence": prevalence,
        # F1 of the constant all-failed predictor: the floor any useful model must beat
        "f1_baseline_all_failed": 2 * prevalence / (1 + prevalence) if prevalence else 0.0,
        "failure_rate_distribution": _failure_rate_distribution(df["failure_rate"]),
    }
    n_minority = min(n_positives, n_negatives)
    outer_k = _max_safe_splits(n_minority, n_outer_splits)
    if outer_k == 0:
        message = (
            f"Failure classifier skipped: only {n_minority} minority sample(s) "
            f"(positives={n_positives}, labeling={labeling}). "
            f"Need >=2 for stratified CV."
        )
        logger.warning("[STAGE-SKIP] %s", message)
        results = {
            "skipped": True,
            "reason": "insufficient_minority_samples",
            "message": message,
            "n_minority": n_minority,
            "min_required": 2,
            **exclusions,
            **context_metrics,
        }
        if analysis_bus is not None:
            analysis_bus.publish(
                LogBundle.from_dict({"json/analysis/classifier_results": results})
            )
        return results

    m_train_worst = n_minority - math.ceil(n_minority / outer_k)
    inner_k = _max_safe_splits(m_train_worst, n_inner_splits)
    if outer_k < n_outer_splits or inner_k < n_inner_splits:
        logger.warning(
            "[CV-ADAPT] Adapting CV (minority=%d): outer %d→%d, inner %d→%d%s",
            n_minority,
            n_outer_splits,
            outer_k,
            n_inner_splits,
            inner_k or 0,
            " (no GridSearchCV — using RF defaults)" if inner_k == 0 else "",
        )

    outer_cv = StratifiedKFold(
        n_splits=outer_k, shuffle=True, random_state=random_state
    )
    inner_cv = (
        StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=random_state)
        if inner_k > 0
        else None
    )

    oof = _run_nested_cv(X, y, outer_cv, outer_k, inner_cv, param_grid, random_state)

    # rank correlation between the RF risk score and the continuous failure
    # rate: measures the relationship without the binarisation
    oof_rates = df.loc[oof["indices"], "failure_rate"].astype(float).to_numpy()
    rho = spearmanr(oof["y_proba"], oof_rates)

    results = {
        **exclusions,
        **context_metrics,
        "spearman_proba_failure_rate": float(rho.statistic),
        "spearman_proba_failure_rate_pvalue": float(rho.pvalue),
        **_aggregate_oof_results(oof, feature_cols),
    }
    if analysis_bus is not None:
        analysis_bus.publish(
            LogBundle.from_dict({"json/analysis/classifier_results": results})
        )
    logger.info(
        "Classifier results — F1: %.4f, ROC-AUC: %.4f",
        results["f1_score"],
        results["roc_auc"],
    )
    return results


def main():
    """Main entry point for failure-classifier training (per-classifier stage)."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    paths = paths_from_cfg(cfg)
    save_config(cfg, paths.configs / "config_composed.json")

    complexity_path = paths.shared / "complexity.json"
    class_complexity_path = paths.shared / "class_complexity.json"
    for p in (complexity_path, class_complexity_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing complexity artifact at {p}. "
                "Run `make complexity` first."
            )
    complexity = load_from_json(complexity_path)
    class_complexity = load_from_json(class_complexity_path)
    predictions = load_from_json(paths.outputs / "analysis/predictions/test.json")

    cluster_summary = build_cluster_summary(
        complexity,
        class_complexity,
        predictions,
        labeling=cfg.failure_classifier.labeling,
        alpha=cfg.failure_classifier.alpha,
        failure_threshold=cfg.failure_classifier.threshold,
    )

    bus = LogDispatcher()
    bus.subscribe(JSONSubscriber(paths.outputs))
    bus.publish(LogBundle.from_dict({"json/analysis/cluster_summary": cluster_summary}))
    logger.info("Cluster summary published.")

    fit_failure_classifier(
        cluster_summary,
        to_container(cfg.failure_classifier.param_grid),
        n_outer_splits=cfg.failure_classifier.n_outer_splits,
        n_inner_splits=cfg.failure_classifier.n_inner_splits,
        labeling=cfg.failure_classifier.labeling,
        alpha=cfg.failure_classifier.alpha,
        failure_threshold=cfg.failure_classifier.threshold,
        min_test_support=cfg.failure_classifier.min_test_support,
        random_state=cfg.seed,
        analysis_bus=bus,
    )

    flush_timing(paths.outputs / "timing.json")


if __name__ == "__main__":
    main()
