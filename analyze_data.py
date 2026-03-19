import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import load_from_json, save_to_json

from src.data.io import load_listed_dfs
from src.data.analyze import (
    compute_pairwise_separability,
    build_cluster_summary,
)

setup_logger()
logger = logging.getLogger(__name__)

SEPARABILITY_FEATURE_COLS = [
    "cluster_size",
    "n_samples",
    "n_unique",
    "unique_ratio",
    "intra_dispersion",
    "std_dispersion",
    "median_dispersion",
    "max_dispersion",
    "density",
    "log_density",
    "dist_to_class_centroid",
    "dist_to_nearest_cluster",
    "dist_to_nearest_foreign_cluster",
    "nearest_separation_ratio",
    "foreign_separation_ratio",
    "silhouette",
    "foreign_avg_avg",
    "foreign_max_avg",
    "foreign_avg_max",
    "foreign_max_max",
    "foreign_avg_std",
    "foreign_max_std",
    "self_avg",
    "self_max",
]

RF_PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", 0.5],
}


def find_correlation(
    cluster_stats: dict,
    param_grid: dict,
    feature_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> dict:
    """Train a Random Forest to predict cluster failure from separability features.

    Args:
        cluster_stats: Per-cluster summary dict (output of ``build_cluster_summary``).
        feature_cols: Column names to use as features. If None, all numeric columns
            except ``is_failed`` are used.
        param_grid: Hyperparameter grid for ``GridSearchCV``.
        test_size: Fraction held out for evaluation.
        random_state: RNG seed.

    Returns:
        Dict with best_params, f1_score, roc_auc, confusion_matrix,
        classification_report, and feature_importances.
    """
    df = pd.DataFrame.from_dict(cluster_stats, orient="index")
    if feature_cols is None:
        feature_cols = [
            c
            for c in df.select_dtypes("number").columns
            if c != "is_failed" and c != "failure_rate"
        ]
    X = df[feature_cols].copy()
    y = df["is_failed"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    grid = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    return {
        "best_params": grid.best_params_,
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            digits=4,
            output_dict=True,
        ),
        "feature_importances": dict(
            zip(feature_cols, best.feature_importances_.tolist())
        ),
    }


def run_separability_analysis(cfg):
    """Compute cluster separability on train/test splits and persist results."""
    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    feature_cols = num_cols + cat_cols

    logger.info("Loading data ...")
    train_df, _, test_df = load_listed_dfs(
        Path(cfg.path.processed_data),
        [
            f"train.{cfg.data.extension}",
            f"val.{cfg.data.extension}",
            f"test.{cfg.data.extension}",
        ],
    )

    results = {}
    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        logger.info("Computing separability on %s set ...", split_name)
        X = split_df[feature_cols].values
        y = split_df["cluster"].values
        sep = compute_pairwise_separability(X, y, max_pairs=50_000, metric="euclidean")
        save_to_json(
            sep,
            Path(cfg.path.json_logs) / f"data/separability/cluster_{split_name}.json",
        )
        results.setdefault("cluster", {})[split_name] = sep

    return results


def analyze(cfg):
    """Run data analysis pipeline."""
    logs = Path(cfg.path.json_logs)

    # --- 1. Separability ---
    run_separability_analysis(cfg)

    # --- 2. Cluster summary ---
    separability = load_from_json(logs / "data/separability/cluster_test.json")
    pred_infos = load_from_json(logs / "inference/pred_infos/test.json")
    clusters_meta = load_from_json(logs / "data/clusters_meta.json")

    cluster_summary = build_cluster_summary(
        class_to_clusters=clusters_meta["class_to_clusters"],
        clusters_distribution=clusters_meta["clusters_distribution"],
        cluster_stats=clusters_meta["cluster_stats"],
        cluster_errors=pred_infos["clusters"]["global"],
        separability=separability,
    )
    save_to_json(cluster_summary, logs / "inference/cluster_summary.json")
    logger.info("Cluster summary saved.")

    # --- 3. Correlation analysis ---
    logger.info("Running correlation analysis ...")
    correlation_results = find_correlation(
        cluster_stats=cluster_summary,
        # feature_cols=SEPARABILITY_FEATURE_COLS,
        param_grid=RF_PARAM_GRID,
    )
    save_to_json(correlation_results, logs / "inference/correlation_results.json")
    logger.info(
        "Correlation results — F1: %.4f, ROC-AUC: %.4f",
        correlation_results["f1_score"],
        correlation_results["roc_auc"],
    )


def main():
    """Main entry point for data analysis."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    analyze(cfg)


if __name__ == "__main__":
    main()
