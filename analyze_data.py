import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ignite.handlers.tensorboard_logger import TensorboardLogger
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

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import load_from_json, save_to_json

from src.data.io import load_listed_dfs
from src.data.analyze import (
    compute_pairwise_separability,
    build_cluster_summary,
)
from src.plot.array import (
    confusion_matrix_to_plot,
    feature_importance_plot,
    roc_curve_plot,
    strip_box_plot,
    violin_box_plot,
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

RELEVANT_GEOMETRIC_FEATURES = [
    "max_dispersion",
    "dist_to_nearest_foreign_cluster",
    "foreign_max_max",
]

N_CONFUSED_PAIRS = 3

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 3, 5, 10, 20],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", 0.5],
}


def find_correlation(
    cluster_stats: dict,
    param_grid: dict,
    feature_cols: list[str] | None = None,
    n_outer_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Train a Random Forest with nested CV to predict cluster failure from separability features.

    Uses nested cross-validation: outer StratifiedKFold for unbiased evaluation,
    inner GridSearchCV for hyperparameter selection. Metrics are aggregated over
    out-of-fold (OOF) predictions.

    Args:
        cluster_stats: Per-cluster summary dict (output of ``build_cluster_summary``).
        feature_cols: Column names to use as features. If None, all numeric columns
            except ``is_failed`` are used.
        param_grid: Hyperparameter grid for ``GridSearchCV``.
        n_outer_splits: Number of outer CV folds.
        random_state: RNG seed.

    Returns:
        Dict with f1_score (mean), f1_score_std, f1_scores_per_fold, roc_auc (mean),
        roc_auc_std, roc_auc_per_fold, roc_curve_data (OOF), confusion_matrix (OOF),
        classification_report (OOF), and feature_importances (mean across folds).
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

    outer_cv = StratifiedKFold(
        n_splits=n_outer_splits, shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    fold_f1s: list[float] = []
    fold_aucs: list[float] = []
    fold_importances: list[np.ndarray] = []
    oof_y_true: list[int] = []
    oof_y_pred: list[int] = []
    oof_y_proba: list[float] = []

    for train_idx, test_idx in tqdm(
        outer_cv.split(X, y), total=n_outer_splits, desc="Outer CV"
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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

        fold_f1s.append(f1_score(y_test, y_pred))
        fold_aucs.append(roc_auc_score(y_test, y_proba))
        fold_importances.append(best.feature_importances_)
        oof_y_true.extend(y_test.tolist())
        oof_y_pred.extend(y_pred.tolist())
        oof_y_proba.extend(y_proba.tolist())

    oof_y_true_arr = np.array(oof_y_true)
    oof_y_pred_arr = np.array(oof_y_pred)
    oof_y_proba_arr = np.array(oof_y_proba)

    fpr, tpr, _ = roc_curve(oof_y_true_arr, oof_y_proba_arr)
    mean_importances = np.mean(fold_importances, axis=0)

    return {
        "f1_score": float(np.mean(fold_f1s)),
        "f1_score_std": float(np.std(fold_f1s)),
        "f1_scores_per_fold": fold_f1s,
        "roc_auc": float(np.mean(fold_aucs)),
        "roc_auc_std": float(np.std(fold_aucs)),
        "roc_auc_per_fold": fold_aucs,
        "roc_curve_data": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "confusion_matrix": confusion_matrix(oof_y_true_arr, oof_y_pred_arr).tolist(),
        "classification_report": classification_report(
            oof_y_true_arr,
            oof_y_pred_arr,
            digits=4,
            output_dict=True,
        ),
        "feature_importances": dict(zip(feature_cols, mean_importances.tolist())),
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
        sep = compute_pairwise_separability(
            X, y, max_pairs=50_000, metric=cfg.distance_metric
        )
        save_to_json(
            sep,
            Path(cfg.path.json_logs)
            / f"analysis/separability/cluster_{split_name}.json",
        )
        results.setdefault("cluster", {})[split_name] = sep

    return results


def _find_most_confused_pairs(
    cm_data: list[list[int]],
    label_mapping: dict[str, str],
    n: int = N_CONFUSED_PAIRS,
) -> list[tuple[str, str]]:
    """Return the top-n most-confused class pairs from a confusion matrix.

    Confusion between class i and j is measured as the symmetric normalised rate:
        cm[i][j] / sum(cm[i])  +  cm[j][i] / sum(cm[j])

    This is robust to class imbalance because each direction is normalised by
    the true-class sample count before summing.
    """
    cm = np.array(cm_data, dtype=float)
    row_sums = cm.sum(axis=1)
    n_classes = cm.shape[0]

    pairs: list[tuple[float, int, int]] = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            rate = cm[i, j] / row_sums[i] + cm[j, i] / row_sums[j]
            pairs.append((rate, i, j))

    pairs.sort(key=lambda x: x[0], reverse=True)

    return [(label_mapping[str(i)], label_mapping[str(j)]) for _, i, j in pairs[:n]]


def compute_summary_visualizzations(
    cluster_summary: dict,
    df_meta: dict,
    test_summary: dict,
    correlation_results: dict,
    cfg,
):
    logger.info("Generating per-class visualizations ...")
    tb_logger = TensorboardLogger(
        log_dir=Path(cfg.path.tb_logs) / "analysis",
    )
    step = cfg.run_id or 0

    summary_df = pd.DataFrame.from_dict(cluster_summary, orient="index")
    summary_df["class_name"] = summary_df["cluster_class"].map(df_meta["label_mapping"])

    fig = strip_box_plot(
        categories=summary_df["class_name"].values,
        values=summary_df["failure_rate"].values,
        color_values=summary_df["failure_rate"].values,
        x_label="class",
        y_label="Failure rate",
        c_label="Failure rate",
        title=f"Failure rate per class ({cfg.data.file_name})",
    )
    tb_logger.writer.add_figure("summary/failure_rate_strip_box", fig, step)
    plt.close(fig)

    fig = strip_box_plot(
        categories=summary_df["class_name"].values,
        values=summary_df["failure_rate"].values,
        color_values=summary_df["cluster_size"].values,
        x_label="class",
        y_label="Failure rate",
        c_label="Cluster size",
        title=f"Failure rate per class ({cfg.data.file_name})",
    )
    tb_logger.writer.add_figure("summary/failure_rate_strip_box_sizes", fig, step)
    plt.close(fig)

    logger.info("Generating per-failed visualizations ...")
    categories = np.where(summary_df["is_failed"], "failed", "correct")

    for feature in RELEVANT_GEOMETRIC_FEATURES:
        fig = violin_box_plot(
            categories=categories,
            values=summary_df[feature].values,
            x_label="outcome",
            y_label=feature,
            title=f"{feature} by outcome ({cfg.data.file_name})",
        )
        tb_logger.writer.add_figure(f"summary/global/{feature}", fig, step)
        plt.close(fig)

    logger.info("Generating per-confused-pair visualizations ...")
    confused_pairs = _find_most_confused_pairs(
        test_summary["confusion_matrix"],
        df_meta["label_mapping"],
    )
    logger.info(
        "Top-%d confused pairs: %s",
        len(confused_pairs),
        [(a, b) for a, b in confused_pairs],
    )

    for rank, (class_a, class_b) in enumerate(confused_pairs):
        pair_df = summary_df[summary_df["class_name"].isin([class_a, class_b])]
        tag_prefix = f"summary/confused_pairs/{rank}_{class_a}_vs_{class_b}"
        for feature in RELEVANT_GEOMETRIC_FEATURES:
            fig = violin_box_plot(
                categories=pair_df["class_name"].values,
                values=pair_df[feature].values,
                x_label="class",
                y_label=feature,
                title=f"{feature}: {class_a} vs {class_b} ({cfg.data.file_name})",
            )
            tb_logger.writer.add_figure(f"{tag_prefix}/{feature}", fig, step)
            plt.close(fig)

    logger.info("Generating correlation confusion matrix ...")
    fig = confusion_matrix_to_plot(
        cm=np.array(correlation_results["confusion_matrix"]),
        class_names=["correct", "failed"],
        title=f"Cluster failure predictor ({cfg.data.file_name})",
        figsize=(6, 5),
    )
    tb_logger.writer.add_figure("summary/correlation/confusion_matrix", fig, step)
    plt.close(fig)

    logger.info("Generating correlation analysis plots ...")
    roc_data = correlation_results["roc_curve_data"]
    fig = roc_curve_plot(
        fpr=np.array(roc_data["fpr"]),
        tpr=np.array(roc_data["tpr"]),
        auc_score=correlation_results["roc_auc"],
        title=f"ROC curve ({cfg.data.file_name})",
    )
    tb_logger.writer.add_figure("summary/correlation/roc_curve", fig, step)
    plt.close(fig)

    n_features = len(correlation_results["feature_importances"])
    fig = feature_importance_plot(
        importances=correlation_results["feature_importances"],
        title=f"Feature importances ({cfg.data.file_name})",
        figsize=(12, max(8, n_features * 0.5)),
    )
    tb_logger.writer.add_figure("summary/correlation/feature_importances", fig, step)
    plt.close(fig)
    tb_logger.close()


def analyze(cfg):
    """Run data analysis pipeline."""
    logs = Path(cfg.path.json_logs)

    # --- 1. Separability ---
    # run_separability_analysis(cfg)

    # --- 2. Cluster summary ---
    separability = load_from_json(logs / "analysis/separability/cluster_test.json")
    pred_infos = load_from_json(logs / "analysis/predictions/test.json")
    clusters_meta = load_from_json(logs / "data/clusters_meta.json")
    df_meta = load_from_json(logs / "data/df_meta.json")
    test_summary = load_from_json(logs / "test/summary.json")

    cluster_summary = build_cluster_summary(
        class_to_clusters=clusters_meta["class_to_clusters"],
        clusters_distribution=clusters_meta["clusters_distribution"],
        cluster_stats=clusters_meta["cluster_stats"],
        cluster_errors=pred_infos["clusters"]["global"],
        separability=separability,
    )
    save_to_json(cluster_summary, logs / "analysis/cluster_summary.json")
    logger.info("Cluster summary saved.")

    # --- 3. Correlation analysis ---
    logger.info("Running correlation analysis ...")
    correlation_results = find_correlation(
        cluster_stats=cluster_summary,
        param_grid=RF_PARAM_GRID,
    )
    save_to_json(correlation_results, logs / "analysis/correlation_results.json")
    logger.info(
        "Correlation results — F1: %.4f, ROC-AUC: %.4f",
        correlation_results["f1_score"],
        correlation_results["roc_auc"],
    )

    # --- 4. Per-class visualizations (TensorBoard) ---
    correlation_results = load_from_json(logs / "analysis/correlation_results.json")
    compute_summary_visualizzations(
        cluster_summary, df_meta, test_summary, correlation_results, cfg
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
