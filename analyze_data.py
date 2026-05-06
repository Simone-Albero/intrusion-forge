import logging
import sys
from dataclasses import dataclass
from pathlib import Path

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

from src.common.config import load_config, save_config
from src.common.log import (
    LogDispatcher,
    JSONSubscriber,
    LogBundle,
    TensorBoardSubscriber,
    setup_logger,
)
from src.common.utils import flush_timing, load_from_json, timed

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
from src.plot.base import Plot

setup_logger(log_file="resources/logs.txt")
logger = logging.getLogger(__name__)

RELEVANT_GEOMETRIC_FEATURES = [
    "max_dispersion",
    "p5_silhouette",
    "frac_at_risk",
]

SEPARABILITY_MAX_PAIRS = 50_000

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 3, 5, 10, 20],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", 0.5],
}


@dataclass
class OutputPaths:
    """Output paths for the analysis pipeline."""

    json_logs: Path
    tb_logs: Path
    processed_data: Path
    configs: Path


@dataclass
class SeparabilityConfig:
    """Parameters for cluster separability computation."""

    feature_cols: list[str]
    extension: str
    distance_metric: str


def _run_outer_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    inner_cv: StratifiedKFold,
    param_grid: dict,
    random_state: int,
) -> dict:
    """Run one outer CV fold: fit GridSearchCV, predict, collect per-fold metrics.

    Output keys: 'f1', 'auc', 'importances' (np.ndarray),
    'y_pred' (list[int]), 'y_proba' (list[float]), 'indices' (list).
    """
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

    return {
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "importances": best.feature_importances_,
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
        "indices": X_test.index.tolist(),
    }


@timed
def fit_failure_classifier(
    cluster_stats: dict,
    param_grid: dict,
    feature_cols: list[str] | None = None,
    n_outer_splits: int = 5,
    n_inner_splits: int = 5,
    random_state: int = 42,
    failure_threshold: float = 0.0,
) -> dict:
    """Train a Random Forest with nested CV to predict cluster failure from separability features.

    Uses nested cross-validation: outer StratifiedKFold for unbiased evaluation,
    inner GridSearchCV for hyperparameter selection. Metrics are aggregated over
    out-of-fold (OOF) predictions.
    """
    df = pd.DataFrame.from_dict(cluster_stats, orient="index")
    if feature_cols is None:
        feature_cols = [
            c
            for c in df.select_dtypes("number").columns
            if c != "is_failed" and c != "failure_rate"
        ]
    X = df[feature_cols].copy()

    y = df["failure_rate"].apply(lambda x: 1 if x > failure_threshold else 0)

    outer_cv = StratifiedKFold(
        n_splits=n_outer_splits, shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedKFold(
        n_splits=n_inner_splits, shuffle=True, random_state=random_state
    )

    fold_f1s: list[float] = []
    fold_aucs: list[float] = []
    fold_importances: list[np.ndarray] = []
    oof_y_true: list[int] = []
    oof_y_pred: list[int] = []
    oof_y_proba: list[float] = []
    oof_indices: list = []

    for train_idx, test_idx in tqdm(
        outer_cv.split(X, y), total=n_outer_splits, desc="Outer CV"
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

    oof_y_true_arr = np.array(oof_y_true)
    oof_y_pred_arr = np.array(oof_y_pred)
    oof_y_proba_arr = np.array(oof_y_proba)

    fpr, tpr, _ = roc_curve(oof_y_true_arr, oof_y_proba_arr)
    mean_importances = np.mean(fold_importances, axis=0)

    results = {
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
        "oof_predictions": {
            str(cid): int(pred == true)
            for cid, pred, true in zip(oof_indices, oof_y_pred, oof_y_true)
        },
    }
    return results


def measure_cluster_separability(
    processed_data_path: Path,
    sep_cfg: SeparabilityConfig,
) -> dict[str, dict]:
    """Compute pairwise cluster separability on train and test splits.

    Returns:
        {"train": <sep_train>, "test": <sep_test>}
    """
    train_df, _, test_df = load_listed_dfs(
        processed_data_path,
        [
            f"train.{sep_cfg.extension}",
            f"val.{sep_cfg.extension}",
            f"test.{sep_cfg.extension}",
        ],
    )

    separability = {}
    for split_name, split_df in (("train", train_df), ("test", test_df)):
        X = split_df[sep_cfg.feature_cols].values
        y = split_df["cluster"].values
        separability[split_name] = compute_pairwise_separability(
            X, y, max_pairs=SEPARABILITY_MAX_PAIRS, metric=sep_cfg.distance_metric
        )
    return separability


def _plot_failure_strips(
    summary_df: pd.DataFrame, rf_correct: np.ndarray, title: str
) -> dict[str, Plot]:
    """Build failure-rate and RF-prediction strip-box plots."""
    return {
        "summary/rf_prediction_strip_box": strip_box_plot(
            categories=summary_df["class_name"].values,
            values=summary_df["failure_rate"].values,
            color_values=summary_df["failure_rate"].values,
            edge_values=rf_correct,
            x_label="class",
            y_label="Failure rate",
            c_label="Failure rate",
            edge_label="RF prediction",
            edge_value_labels={0.0: "failed", 1.0: "correct"},
            title=title,
        ),
        "summary/failure_rate_strip_box": strip_box_plot(
            categories=summary_df["class_name"].values,
            values=summary_df["failure_rate"].values,
            color_values=summary_df["failure_rate"].values,
            x_label="class",
            y_label="Failure rate",
            c_label="Failure rate",
            title=title,
        ),
    }


def _plot_feature_by_outcome(
    summary_df: pd.DataFrame, features: list[str], title: str
) -> dict[str, Plot]:
    """Build per-feature violin-box plots split by failed/correct outcome."""
    categories = np.where(summary_df["is_failed"], "failed", "correct")
    return {
        f"summary/global/{feature}": violin_box_plot(
            categories=categories,
            values=summary_df[feature].values,
            x_label="outcome",
            y_label=feature,
            title=title,
            category_order=["correct", "failed"],
        )
        for feature in features
    }


def _plot_rf_evaluation(classifier_results: dict, title: str) -> dict[str, Plot]:
    """Build confusion matrix, ROC curve, and feature importance plots."""
    n_features = len(classifier_results["feature_importances"])
    max_features_to_plot = min(n_features, 20)
    top_importances = dict(
        sorted(
            classifier_results["feature_importances"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:max_features_to_plot]
    )
    roc_data = classifier_results["roc_curve_data"]
    return {
        "summary/correlation/confusion_matrix": confusion_matrix_to_plot(
            cm=np.array(classifier_results["confusion_matrix"]),
            class_names=["correct", "failed"],
            title=f"Random Forest prediction [{title}]",
            figsize=(6, 5),
        ),
        "summary/correlation/roc_curve": roc_curve_plot(
            fpr=np.array(roc_data["fpr"]),
            tpr=np.array(roc_data["tpr"]),
            auc_score=classifier_results["roc_auc"],
            title=f"ROC curve [{title}]",
        ),
        "summary/correlation/feature_importances": feature_importance_plot(
            importances=top_importances,
            title=f"Feature importances [{title}]",
            figsize=(12, max(8, max_features_to_plot * 0.5)),
        ),
    }


def assemble_analysis_figures(
    cluster_summary: dict,
    df_meta: dict,
    classifier_results: dict,
    title: str,
) -> dict[str, Plot]:
    """Build all analysis figures and return them as a prefixed dict."""
    summary_df = pd.DataFrame.from_dict(cluster_summary, orient="index")
    summary_df["class_name"] = summary_df["cluster_class"].map(df_meta["label_mapping"])

    oof_preds = classifier_results.get("oof_predictions", {})
    rf_correct = np.array(
        [oof_preds.get(str(cid), np.nan) for cid in summary_df.index], dtype=float
    )

    figures: dict[str, Plot] = {}
    figures.update(_plot_failure_strips(summary_df, rf_correct, title))
    figures.update(
        _plot_feature_by_outcome(summary_df, RELEVANT_GEOMETRIC_FEATURES, title)
    )
    figures.update(_plot_rf_evaluation(classifier_results, title))
    return figures


@timed
def analyze(
    paths: OutputPaths,
    separability: dict[str, dict],
    step: int,
    title: str,
    failure_threshold: float,
) -> None:
    """Run data analysis pipeline."""
    logger.info("Starting analysis pipeline ...")

    pred_infos = load_from_json(paths.json_logs / "analysis/predictions/test.json")
    clusters_meta = load_from_json(paths.json_logs / "data/clusters_meta.json")
    df_meta = load_from_json(paths.json_logs / "data/df_meta.json")

    cluster_summary = build_cluster_summary(
        class_to_clusters=clusters_meta["class_to_clusters"],
        clusters_distribution=clusters_meta["clusters_distribution"],
        cluster_stats=clusters_meta["cluster_stats"],
        cluster_errors=pred_infos["clusters"]["global"],
        separability=separability["test"],
    )

    analysis_bus = LogDispatcher()
    tb_logger = TensorboardLogger(log_dir=paths.tb_logs / "analysis")
    analysis_bus.subscribe(TensorBoardSubscriber(tb_logger.writer))
    analysis_bus.subscribe(JSONSubscriber(paths.json_logs))
    try:
        for split_name, sep in separability.items():
            analysis_bus.publish(
                LogBundle.from_dict(
                    {f"json/analysis/separability/cluster_{split_name}": sep}
                )
            )
        analysis_bus.publish(
            LogBundle.from_dict({"json/analysis/cluster_summary": cluster_summary})
        )
        logger.info("Cluster summary and separability published.")
        logger.info("Running failure classifier ...")
        classifier_results = fit_failure_classifier(
            cluster_stats=cluster_summary,
            param_grid=RF_PARAM_GRID,
            failure_threshold=failure_threshold,
        )
        analysis_bus.publish(
            LogBundle.from_dict(
                {"json/analysis/classifier_results": classifier_results}
            )
        )
        logger.info(
            "Classifier results — F1: %.4f, ROC-AUC: %.4f",
            classifier_results["f1_score"],
            classifier_results["roc_auc"],
        )
        logger.info("Building summary visualizations ...")
        figures = assemble_analysis_figures(
            cluster_summary=cluster_summary,
            df_meta=df_meta,
            classifier_results=classifier_results,
            title=title,
        )
        analysis_bus.publish(LogBundle(figures=figures, step=step))
    finally:
        tb_logger.close()


def main():
    """Main entry point for data analysis."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    paths = OutputPaths(
        json_logs=Path(cfg.path.json_logs),
        tb_logs=Path(cfg.path.tb_logs),
        processed_data=Path(cfg.path.processed_data),
        configs=Path(cfg.path.configs),
    )
    save_config(cfg, paths.configs / "config_composed.json")

    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    sep_cfg = SeparabilityConfig(
        feature_cols=num_cols + cat_cols,
        extension=cfg.data.extension,
        distance_metric=cfg.distance_metric,
    )

    logger.info("Computing cluster separability ...")
    separability = measure_cluster_separability(paths.processed_data, sep_cfg)
    # separability = {
    #     split: load_from_json(paths.json_logs / f"analysis/separability/cluster_{split}.json")
    #     for split in ("train", "test")
    # }

    analyze(
        paths=paths,
        separability=separability,
        step=cfg.run_id or 0,
        title=cfg.data.file_name,
        failure_threshold=cfg.failure_threshold or 0.0,
    )
    flush_timing(paths.json_logs / "timing.json")


if __name__ == "__main__":
    main()
