import logging
import pickle as pkl
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from ignite.handlers.tensorboard_logger import TensorboardLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
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

from src.data.io import load_df
from src.data.complexity import compute_all_complexity_measures
from src.plot.array import (
    cluster_tsne_scatter,
    class_pair_tsne_scatter,
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
    # G-family (Euclidean)
    "max_dispersion",
    "p95_dispersion",
    "dist_to_nearest_foreign_cluster",
    "p5_silhouette",
    "frac_at_risk",
    "min_sibling_centroid_dist",
    # G-family (cosine — always computed on centroids)
    "max_dispersion_cosine",
    "p95_dispersion_cosine",
    "dist_to_nearest_foreign_cluster_cosine",
    "p5_silhouette_cosine",
    "min_sibling_centroid_dist_cosine",
    # F-family
    "f1_class_mean",
    "f2_class_mean",
    "f3_class_mean",
    "f4_class_mean",
    "f1_cluster_mean",
    "f2_cluster_mean",
    "f3_cluster_mean",
    "f4_cluster_mean",
    # N-family (Gower-cosine hybrid)
    "n1_class_mean",
    "n2_class_mean",
    "n3_class_mean",
    "n4_class_mean",
    "n1_cluster_mean",
    "n2_cluster_mean",
    "n3_cluster_mean",
    "n4_cluster_mean",
    # ND-family (Gower-cosine hybrid)
    "network_density_class_mean",
    "network_density_cluster_mean",
    # T-family
    "t2",
    "t3",
    "t4",
]

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
    pickle: Path


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
    paths: OutputPaths,
    param_grid: dict,
    cluster_stats: dict | None = None,
    feature_cols: list[str] | None = None,
    n_outer_splits: int = 5,
    n_inner_splits: int = 5,
    random_state: int = 42,
    failure_threshold: float = 0.0,
    analysis_bus: LogDispatcher | None = None,
) -> dict:
    """Train a Random Forest with nested CV to predict cluster failure from separability features.

    Uses nested cross-validation: outer StratifiedKFold for unbiased evaluation,
    inner GridSearchCV for hyperparameter selection. Metrics are aggregated over
    out-of-fold (OOF) predictions.
    """
    if cluster_stats is None:
        cluster_stats = load_from_json(
            paths.json_logs / "analysis/cluster_summary.json"
        )
    logger.info("Running failure classifier ...")
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


def _plot_failure_strips(
    summary_df: pd.DataFrame, rf_correct: np.ndarray
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
        ),
        "summary/failure_rate_strip_box": strip_box_plot(
            categories=summary_df["class_name"].values,
            values=summary_df["failure_rate"].values,
            color_values=summary_df["failure_rate"].values,
            x_label="class",
            y_label="Failure rate",
            c_label="Failure rate",
        ),
    }


def _plot_feature_by_outcome(
    summary_df: pd.DataFrame, features: list[str]
) -> dict[str, Plot]:
    """Build per-feature violin-box plots split by failed/correct outcome."""
    categories = np.where(summary_df["is_failed"], "failed", "correct")
    return {
        f"summary/global/{feature}": violin_box_plot(
            categories=categories,
            values=summary_df[feature].values,
            x_label="outcome",
            y_label=feature,
            category_order=["correct", "failed"],
        )
        for feature in features
    }


def _plot_rf_evaluation(classifier_results: dict) -> dict[str, Plot]:
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
            figsize=(6, 5),
        ),
        "summary/correlation/roc_curve": roc_curve_plot(
            fpr=np.array(roc_data["fpr"]),
            tpr=np.array(roc_data["tpr"]),
            auc_score=classifier_results["roc_auc"],
        ),
        "summary/correlation/feature_importances": feature_importance_plot(
            importances=top_importances,
            figsize=(12, max(8, max_features_to_plot * 0.5)),
        ),
    }


def _plot_class_cluster_tsne(
    X_num: np.ndarray,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    df_meta: dict,
    noise_cluster_ids: list[int],
    tsne_max_samples: int,
) -> dict[str, Plot]:
    """t-SNE scatter per class, colored by cluster; title shows class name and cluster count."""
    label_mapping: dict = df_meta.get("label_mapping", {})
    noise_set = set(noise_cluster_ids)
    rng = np.random.default_rng(42)

    result: dict[str, Plot] = {}
    for class_id in np.unique(y_class):
        class_mask = y_class == class_id
        X_c = X_num[class_mask]
        y_clust_c = y_cluster[class_mask]
        if len(X_c) < 6:
            continue
        unique_c = [c for c in np.unique(y_clust_c) if c != -1]
        if len(X_c) > tsne_max_samples and unique_c:
            per_cluster = max(1, tsne_max_samples // len(unique_c))
            idx_parts = []
            for cid in unique_c:
                cid_idx = np.where(y_clust_c == cid)[0]
                take = min(len(cid_idx), per_cluster)
                idx_parts.append(rng.choice(cid_idx, take, replace=False))
            noise_idx = np.where(y_clust_c == -1)[0]
            if len(noise_idx):
                remaining = max(0, tsne_max_samples - sum(len(p) for p in idx_parts))
                if remaining:
                    idx_parts.append(
                        rng.choice(
                            noise_idx, min(len(noise_idx), remaining), replace=False
                        )
                    )
            idx = np.concatenate(idx_parts)
            X_c = X_c[idx]
            y_clust_c = y_clust_c[idx]
        n_clusters = len([c for c in np.unique(y_clust_c) if c != -1])
        class_name = str(label_mapping.get(int(class_id), int(class_id)))
        title = f"{class_name}  ·  {n_clusters} clusters"
        perplexity = min(30.0, max(5.0, len(X_c) / 5))
        embedding = TSNE(
            n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1
        ).fit_transform(X_c)
        result[f"analysis/cluster_tsne/class_{class_name}"] = cluster_tsne_scatter(
            embedding, y_clust_c, noise_set, title=title
        )
    return result


def _plot_class_pairs_tsne(
    X_num: np.ndarray,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    cluster_summary: dict,
    df_meta: dict,
    paths: "OutputPaths",
    noise_cluster_ids: list[int],
    failure_threshold: float,
    max_class_pairs: int,
    pair_tsne_max_samples: int,
) -> dict[str, Plot]:
    """t-SNE scatter for top-K confused class pairs from the test confusion matrix."""
    cm_path = paths.pickle / "analysis/confusion_matrices/test.pkl"
    if not cm_path.exists():
        logger.warning(
            "Confusion matrix not found at %s; skipping class-pair t-SNE plots.",
            cm_path,
        )
        return {}

    with open(cm_path, "rb") as f:
        cm: np.ndarray = pkl.load(f)

    label_mapping: dict = df_meta.get("label_mapping", {})
    n_classes = cm.shape[0]

    pair_scores: dict[tuple[int, int], float] = {}
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            pair_scores[key] = pair_scores.get(key, 0.0) + float(cm[i, j])

    top_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)[
        :max_class_pairs
    ]

    failed_clusters = {
        int(cid)
        for cid, row in cluster_summary.items()
        if row.get("failure_rate") is not None
        and row["failure_rate"] > failure_threshold
    }
    noise_set = set(noise_cluster_ids)
    rng = np.random.default_rng(42)
    result: dict[str, Plot] = {}

    for (cls_a, cls_b), _ in top_pairs:
        pair_mask = (y_class == cls_a) | (y_class == cls_b)
        X_p = X_num[pair_mask]
        y_clust_p = y_cluster[pair_mask]
        y_cls_p = y_class[pair_mask]
        if len(X_p) < 6:
            continue
        if len(X_p) > pair_tsne_max_samples:
            half = pair_tsne_max_samples // 2
            idx_parts = []
            for cls in (cls_a, cls_b):
                cls_idx = np.where(y_cls_p == cls)[0]
                cls_clusters = [c for c in np.unique(y_clust_p[cls_idx]) if c != -1]
                if cls_clusters:
                    per_clu = max(1, half // len(cls_clusters))
                    for cid in cls_clusters:
                        cid_idx = cls_idx[y_clust_p[cls_idx] == cid]
                        take = min(len(cid_idx), per_clu)
                        idx_parts.append(rng.choice(cid_idx, take, replace=False))
                else:
                    take = min(len(cls_idx), half)
                    idx_parts.append(rng.choice(cls_idx, take, replace=False))
            idx = np.concatenate(idx_parts)
            X_p = X_p[idx]
            y_clust_p = y_clust_p[idx]
            y_cls_p = y_cls_p[idx]
        name_a = str(label_mapping.get(int(cls_a), int(cls_a)))
        name_b = str(label_mapping.get(int(cls_b), int(cls_b)))
        perplexity = min(30.0, max(5.0, len(X_p) / 5))
        embedding = TSNE(
            n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1
        ).fit_transform(X_p)
        result[f"analysis/class_pairs/{name_a}__vs__{name_b}"] = (
            class_pair_tsne_scatter(
                embedding,
                y_clust_p,
                y_cls_p,
                failed_clusters,
                noise_set,
                class_names={cls_a: name_a, cls_b: name_b},
            )
        )
    return result


def assemble_analysis_figures(
    paths: OutputPaths,
    X_num: np.ndarray,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    noise_cluster_ids: list[int],
    failure_threshold: float,
    max_class_pairs: int = 10,
    tsne_max_samples: int = 5000,
    pair_tsne_max_samples: int = 5000,
    cluster_summary: dict | None = None,
    df_meta: dict | None = None,
    classifier_results: dict | None = None,
    analysis_bus: LogDispatcher | None = None,
    step: int = 0,
) -> dict[str, Plot]:
    """Build all analysis figures, optionally loading inputs from disk and publishing to log bus."""
    if cluster_summary is None:
        cluster_summary = load_from_json(
            paths.json_logs / "analysis/cluster_summary.json"
        )
    if df_meta is None:
        df_meta = load_from_json(paths.json_logs / "data/df_meta.json")
    if classifier_results is None:
        classifier_results = load_from_json(
            paths.json_logs / "analysis/classifier_results.json"
        )
    logger.info("Building summary visualizations ...")
    summary_df = pd.DataFrame.from_dict(cluster_summary, orient="index")
    summary_df["class_name"] = summary_df["cluster_class"].map(df_meta["label_mapping"])

    oof_preds = classifier_results.get("oof_predictions", {})
    rf_correct = np.array(
        [oof_preds.get(str(cid), np.nan) for cid in summary_df.index], dtype=float
    )

    figures: dict[str, Plot] = {}
    figures.update(_plot_failure_strips(summary_df, rf_correct))
    figures.update(_plot_feature_by_outcome(summary_df, RELEVANT_GEOMETRIC_FEATURES))
    figures.update(_plot_rf_evaluation(classifier_results))
    figures.update(
        _plot_class_cluster_tsne(
            X_num,
            y_class,
            y_cluster,
            df_meta,
            noise_cluster_ids,
            tsne_max_samples,
        )
    )
    figures.update(
        _plot_class_pairs_tsne(
            X_num,
            y_class,
            y_cluster,
            cluster_summary,
            df_meta,
            paths,
            noise_cluster_ids,
            failure_threshold,
            max_class_pairs,
            pair_tsne_max_samples,
        )
    )
    if analysis_bus is not None:
        analysis_bus.publish(LogBundle(figures=figures, step=step))
    return figures


def run_complexity(
    paths: OutputPaths,
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    centroids: dict,
    noise_cluster_ids: list[int],
    k: int,
    top_k_clusters: int,
    min_subsample_per_cluster: int,
    max_complexity_samples: int | None,
    topk_metric: str,
    analysis_bus: LogDispatcher,
) -> tuple[dict, dict]:
    """Compute complexity measures, build cluster summary, publish to log bus."""
    pred_infos = load_from_json(paths.json_logs / "analysis/predictions/test.json")
    df_meta = load_from_json(paths.json_logs / "data/df_meta.json")

    logger.info("Computing complexity measures ...")
    complexity = compute_all_complexity_measures(
        X_num,
        X_cat,
        y_class,
        y_cluster,
        centroids,
        k=k,
        top_k_clusters=top_k_clusters,
        max_samples=max_complexity_samples,
        min_per_cluster=min_subsample_per_cluster,
        topk_metric=topk_metric,
    )

    cluster_errors = pred_infos["clusters"]["global"]

    cluster_to_class: dict[str, int] = {}
    for cid in np.unique(y_cluster):
        if cid == -1:
            continue
        mask = y_cluster == cid
        cluster_to_class[str(cid)] = int(y_class[mask][0])

    noise_set: set[int] = set(noise_cluster_ids)
    cluster_summary = {}
    for cid, measures in complexity.items():
        error_entry = (cluster_errors or {}).get(str(cid), {})
        failure_rate = error_entry.get("error_rate")
        cluster_summary[str(cid)] = {
            **measures,
            "cluster_class": cluster_to_class.get(str(cid)),
            "failure_rate": failure_rate,
            "is_failed": failure_rate is not None and failure_rate > 0.0,
            "is_noise_cluster": int(int(cid) in noise_set),
        }

    analysis_bus.publish(
        LogBundle.from_dict({"json/analysis/cluster_summary": cluster_summary})
    )
    logger.info("Cluster summary published.")
    return cluster_summary, df_meta


@timed
def analyze(
    paths: OutputPaths,
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    centroids: dict,
    step: int,
    failure_threshold: float,
    k: int,
    top_k_clusters: int,
    min_subsample_per_cluster: int,
    max_complexity_samples: int | None = None,
    noise_cluster_ids: list[int] | None = None,
    topk_metric: str = "euclidean",
    max_class_pairs: int = 10,
    tsne_max_samples: int = 5000,
    pair_tsne_max_samples: int = 5000,
) -> None:
    """Run data analysis pipeline."""
    logger.info("Starting analysis pipeline ...")
    noise_ids = noise_cluster_ids or []
    analysis_bus = LogDispatcher()
    tb_logger = TensorboardLogger(log_dir=paths.tb_logs / "analysis")
    analysis_bus.subscribe(TensorBoardSubscriber(tb_logger.writer))
    analysis_bus.subscribe(JSONSubscriber(paths.json_logs))
    cluster_summary, df_meta, classifier_results = None, None, None

    try:
        # cluster_summary, df_meta = run_complexity(
        #     paths,
        #     X_num,
        #     X_cat,
        #     y_class,
        #     y_cluster,
        #     centroids,
        #     noise_ids,
        #     k,
        #     top_k_clusters,
        #     min_subsample_per_cluster,
        #     max_complexity_samples,
        #     topk_metric,
        #     analysis_bus,
        # )
        # classifier_results = fit_failure_classifier(
        #     paths,
        #     RF_PARAM_GRID,
        #     cluster_stats=cluster_summary,
        #     failure_threshold=failure_threshold,
        #     analysis_bus=analysis_bus,
        # )
        assemble_analysis_figures(
            paths=paths,
            X_num=X_num,
            y_class=y_class,
            y_cluster=y_cluster,
            noise_cluster_ids=noise_ids,
            failure_threshold=failure_threshold,
            max_class_pairs=max_class_pairs,
            tsne_max_samples=tsne_max_samples,
            pair_tsne_max_samples=pair_tsne_max_samples,
            cluster_summary=cluster_summary,
            df_meta=df_meta,
            classifier_results=classifier_results,
            analysis_bus=analysis_bus,
            step=step,
        )
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
        pickle=Path(cfg.path.pickle),
    )
    save_config(cfg, paths.configs / "config_composed.json")

    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    ext = cfg.data.extension

    train_df = load_df(str(paths.processed_data / f"train.{ext}"))
    val_df = load_df(str(paths.processed_data / f"val.{ext}"))
    test_df = load_df(str(paths.processed_data / f"test.{ext}"))
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

    X_num = (
        combined[num_cols].to_numpy(dtype=np.float64)
        if num_cols
        else np.empty((len(combined), 0))
    )
    X_cat = combined[cat_cols].to_numpy() if cat_cols else None
    y_class = combined[f"encoded_{cfg.data.label_col}"].to_numpy(dtype=np.int64)
    y_cluster = combined["cluster"].to_numpy(dtype=np.int64)

    clusters_meta = load_from_json(Path(cfg.path.json_logs) / "data/clusters_meta.json")
    centroids = clusters_meta.get("centroids", {})
    noise_cluster_ids = clusters_meta.get("noise_cluster_ids", [])

    analyze(
        paths=paths,
        X_num=X_num,
        X_cat=X_cat,
        y_class=y_class,
        y_cluster=y_cluster,
        centroids=centroids,
        step=cfg.run_id or 0,
        failure_threshold=cfg.analysis.failure_threshold or 0.0,
        k=cfg.complexity.k,
        top_k_clusters=cfg.complexity.top_k_clusters,
        min_subsample_per_cluster=cfg.complexity.min_subsample_per_cluster,
        max_complexity_samples=cfg.complexity.max_complexity_samples,
        noise_cluster_ids=noise_cluster_ids,
        topk_metric=cfg.complexity.topk_metric,
        max_class_pairs=cfg.plots.max_class_pairs,
        tsne_max_samples=cfg.plots.tsne_max_samples,
        pair_tsne_max_samples=cfg.plots.pair_tsne_max_samples,
    )
    flush_timing(paths.json_logs / "timing.json")


if __name__ == "__main__":
    main()
