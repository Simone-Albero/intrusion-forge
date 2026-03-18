import logging
import sys
from pathlib import Path

import numpy as np
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
from src.data.analyze import compute_class_separability

setup_logger()
logger = logging.getLogger(__name__)


def run_separability_analysis(cfg):
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
        logger.info(f"Computing separability on {split_name} set ...")
        X = split_df[feature_cols].values
        y = split_df["cluster"].values
        sep = compute_class_separability(X, y, max_pairs=50_000, metric="euclidean")
        save_to_json(
            sep,
            Path(cfg.path.json_logs) / f"data/separability/cluster_{split_name}.json",
        )
        results.setdefault("cluster", {})[split_name] = sep

    return results


def _to_finite_float(value) -> float | None:
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def compute_class_stats(separability: dict, class_to_clusters: dict) -> dict:
    """
    For each cluster, compute mean_ratio and max_ratio against clusters of each class.
    Non-finite values are ignored; returns None when no valid ratios exist for a class.
    """
    class_clusters = {
        str(cls): {str(c) for c in clusters}
        for cls, clusters in class_to_clusters.items()
    }

    result = {}
    for cid, ratios in separability.items():
        peer_ratios = {
            str(k): fv
            for k, v in ratios.items()
            if k != "_mean_ratio" and (fv := _to_finite_float(v)) is not None
        }

        result[str(cid)] = {
            cls: {
                "mean_ratio": (
                    float(np.mean(vals))
                    if (
                        vals := [
                            peer_ratios[c]
                            for c in ids
                            if c in peer_ratios and c != str(cid)
                        ]
                    )
                    else None
                ),
                "max_ratio": float(np.max(vals)) if vals else None,
            }
            for cls, ids in class_clusters.items()
        }

    return result


def build_cluster_stats(
    failures: dict, meta: dict, separability: dict, class_stats: dict
) -> dict:
    """
    For each cluster, return separability stats, failure rate, and class membership.

    Fields: cluster_class, failure_rate, is_failed, cluster_size,
            foreign_{avg,max}_{avg,max,std}, self_{avg,max},
            separability_distances (per-cluster).
    """
    cluster_to_class = {
        str(c): cls
        for cls, clusters in meta["class_to_clusters"].items()
        for c in clusters
    }

    results = {}
    for cid in meta["clusters_distribution"]:
        cid = str(cid)
        cluster_class = cluster_to_class.get(cid)
        cluster_size = meta["clusters_distribution"].get(cid)

        foreign_avgs, foreign_maxs = [], []
        self_avg = self_max = None

        for cls, stats in class_stats.get(cid, {}).items():
            if cls == cluster_class:
                self_avg = stats.get("mean_ratio")
                self_max = stats.get("max_ratio")
            else:
                if (v := stats.get("mean_ratio")) is not None:
                    foreign_avgs.append(v)
                if (v := stats.get("max_ratio")) is not None:
                    foreign_maxs.append(v)

        distances = {
            k: _to_finite_float(v)
            for k, v in separability.get(cid, {}).items()
            if k != "_mean_ratio"
        }
        distances[cid] = 1.0

        failure_rate = failures["clusters"]["total"].get(cid, {}).get("error_rate")

        results[cid] = {
            "cluster_class": cluster_class,
            "failure_rate": failure_rate,
            "is_failed": failure_rate is not None and failure_rate > 0.0,
            "cluster_size": cluster_size,
            "foreign_avg_avg": float(np.mean(foreign_avgs)) if foreign_avgs else None,
            "foreign_max_avg": float(np.mean(foreign_maxs)) if foreign_maxs else None,
            "foreign_avg_max": float(np.max(foreign_avgs)) if foreign_avgs else None,
            "foreign_max_max": float(np.max(foreign_maxs)) if foreign_maxs else None,
            "foreign_avg_std": float(np.std(foreign_avgs)) if foreign_avgs else None,
            "foreign_max_std": float(np.std(foreign_maxs)) if foreign_maxs else None,
            "self_avg": self_avg,
            "self_max": self_max,
            **distances,
        }

    return results


def find_correlation_between_separability_and_failure(cluster_stats: dict) -> dict:
    FEATURE_COLS = [
        "cluster_size",
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

    df = pd.DataFrame.from_dict(cluster_stats, orient="index")
    X = df[FEATURE_COLS].copy()
    y = df["is_failed"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape}  Test: {X_test.shape}")

    grid = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        param_grid=RF_PARAM_GRID,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    logger.info("Best params:", grid.best_params_)
    logger.info(
        f"F1: {f1_score(y_test, y_pred):.4f}  ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}"
    )
    logger.info(
        f"F1: {f1_score(y_test, y_pred):.4f}  ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}"
    )
    logger.info("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    logger.info(
        "\nClassification Report:\n", classification_report(y_test, y_pred, digits=4)
    )
    return {
        "best_params": grid.best_params_,
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, digits=4, output_dict=True
        ),
    }


def main():
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    logs = Path(cfg.path.json_logs)
    run_separability_analysis(cfg)

    # failures = load_from_json(logs / "inference/pred_infos/test.json")
    # separability = load_from_json(logs / "data/separability/cluster_test.json")
    # cluster_meta = load_from_json(logs / "data/df_meta.json")["clusters"]

    # class_stats = compute_class_stats(separability, cluster_meta["class_to_clusters"])
    # save_to_json(class_stats, logs / "data/separability/class.json")

    # cluster_stats = build_cluster_stats(
    #     failures=failures,
    #     meta=cluster_meta,
    #     separability=separability,
    #     class_stats=class_stats,
    # )
    # save_to_json(cluster_stats, logs / "inference/cluster_summary.json")

    # correlation_results = find_correlation_between_separability_and_failure(
    #     cluster_stats
    # )
    # save_to_json(correlation_results, logs / "inference/correlation_results.json")


if __name__ == "__main__":
    main()
