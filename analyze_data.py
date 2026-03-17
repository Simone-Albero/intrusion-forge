import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import save_to_json, load_from_json, load_from_pickle

from src.data.io import load_listed_dfs
from src.data.analyze import compute_class_separability

setup_logger()
logger = logging.getLogger(__name__)


def analyze(cfg):
    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    feature_cols = num_cols + cat_cols

    logger.info("Loading data ...")
    train_df, val_df, test_df = load_listed_dfs(
        Path(cfg.path.processed_data),
        [
            f"train.{cfg.data.extension}",
            f"val.{cfg.data.extension}",
            f"test.{cfg.data.extension}",
        ],
    )

    separability_results = {}
    label_cols = [f"cluster"]

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        if split_name == "val":
            continue

        for label_col in label_cols:
            logger.info(f"Computing separability on {split_name} set ...")
            X = split_df[feature_cols].values
            y = split_df[label_col].values
            separability_result = compute_class_separability(
                X, y, max_pairs=50_000, metric="cosine"
            )
            save_to_json(
                separability_result,
                Path(cfg.path.json_logs)
                / f"data/separability/{label_col}_{split_name}.json",
            )
            if label_col not in separability_results:
                separability_results[label_col] = {}
            separability_results[label_col][split_name] = separability_result

    return separability_results


def _to_finite_float(value) -> float | None:
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def compute_cluster_class_stats(
    cluster_separability: dict,
    class_to_clusters: dict,
) -> dict:
    """
    For each cluster in cluster_separability, computes class-wise stats
    against clusters of each class:
    - mean_ratio
    - max_ratio

    NaN/inf/non-numeric values are ignored; returns None when no valid
    ratios are available for a class.
    """
    class_clusters = {
        str(cls): {str(c) for c in clusters}
        for cls, clusters in class_to_clusters.items()
    }

    result = {}
    for cluster_id, ratios in cluster_separability.items():
        cid = str(cluster_id)

        peer_ratios = {}
        for k, v in ratios.items():
            if k == "_mean_ratio":
                continue
            fv = _to_finite_float(v)
            if fv is not None:
                peer_ratios[str(k)] = fv

        class_stats = {}
        for cls, cls_cluster_ids in class_clusters.items():
            relevant = [
                peer_ratios[c] for c in cls_cluster_ids if c in peer_ratios and c != cid
            ]
            class_stats[cls] = {
                "mean_ratio": float(np.mean(relevant)) if relevant else None,
                "max_ratio": float(np.max(relevant)) if relevant else None,
            }

        result[cid] = class_stats

    return result


def build_cluster_statistics(
    test_failures: dict,
    cluster_meta: dict,
    cluster_separability: dict,
    cluster_class_avgs: dict,
) -> dict:
    """
    For each cluster cid, returns:
    - cluster_class
    - failure_rate
    - is_failed (boolean, whether failure_rate > 0)
    - cluster_size
    - foreign_avg_avg: average of mean_ratios against clusters of other classes
    - foreign_max_avg: average of max_ratios against clusters of other classes
    - foreign_avg_max: max of mean_ratios against clusters of other classes
    - foreign_max_max: max of max_ratios against clusters of other classes
    - foreign_avg_std: std of mean_ratios against clusters of other classes
    - foreign_max_std: std of max_ratios against clusters of other classes
    - self_avg: mean_ratio against clusters of the same class
    - self_max: max_ratio against clusters of the same class
    - separability_distances: dict of distances to other clusters (excluding self), if available
    """
    results = {}

    for cid in cluster_meta["clusters_distribution"].keys():
        cluster_to_class = {
            c: cls
            for cls, clusters in cluster_meta["class_to_clusters"].items()
            for c in clusters
        }
        cluster_class = cluster_to_class.get(str(cid), None)

        cluster_size = cluster_meta["clusters_distribution"].get(str(cid), None)

        class_avgs = cluster_class_avgs.get(str(cid), {})

        foreign_avgs = []
        foreign_maxs = []
        self_avg = None
        self_max = None

        for cls, stats in class_avgs.items():
            if cls == cluster_class:
                self_avg = stats.get("mean_ratio", None)
                self_max = stats.get("max_ratio", None)
            else:
                if stats.get("mean_ratio", None) is not None:
                    foreign_avgs.append(stats["mean_ratio"])
                if stats.get("max_ratio", None) is not None:
                    foreign_maxs.append(stats["max_ratio"])

        foreign_avg_avg = float(np.mean(foreign_avgs)) if foreign_avgs else None
        foreign_max_avg = float(np.mean(foreign_maxs)) if foreign_maxs else None
        foreign_avg_max = float(np.max(foreign_avgs)) if foreign_avgs else None
        foreign_max_max = float(np.max(foreign_maxs)) if foreign_maxs else None
        foreign_avg_std = float(np.std(foreign_avgs)) if foreign_avgs else None
        foreign_max_std = float(np.std(foreign_maxs)) if foreign_maxs else None

        failure_rate = (
            test_failures["cluster_errors"]["total"]
            .get(str(cid), {})
            .get("error_rate", None)
        )
        is_failed = failure_rate is not None and failure_rate > 0.0

        separability_distances = cluster_separability.get(str(cid), {})
        separability_distances = {
            k: _to_finite_float(v)
            for k, v in separability_distances.items()
            if k != "_mean_ratio"
        }
        separability_distances[cid] = 1.0

        results[str(cid)] = {
            "cluster_class": cluster_class,
            "failure_rate": failure_rate,
            "is_failed": is_failed,
            "cluster_size": cluster_size,
            "foreign_avg_avg": foreign_avg_avg,
            "foreign_max_avg": foreign_max_avg,
            "foreign_avg_max": foreign_avg_max,
            "foreign_max_max": foreign_max_max,
            "foreign_avg_std": foreign_avg_std,
            "foreign_max_std": foreign_max_std,
            "self_avg": self_avg,
            "self_max": self_max,
            **(separability_distances if separability_distances else {}),
        }
    return results


def main():
    """Main entry point for data analysis."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    # analyze(cfg)

    test_failures = load_from_json(
        Path(cfg.path.json_logs) / "inference/pred_infos/test.json"
    )
    cluster_separability = load_from_json(
        Path(cfg.path.json_logs) / "data/separability/cluster_test.json"
    )
    clusters_meta = load_from_json(Path(cfg.path.json_logs) / "data/df_meta.json")[
        "clusters"
    ]

    cluster_class_stats = compute_cluster_class_stats(
        cluster_separability, clusters_meta["class_to_clusters"]
    )
    save_to_json(
        cluster_class_stats,
        Path(cfg.path.json_logs) / "data/separability/cluster_class_stats.json",
    )

    cluster_stats = build_cluster_statistics(
        test_failures=test_failures,
        cluster_meta=clusters_meta,
        cluster_separability=cluster_separability,
        cluster_class_avgs=cluster_class_stats,
    )

    save_to_json(
        cluster_stats,
        Path(cfg.path.json_logs) / "inference/cluster_statistics.json",
    )

    cluster_stats_df = pd.DataFrame.from_dict(cluster_stats, orient="index")
    target_col = "failure_rate"
    exclude_cols = [target_col, "is_failed", "cluster_class"]
    # feature_cols = [
    #     "cluster_size",
    #     "foreign_avg_avg",
    #     "foreign_max_avg",
    #     "foreign_avg_max",
    #     "foreign_max_max",
    #     "foreign_avg_std",
    #     "foreign_max_std",
    #     "self_avg",
    #     "self_max",
    # ]

    X = cluster_stats_df.drop(columns=exclude_cols, errors="ignore")
    # X = X[feature_cols].copy()
    y = cluster_stats_df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train size:", X_train.shape, y_train.shape)
    print("Test size:", X_test.shape, y_test.shape)

    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.5],
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Best params:", grid.best_params_)
    print("Target Mean:", y.mean())
    print("Target Std:", y.std())
    print("Test MAE   :", mean_absolute_error(y_test, y_pred))
    print("Test R2    :", r2_score(y_test, y_pred))


if __name__ == "__main__":
    main()
