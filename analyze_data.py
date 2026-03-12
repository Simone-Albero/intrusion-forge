import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


def analyze_misclassification_ratios(
    test_failures: dict,
    cluster_separability: dict,
    class_to_clusters: dict,
) -> tuple[dict, float | None, int, int]:
    """
    For each triple (true_class, mispredicted_class, misclassified_cluster), computes:
      1. rank: 1-based rank (1 = highest ratio) of the best-separability cluster of
               mispredicted_class in the ratio dict of misclassified_cluster.
      2. ratio_diff: avg ratio to true_class clusters - avg ratio to mispredicted_class clusters.

    Also returns:
      avg_rank: mean rank across all triples.
      num_negative_diffs: count of triples where ratio_diff < 0.
      tot_diffs: total number of triples with a valid ratio_diff.
    """
    results = {}

    for true_class, failure_info in test_failures.items():
        true_clusters = [str(c) for c in class_to_clusters.get(str(true_class), [])]

        for mispredicted_class, misclassified_clusters in failure_info[
            "clusters_in_failures"
        ].items():
            mispred_clusters = set(
                str(c) for c in class_to_clusters.get(str(mispredicted_class), [])
            )

            for misclassified_cluster in misclassified_clusters:
                mc = str(misclassified_cluster)
                if mc not in cluster_separability:
                    continue

                ratios = {
                    k: v
                    for k, v in cluster_separability[mc].items()
                    if k != "_mean_ratio"
                }

                # --- Metric 1: rank of the best (highest-ratio) mispredicted-class cluster ---
                sorted_desc = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
                rank = next(
                    (
                        r
                        for r, (cid, _) in enumerate(sorted_desc, start=1)
                        if cid in mispred_clusters
                    ),
                    None,
                )

                # --- Metric 2: avg ratio to true class vs avg ratio to mispredicted class ---
                true_ratios = [
                    ratios[c] for c in true_clusters if c in ratios and c != mc
                ]
                mispred_ratios = [ratios[c] for c in mispred_clusters if c in ratios]

                avg_true = float(np.mean(true_ratios)) if true_ratios else None
                avg_mispred = float(np.mean(mispred_ratios)) if mispred_ratios else None
                ratio_diff = (
                    (avg_true - avg_mispred)
                    if (avg_true is not None and avg_mispred is not None)
                    else None
                )

                results.setdefault(true_class, {}).setdefault(mispredicted_class, {})[
                    mc
                ] = {
                    "rank": rank,
                    "ratio_diff": ratio_diff,
                    "avg_true_ratio": avg_true,
                    "avg_mispred_ratio": avg_mispred,
                }

    all_values = [
        v
        for per_mispred in results.values()
        for per_cluster in per_mispred.values()
        for v in per_cluster.values()
    ]

    ranks = [v["rank"] for v in all_values if v["rank"] is not None]
    diffs = [v["ratio_diff"] for v in all_values if v["ratio_diff"] is not None]

    avg_rank = float(np.mean(ranks)) if ranks else None
    num_negative_diffs = sum(1 for d in diffs if d < 0)
    tot_diffs = len(diffs)

    return results, avg_rank, num_negative_diffs, tot_diffs


def main():
    """Main entry point for data analysis."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    # analyze(cfg)

    test_failures = load_from_json(
        Path(cfg.path.json_logs) / "inference/class_failures/test.json"
    )
    cluster_separability = load_from_json(
        Path(cfg.path.json_logs) / "data/separability/cluster_test.json"
    )
    class_to_clusters = load_from_json(Path(cfg.path.json_logs) / "data/df_meta.json")[
        "clusters"
    ]["class_to_clusters"]

    results, avg_rank, num_negative_diffs, tot_diffs = analyze_misclassification_ratios(
        test_failures, cluster_separability, class_to_clusters
    )

    save_to_json(
        {
            "results": results,
            "avg_rank": avg_rank,
            "num_negative_diffs": num_negative_diffs,
            "tot_diffs": tot_diffs,
        },
        Path(cfg.path.json_logs) / "inference/misclassification_ratios.json",
    )


if __name__ == "__main__":
    main()
