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
    # label_cols = (
    #     [f"encoded_{cfg.data.label_col}"]
    #     if "cluster" not in train_df.columns
    #     else [f"encoded_{cfg.data.label_col}", "cluster"]
    # )
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


def compute_results(cfg):
    """For each class, compute a score of how well the model's failures on that class
    correlate with the classes that are hardest to separate from it.

    A per-class score of 1 means the model confuses class c most with the classes
    that are hardest to separate from it — the ideal, explainable behaviour.
    """
    results = {}
    label_col = f"encoded_{cfg.data.label_col}"

    for split_name in ("train", "val", "test"):

        separability_results = load_from_json(
            Path(cfg.path.json_logs)
            / f"data/separability/{label_col}_{split_name}.json"
        )

        inference_cm = load_from_pickle(
            Path(cfg.path.pickle) / f"inference/confusion_matrices/{split_name}.pkl"
        )

        cm = np.asarray(inference_cm, dtype=float)

        split_scores = []
        for entry in separability_results:
            class_idx = int(entry["class"])

            pairs = entry["pairs"]

            pair_indices = [int(j) for j in pairs]
            ratios = np.array([pairs[str(j)]["ratio"] for j in pair_indices])
            misclassifications = cm[class_idx, pair_indices]

            ratios_norm = ratios / ratios.max() if ratios.max() > 0 else ratios

            class_total = cm[class_idx].sum()
            misclassifications_norm = misclassifications / class_total
            misclassifications_norm = (
                misclassifications_norm / misclassifications_norm.max()
                if misclassifications_norm.max() > 0
                else misclassifications_norm
            )

            valid_mask = misclassifications > 0
            ratios_norm = ratios_norm[valid_mask]
            misclassifications_norm = misclassifications_norm[valid_mask]

            if len(ratios_norm) < 2:
                split_scores.append(
                    {
                        "class": class_idx,
                        "tot_misclassifications": int(misclassifications.sum()),
                        "score": None,
                    }
                )
                continue

            corr, _ = spearmanr(ratios_norm, misclassifications_norm)

            rank_ratios = np.argsort(np.argsort(ratios_norm)) + 1
            rank_misc = np.argsort(np.argsort(misclassifications_norm)) + 1
            disagreements = np.sum(rank_misc != rank_ratios)

            logger.info(
                f"class={class_idx}, class_total={class_total:.0f}, "
                f"misclassifications={misclassifications_norm}\n"
                f"ratios={ratios_norm}\ndisagreements={disagreements}/{len(misclassifications)}, corr={corr:.4f}"
            )

            split_scores.append(
                {
                    "class": class_idx,
                    "tot_misclassifications": int(misclassifications.sum()),
                    "error_rate": (
                        float(misclassifications.sum() / class_total)
                        if class_total > 0
                        else 0.0
                    ),
                    "disagreements": int(disagreements),
                    "total_pairs": len(misclassifications),
                    "corr": float(corr),
                    "score": (1 - disagreements / len(misclassifications)),
                }
            )

        logger.info(f"Separability scores for {split_name}:")
        logger.info(pd.DataFrame(split_scores))

        results[split_name] = split_scores

    save_to_json(
        results, Path(cfg.path.json_logs) / "data/separability_failure_score.json"
    )
    return results


def main():
    """Main entry point for data analysis."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    analyze(cfg)
    # compute_results(cfg)


if __name__ == "__main__":
    main()
