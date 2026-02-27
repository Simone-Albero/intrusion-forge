import logging
import sys
from pathlib import Path

import numpy as np
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
    label_cols = (
        [f"encoded_{cfg.data.label_col}"]
        if "cluster" not in train_df.columns
        else [f"encoded_{cfg.data.label_col}", "cluster"]
    )
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        for label_col in label_cols:
            logger.info(f"Computing separability on {split_name} set ...")
            X = split_df[feature_cols].values
            y = split_df[label_col].values
            separability_result = compute_class_separability(
                X, y, max_pairs=None, metric="cosine"
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
            Path(cfg.path.json_logs) / f"inference/confusion_matrices/{split_name}.pkl"
        )

        cm = np.asarray(inference_cm, dtype=float)

        split_scores = []
        for entry in separability_results:
            class_idx = int(entry["class"])
            pairs = entry["pairs"]

            if len(pairs) < 2:
                continue

            pair_indices = [int(j) for j in pairs]
            ratios = np.array([pairs[str(j)]["ratio"] for j in pair_indices])
            misclassifications = cm[class_idx, pair_indices]

            corr, pvalue = spearmanr(ratios, misclassifications)
            split_scores.append(
                {
                    "class": class_idx,
                    "score": float(corr),
                    "pvalue": float(pvalue),
                }
            )
            logger.info(
                f"[{split_name}] Class {class_idx}: Spearman r={corr:.4f} (p={pvalue:.4f})"
            )

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


if __name__ == "__main__":
    main()
