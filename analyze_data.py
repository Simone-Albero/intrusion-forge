import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import paired_distances
from ignite.handlers.tensorboard_logger import TensorboardLogger

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import save_to_json, load_from_pickle
from src.data.io import load_data_splits
from src.ml.projection import tsne_projection, create_subsample_mask
from src.plot.array import vectors_plot

setup_logger()
logger = logging.getLogger(__name__)


def load_data(processed_data_path, file_name, extension):
    """Load and prepare data for analysis."""
    train_df, val_df, test_df = load_data_splits(
        processed_data_path, file_name, extension
    )
    return train_df, val_df, test_df


def sample_distances(X, idx_a, idx_b=None, n_pairs=200_000, metric="euclidean"):
    """Sample pairwise distances between points.

    Args:
        X: Feature matrix (n_samples, n_features)
        idx_a: Indices for first set of points
        idx_b: Indices for second set (None for intra-class distances)
        n_pairs: Maximum number of distance pairs to compute

    Returns:
        Array of pairwise distances between sampled pairs
    """
    idx_a = np.asarray(idx_a)

    if idx_b is None:  # Intra-class distances
        if len(idx_a) < 2:
            return np.array([])

        max_possible_pairs = len(idx_a) * (len(idx_a) - 1) // 2
        n_samples = min(n_pairs, max_possible_pairs)

        if n_samples < max_possible_pairs:
            i = np.random.randint(0, len(idx_a), size=n_samples)
            j = np.random.randint(0, len(idx_a), size=n_samples)
            # Ensure i != j by incrementing j and wrapping
            j = (
                j
                + 1
                + (j >= i).astype(int)
                * (np.random.randint(0, len(idx_a) - 1, size=n_samples))
            ) % len(idx_a)

            mask = i != j
            attempts = 0
            while np.sum(~mask) > 0 and attempts < 10:
                n_resample = np.sum(~mask)
                i[~mask] = np.random.randint(0, len(idx_a), size=n_resample)
                j[~mask] = np.random.randint(0, len(idx_a), size=n_resample)
                mask = i != j
                attempts += 1

            i, j = i[mask], j[mask]
        else:
            # compute all pairs
            from itertools import combinations

            pairs = list(combinations(range(len(idx_a)), 2))
            np.random.shuffle(pairs)
            pairs_array = np.array(pairs[:n_samples])
            i, j = pairs_array[:, 0], pairs_array[:, 1]

        return paired_distances(X[idx_a[i]], X[idx_a[j]], metric=metric)

    else:  # Inter-class distances
        idx_b = np.asarray(idx_b)
        if len(idx_a) == 0 or len(idx_b) == 0:
            return np.array([])

        max_possible_pairs = len(idx_a) * len(idx_b)
        n_samples = min(n_pairs, max_possible_pairs)

        need_replacement = n_samples > min(len(idx_a), len(idx_b))

        i = np.random.choice(len(idx_a), size=n_samples, replace=need_replacement)
        j = np.random.choice(len(idx_b), size=n_samples, replace=need_replacement)

        return paired_distances(X[idx_a[i]], X[idx_b[j]], metric=metric)


def run_separability_analysis(df, label_col, feature_cols):
    """Analyze class separability using intra/inter-class distances."""
    y = df[label_col].astype(str).values
    X = df[feature_cols].values

    results = []
    for class_name in np.unique(y):
        logger.info(f"Analyzing class {class_name} ...")

        idx_class = np.where(y == class_name)[0]
        idx_other = np.where(y != class_name)[0]

        intra_dist = sample_distances(X, idx_class)
        inter_dist = sample_distances(X, idx_class, idx_other)

        intra_mean = np.mean(intra_dist) if len(intra_dist) > 0 else np.nan
        inter_mean = np.mean(inter_dist) if len(inter_dist) > 0 else np.nan

        gap = (
            inter_mean - intra_mean
            if np.isfinite(intra_mean) and np.isfinite(inter_mean)
            else np.nan
        )
        ratio = (
            intra_mean / inter_mean
            if np.isfinite(intra_mean) and np.isfinite(inter_mean)
            else np.nan
        )

        try:
            sample_size = min(50_000, X.shape[0])
            sil = silhouette_score(X, (y == class_name), sample_size=sample_size)
        except ValueError:
            sil = np.nan

        results.append(
            {
                "class": class_name,
                "n_samples": int(idx_class.size),
                "intra_mean": float(intra_mean),
                "inter_mean": float(inter_mean),
                "gap": float(gap),
                "ratio": float(ratio),
                "silhouette_score": float(sil),
            }
        )

    results.sort(key=lambda x: x["ratio"])
    return results


def compute_distance_roc_auc(intra_a, intra_b, inter_ab):
    """
    Compute ROC-AUC where:
    - Label 1 (positive): inter-class pairs
    - Label 0 (negative): intra-class pairs
    - Score: distance (higher distance should predict inter-class)
    """

    from sklearn.metrics import roc_auc_score

    y_true = np.concatenate(
        [np.zeros(len(intra_a) + len(intra_b)), np.ones(len(inter_ab))]
    )
    y_scores = np.concatenate([intra_a, intra_b, inter_ab])

    if len(np.unique(y_true)) < 2:
        return np.nan
    # intra-class pairs should have lower distance than inter-class pairs
    # AUC 1 means all inter-class pairs have higher distance than intra-class pairs
    auc = roc_auc_score(y_true, y_scores)
    return auc


def compute_class_similarity(X, idx_a, idx_b, n_pairs=200_000):
    """Compute similarity between two classes based on distance distributions."""
    intra_a = sample_distances(X, idx_a, n_pairs=n_pairs)
    intra_b = sample_distances(X, idx_b, n_pairs=n_pairs)
    inter_ab = sample_distances(X, idx_a, idx_b, n_pairs=n_pairs)

    intra_a_mean = np.mean(intra_a) if len(intra_a) > 0 else np.nan
    intra_b_mean = np.mean(intra_b) if len(intra_b) > 0 else np.nan
    inter_ab_mean = np.mean(inter_ab) if len(inter_ab) > 0 else np.nan

    gap = (
        inter_ab_mean - (intra_a_mean + intra_b_mean) / 2
        if np.isfinite(intra_a_mean)
        and np.isfinite(intra_b_mean)
        and np.isfinite(inter_ab_mean)
        else np.nan
    )
    ratio = (
        (intra_a_mean + intra_b_mean) / 2 / inter_ab_mean
        if np.isfinite(intra_a_mean)
        and np.isfinite(intra_b_mean)
        and np.isfinite(inter_ab_mean)
        else np.nan
    )

    # probability that an inter-class pair is smaller than an intra-class pair
    p = np.mean(inter_ab[:, None] < intra_a[None, :])
    q = np.mean(inter_ab[:, None] < intra_b[None, :])
    roc_auc = compute_distance_roc_auc(intra_a, intra_b, inter_ab)

    return (
        {
            "intra_a_mean": float(intra_a_mean),  # low value means class A is compact
            "intra_b_mean": float(intra_b_mean),  # low value means class B is compact
            "inter_ab_mean": float(
                inter_ab_mean
            ),  # high value means classes are well separated
            "gap": float(
                gap
            ),  # negative gap means classes are closer to each other than they are internally
            "ratio": float(ratio),  # low ratio means good separation
            "overlap_a": float(p),  # high value means class A overlaps with class B
            "overlap_b": float(q),  # high value means class B overlaps with class A
            "roc_auc": float(roc_auc),  # high value means good separation
        },
    )


def compute_separability_analysis(
    train_df, val_df, test_df, label_col, feature_cols, cfg
):
    """Compute separability analysis for the dataset."""
    for split_name, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        logger.info(f"Running analysis on {split_name} set ...")

        logger.info("Computing class separability ...")
        separability_results = run_separability_analysis(df, label_col, feature_cols)
        separability_df = pd.DataFrame(separability_results)

        logger.info(f"\n{separability_df}")
        global_silhouette = np.nanmean(
            [
                r["silhouette_score"]
                for r in separability_results
                if np.isfinite(r["silhouette_score"])
            ]
        )
        logger.info(f"Global silhouette score: {global_silhouette:.4f}")

        separability_results.append({"global_silhouette": global_silhouette})
        save_to_json(
            separability_results,
            Path(cfg.path.json_logs) / f"separability/{split_name}_{label_col}.json",
        )


def compute_similarity_analysis(
    train_df, val_df, test_df, label_col, feature_cols, class_a, class_b, cfg
):
    """Compute similarity analysis between specified classes."""
    for split_name, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        logger.info(f"Computing similarity on {split_name} set ...")
        X = df[feature_cols].values
        idx_a = np.where(df[label_col].values == class_a)[0]
        idx_b = np.where(df[label_col].values == class_b)[0]
        similarity_result = compute_class_similarity(X, idx_a, idx_b)
        similarity_df = pd.DataFrame(similarity_result)
        logger.info(f"\n{similarity_df}")

        save_to_json(
            similarity_result,
            Path(cfg.path.json_logs)
            / f"similarity/{split_name}_{class_a}_vs_{class_b}.json",
        )


def visualize_overall(X, y, exclude_classes=[], n_samples=3000):
    """Create overall visualizations excluding specific class."""
    mask = ~np.isin(y, exclude_classes)
    vis_mask = create_subsample_mask(y[mask], n_samples=n_samples, stratify=False)
    reduced_x = tsne_projection(X[mask][vis_mask])
    return vectors_plot(reduced_x, y[mask][vis_mask])


def main():
    """Main entry point for data analysis."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    feature_cols = num_cols + cat_cols
    label_col = cfg.data.label_col

    # Load data
    logger.info("Loading data ...")
    train_df, val_df, test_df = load_data(
        Path(cfg.path.processed_data),
        cfg.data.file_name,
        cfg.data.extension,
    )

    logger.info("Starting separability analysis ...")
    compute_separability_analysis(
        train_df, val_df, test_df, label_col, feature_cols, cfg
    )
    compute_separability_analysis(
        train_df, val_df, test_df, "cluster", feature_cols, cfg
    )

    # logger.info("Starting similarity analysis ...")
    # compute_similarity_analysis(
    #     train_df, val_df, test_df, label_col, feature_cols, "DoS", "Exploits", cfg
    # )

    for suffix, df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        logger.info(f"Visualizing {suffix} set ...")
        log_dir = Path(cfg.path.tb_logs) / "visualize" / suffix
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_logger = TensorboardLogger(log_dir=log_dir)

        fig = visualize_overall(
            df[feature_cols].to_numpy(),
            df["multi_" + label_col].to_numpy(),
        )
        tb_logger.writer.add_figure("label_projection", fig, global_step=cfg.run_id)
        tb_logger.close()

        fig = visualize_overall(
            df[feature_cols].to_numpy(),
            df["cluster"].to_numpy(),
        )
        tb_logger.writer.add_figure("cluster_projection", fig, global_step=cfg.run_id)
        tb_logger.close()


if __name__ == "__main__":
    main()
