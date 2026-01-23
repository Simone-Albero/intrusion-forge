import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import paired_distances

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import save_to_json
from src.data.io import load_data_splits

setup_logger()
logger = logging.getLogger(__name__)


def load_data(processed_data_path, file_name, extension):
    """Load and prepare data for analysis."""
    train_df, val_df, test_df = load_data_splits(
        processed_data_path, file_name, extension
    )
    return train_df, val_df, test_df


def sample_distances(X, idx_a, idx_b=None, n_pairs=200_000):
    """Sample pairwise distances between points."""
    idx_a = np.asarray(idx_a)

    if idx_b is None:  # Intra-class distances
        if len(idx_a) < 2:
            return np.array([])
        i = np.random.randint(0, len(idx_a), size=n_pairs)
        j = np.random.randint(0, len(idx_a), size=n_pairs)
        mask = i != j
        return paired_distances(X[idx_a[i[mask]]], X[idx_a[j[mask]]])
    else:  # Inter-class distances
        idx_b = np.asarray(idx_b)
        if len(idx_a) == 0 or len(idx_b) == 0:
            return np.array([])
        n_samples = min(n_pairs, len(idx_a), len(idx_b))
        i = np.random.choice(idx_a, size=n_samples, replace=False)
        j = np.random.choice(idx_b, size=n_samples, replace=False)
        return paired_distances(X[i], X[j])


def run_separability_analysis(df, label_col, feature_cols):
    """Analyze class separability using intra/inter-class distances."""
    y = df[label_col].astype(str).values
    X = df[feature_cols].values

    results = []
    for class_name in np.unique(y):
        logger.info(f"Analyzing class {class_name}...")

        idx_class = np.where(y == class_name)[0]
        idx_other = np.where(y != class_name)[0]

        intra_dist = sample_distances(X, idx_class)
        inter_dist = sample_distances(X, idx_class, idx_other)

        intra_mean = np.mean(intra_dist) if len(intra_dist) > 0 else np.nan
        inter_mean = np.mean(inter_dist) if len(inter_dist) > 0 else np.nan
        ratio = (
            intra_mean / inter_mean
            if np.isfinite(intra_mean) and np.isfinite(inter_mean)
            else np.nan
        )

        try:
            silhouette_score_value = silhouette_score(
                X, y == class_name, sample_size=min(50_000, len(X))
            )
        except ValueError:
            silhouette_score_value = np.nan

        results.append(
            {
                "class": class_name,
                "n_samples": len(idx_class),
                "intra_mean": float(intra_mean),
                "inter_mean": float(inter_mean),
                "ratio": float(ratio),
                "silhouette_score": float(silhouette_score_value),
            }
        )

    results.sort(key=lambda x: x["ratio"])
    return results


def compute_class_similarity(df, class_a, class_b, label_col, num_cols, cat_cols):
    """Compute feature-wise similarity between two classes."""
    df_a = df[df[label_col] == class_a]
    df_b = df[df[label_col] == class_b]

    results = []
    total_sim = 0.0
    n_features = 0

    for col in num_cols:
        mean_a, std_a = df_a[col].mean(), df_a[col].std()
        mean_b, std_b = df_b[col].mean(), df_b[col].std()

        similarity = 1 - abs(mean_a - mean_b) / (std_a + std_b + 1e-8)
        similarity = float(max(0.0, min(1.0, similarity)))

        results.append(
            {
                "feature": col,
                "class_a_mean": float(mean_a),
                "class_b_mean": float(mean_b),
                "class_a_top_category": None,
                "class_b_top_category": None,
                "similarity": similarity,
            }
        )
        total_sim += similarity
        n_features += 1

    for col in cat_cols:
        freq_a = df_a[col].value_counts(normalize=True)
        freq_b = df_b[col].value_counts(normalize=True)

        all_cats = set(freq_a.index).union(set(freq_b.index))
        intersection = sum(min(freq_a.get(c, 0), freq_b.get(c, 0)) for c in all_cats)
        union = sum(max(freq_a.get(c, 0), freq_b.get(c, 0)) for c in all_cats)

        similarity = float(intersection / union) if union > 0 else 0.0

        results.append(
            {
                "feature": col,
                "class_a_mean": None,
                "class_b_mean": None,
                "class_a_top_category": freq_a.idxmax() if not freq_a.empty else None,
                "class_b_top_category": freq_b.idxmax() if not freq_b.empty else None,
                "similarity": similarity,
            }
        )
        total_sim += similarity
        n_features += 1

    results.sort(key=lambda x: x["similarity"], reverse=True)
    global_similarity = float(total_sim / n_features) if n_features > 0 else 0.0
    return results, global_similarity


def main():
    """Main entry point for data analysis."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    feature_cols = num_cols + cat_cols
    label_col = cfg.data.label_col

    # Load data
    train_df, val_df, test_df = load_data(
        Path(cfg.path.processed_data), cfg.data.file_name, cfg.data.extension
    )

    for split_name, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        logger.info(f"Running analysis on {split_name} set...")

        logger.info("Computing class separability...")
        separability_results = run_separability_analysis(df, label_col, feature_cols)
        separability_df = pd.DataFrame(separability_results)
        save_to_json(
            separability_results,
            Path(cfg.path.json_logs) / f"{split_name}_separability.json",
        )

        logger.info(f"\n{separability_df}")
        global_silhouette = np.nanmean(
            [
                r["silhouette_score"]
                for r in separability_results
                if np.isfinite(r["silhouette_score"])
            ]
        )
        logger.info(f"Global silhouette score: {global_silhouette:.4f}")

    # logger.info("Computing class similarity...")
    # similarity_results, global_similarity = compute_class_similarity(
    #     df,
    #     class_a="Benign",
    #     class_b="Backdoor",
    #     label_col=label_col,
    #     num_cols=num_cols,
    #     cat_cols=cat_cols,
    # )
    # similarity_df = pd.DataFrame(similarity_results)
    # save_to_json(
    #     similarity_results,
    #     Path(cfg.path.json_logs) / f"{split_name}_similarity.json",
    # )

    # logger.info(f"\n{similarity_df}")
    # logger.info(f"Global similarity: {global_similarity:.4f}")


if __name__ == "__main__":
    main()
