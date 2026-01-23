import json
import logging
import sys
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import paired_distances
from sklearn.model_selection import train_test_split


from src.common.config import load_config
from src.common.logging import setup_logger

from src.data.io import load_data_splits, load_df, save_df
from src.data.preprocessing import (
    subsample_df,
    random_oversample_df,
    random_undersample_df,
    query_filter,
)

setup_logger()
logger = logging.getLogger(__name__)


def prepare_data(cfg, filter_query: str = None):
    """Prepare data for PyTorch."""
    logger.info("Preparing data for PyTorch...")

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    feature_cols = num_cols + cat_cols
    label_col = cfg.data.label_col

    base_path = Path(cfg.path.processed_data)
    train_df, _, test_df = load_data_splits(
        base_path, cfg.data.file_name, cfg.data.extension
    )

    train_df = query_filter(train_df, filter_query)
    test_df = query_filter(test_df, filter_query)

    logger.info("Data preparation for PyTorch completed.")
    return train_df[feature_cols + [label_col]], test_df[feature_cols + [label_col]]


def analyze_df(
    df: pd.DataFrame,
    cfg: Any,
) -> None:
    """Analyze dataframe and log basic statistics."""
    logger.info("Dataframe Analysis:")
    analysis = {}

    logger.info("General Info:")
    logger.info(f"Number of samples: {len(df)}")
    logger.info(f"Columns: {df.columns.tolist()}")

    analysis["general_info"] = {
        "n_samples": int(len(df)),
        "columns": df.columns.tolist(),
    }

    logger.info("Missing values per column:")
    df = df.replace([np.inf, -np.inf], np.nan)
    missing_values = df.isnull().sum()
    logger.info(missing_values)

    analysis["missing_values"] = missing_values.to_dict()

    logger.info("Data types:")
    logger.info(df.dtypes)
    analysis["data_types"] = df.dtypes.astype(str).to_dict()

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = cfg.data.label_col

    logger.info("Numerical Columns Statistics:")
    num_stats = df[num_cols].describe().T
    logger.info(num_stats)
    analysis["numerical_statistics"] = num_stats.to_dict()

    logger.info("Categorical Columns Statistics:")
    cat_stats = {}
    for col in cat_cols:
        logger.info(f"Column: {col}")
        logger.info(f"Number of unique values: {df[col].nunique()}")
        value_counts = df[col].value_counts()
        logger.info(value_counts)

        cat_stats[col] = {
            "n_unique": int(df[col].nunique()),
            "value_counts": value_counts.to_dict(),
        }
    analysis["categorical_statistics"] = cat_stats

    logger.info("Label Distribution:")
    label_dist = df[label_col].value_counts()
    logger.info(label_dist)
    analysis["label_distribution"] = label_dist.to_dict()

    output_path = Path(cfg.path.json_logs) / "data_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Analysis saved to {output_path}")


def sample_intra_class_distances(X, idx_c, n_pairs=200_000, metric="euclidean"):
    """
    Sample distances between pairs of points within the same class.
    Args:
        X: np.ndarray of shape (n_samples, n_features)
        idx_c: indices of samples belonging to the same class
        n_pairs: number of distance pairs to sample
        metric: distance metric to use
    """
    idx_c = np.asarray(idx_c)
    n = len(idx_c)
    if n < 2:
        return np.array([])

    i = np.random.randint(0, n, size=n_pairs)
    j = np.random.randint(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]

    Xi = X[idx_c[i]]
    Xj = X[idx_c[j]]
    d = paired_distances(Xi, Xj, metric=metric)
    return d


def sample_inter_class_distances(
    X, idx_c, idx_not_c, n_pairs=200_000, metric="euclidean"
):
    """
    Sample distances between pairs of points from different classes.
    Args:
        X: np.ndarray of shape (n_samples, n_features)
        idx_c: indices of samples belonging to the first class
        idx_not_c: indices of samples not belonging to the first class
        n_pairs: number of distance pairs to sample
        metric: distance metric to use
    """
    idx_c = np.asarray(idx_c)
    idx_not_c = np.asarray(idx_not_c)

    if len(idx_c) == 0 or len(idx_not_c) == 0:
        return np.array([])

    i = np.random.randint(0, len(idx_c), size=n_pairs)
    j = np.random.randint(0, len(idx_not_c), size=n_pairs)

    Xi = X[idx_c[i]]
    Xj = X[idx_not_c[j]]
    d = paired_distances(Xi, Xj, metric=metric)
    return d


def summarize_distances(d):
    if d.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "n": 0,
        }
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "p90": float(np.quantile(d, 0.90)),
        "p95": float(np.quantile(d, 0.95)),
        "n": int(d.size),
    }


def run_separability_analysis(df: pd.DataFrame, label_col: str):
    y = df[label_col].astype(str).values
    X = df.drop(columns=[label_col]).values

    labels = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in labels}

    rows = []
    all_idx = np.arange(len(y))

    for c in labels:
        logger.info(f"Analyzing class {c}...")
        idx_c = class_indices[c]
        idx_not = all_idx[y != c]

        X_c = X[idx_c]
        centroid = np.mean(X_c, axis=0)
        distances_from_centroid = np.linalg.norm(X_c - centroid, axis=1)
        class_radius = float(np.max(distances_from_centroid))
        radius_mean = float(np.mean(distances_from_centroid))
        radius_median = float(np.median(distances_from_centroid))
        intra = sample_intra_class_distances(X, idx_c)
        inter = sample_inter_class_distances(X, idx_c, idx_not)

        s_intra = summarize_distances(intra)
        s_inter = summarize_distances(inter)

        ratio = (
            s_intra["mean"] / s_inter["mean"]
            if np.isfinite(s_intra["mean"]) and np.isfinite(s_inter["mean"])
            else np.nan
        )

        def approximate_class_silhouette(intra_mean: float, inter_mean: float):
            """
            Approximate silhouette score for class c using mean distances.
            s_c ≈ (b̄_c - ā_c) / max(ā_c, b̄_c)
            """
            if not np.isfinite(intra_mean) or not np.isfinite(inter_mean):
                return np.nan
            return (inter_mean - intra_mean) / max(intra_mean, inter_mean)

        class_sil_score = approximate_class_silhouette(s_intra["mean"], s_inter["mean"])

        rows.append(
            {
                "class": c,
                "n_samples": int(len(idx_c)),
                "class_radius": class_radius,
                "radius_mean": radius_mean,
                "radius_median": radius_median,
                "intra_mean": s_intra["mean"],
                "intra_median": s_intra["median"],
                "intra_min": s_intra["min"],
                "intra_max": s_intra["max"],
                "intra_p90": s_intra["p90"],
                "intra_p95": s_intra["p95"],
                "inter_mean": s_inter["mean"],
                "inter_median": s_inter["median"],
                "inter_min": s_inter["min"],
                "inter_max": s_inter["max"],
                "inter_p90": s_inter["p90"],
                "inter_p95": s_inter["p95"],
                "intra_inter_ratio": float(ratio),
                "silhouette_score": float(class_sil_score),
            }
        )

    separability_df = pd.DataFrame(rows).sort_values(
        "intra_inter_ratio", ascending=False
    )

    logger.info("  Computing silhouette scores.")
    max_samples_for_silhouette = 100_000

    logger.info(f"  Stratified subsampling to {max_samples_for_silhouette} samples.")
    sample_idx, _ = train_test_split(
        np.arange(len(X)),
        train_size=max_samples_for_silhouette,
        stratify=y,
        random_state=42,
    )
    Xs_sample = X[sample_idx]
    y_sample = y[sample_idx]

    sil_score = silhouette_score(Xs_sample, y_sample)

    return separability_df, sil_score


def class_feature_similarity(df, class_a, class_b, label_col, num_cols, cat_cols):
    """
    Compute feature similarity between two classes.

    Returns similarity scores in [0, 1] where:
    - 1 = identical distributions
    - 0 = completely different distributions

    For numerical features: uses Bhattacharyya coefficient (overlap between distributions)
    For categorical features: uses Jaccard similarity
    """
    df_a = df[df[label_col] == class_a]
    df_b = df[df[label_col] == class_b]

    out = {}
    global_similarity = 0.0
    n_features = 0

    # Numerical features: Bhattacharyya coefficient
    # Assumes normal distributions and measures overlap
    for col in num_cols:
        mean_a = df_a[col].mean()
        mean_b = df_b[col].mean()

        std_a = df_a[col].std()
        std_b = df_b[col].std()

        # Bhattacharyya coefficient for normal distributions
        # BC = exp(-0.25 * D_B) where D_B is Bhattacharyya distance
        variance_a = std_a**2 + 1e-8
        variance_b = std_b**2 + 1e-8

        similarity = 1 - abs(mean_a - mean_b) / (std_a + std_b + 1e-8)
        similarity = float(max(0.0, min(1.0, similarity)))

        out[col] = {
            "class_a_mean": float(mean_a),
            "class_a_std": float(std_a),
            "class_b_mean": float(mean_b),
            "class_b_std": float(std_b),
            "similarity": similarity,
        }
        global_similarity += similarity
        n_features += 1

    # Categorical features: Jaccard similarity
    for col in cat_cols:
        freq_a = df_a[col].value_counts(normalize=True)
        freq_b = df_b[col].value_counts(normalize=True)

        # Get all categories from both classes
        all_categories = set(freq_a.index).union(set(freq_b.index))

        # Compute intersection (minimum frequencies) and union (maximum frequencies)
        intersection = sum(
            min(freq_a.get(cat, 0), freq_b.get(cat, 0)) for cat in all_categories
        )
        union = sum(
            max(freq_a.get(cat, 0), freq_b.get(cat, 0)) for cat in all_categories
        )

        # Jaccard similarity: |A ∩ B| / |A ∪ B|
        similarity = float(intersection / union) if union > 0 else 0.0

        out[col] = {
            "class_a_top_5_freq": list(freq_a[:5].index),
            "class_b_top_5_freq": list(freq_b[:5].index),
            "similarity": similarity,
        }
        global_similarity += similarity
        n_features += 1

    # Global similarity: average across all features
    out["global_similarity"] = (
        float(global_similarity / n_features) if n_features > 0 else 0.0
    )
    return out


def main() -> None:
    """Main entry point for data preparation."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    label_col = "multi_" + cfg.data.label_col

    # Load raw data

    # Analyze data
    # logger.info("Loading raw data...")
    # raw_data_path = Path(cfg.path.raw_data) / f"{cfg.data.file_name}.csv"
    # df = load_df(raw_data_path)
    # logger.info("Analyzing data...")
    # analyze_df(df, cfg)

    filter_query = cfg.get("filter_query", None)
    train_df, test_df = prepare_data(cfg, filter_query=filter_query)
    # sep_df_train, sil_score_train = run_separability_analysis(
    #     train_df, label_col=label_col
    # )
    # sep_df_test, sil_score_test = run_separability_analysis(
    #     test_df, label_col=label_col
    # )

    # logger.info("=== Separability (intra/inter) Train ===")
    # logger.info(sep_df_train.head(11))
    # logger.info("=== Separability (intra/inter) Test ===")
    # logger.info(sep_df_test.head(11))

    # json_path = Path(cfg.path.json_logs)
    # json_path.mkdir(parents=True, exist_ok=True)
    # sep_df_train.to_json(
    #     json_path
    #     / f"separability_train{'_run_' + str(cfg.get('run_id')) if cfg.get('run_id') is not None else ''}.json",
    #     orient="records",
    #     indent=2,
    # )
    # sep_df_test.to_json(
    #     json_path
    #     / f"separability_test{'_run_' + str(cfg.get('run_id')) if cfg.get('run_id') is not None else ''}.json",
    #     orient="records",
    #     indent=2,
    # )

    # logger.info("\n=== Silhouette Score ===")
    # logger.info(sil_score_train)
    # logger.info(sil_score_test)
    out = class_feature_similarity(
        train_df,
        class_a="Benign",
        class_b="Backdoor",
        label_col=cfg.data.label_col,
        num_cols=cfg.data.num_cols,
        cat_cols=cfg.data.cat_cols,
    )

    rows = []
    for feature, stats in out.items():
        if feature == "global_similarity":
            continue

        row = {
            "feature": feature,
            "similarity": stats["similarity"],
            "class_a_mean": stats.get("class_a_mean", None),
            "class_a_std": stats.get("class_a_std", None),
            "class_b_mean": stats.get("class_b_mean", None),
            "class_b_std": stats.get("class_b_std", None),
            "class_a_top_5_freq": str(stats.get("class_a_top_5_freq", None)),
            "class_b_top_5_freq": str(stats.get("class_b_top_5_freq", None)),
        }
        rows.append(row)

    similarity_df = pd.DataFrame(rows)
    similarity_df = similarity_df.sort_values("similarity", ascending=False)
    logger.info(similarity_df)
    logger.info(f"Global similarity: {out['global_similarity']}")
    save_df(similarity_df, Path(cfg.path.json_logs) / "class_feature_similarity.csv")


if __name__ == "__main__":
    main()
