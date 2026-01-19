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
from sklearn.decomposition import PCA

from src.common.config import load_config
from src.common.logging import setup_logger
from src.data.io import load_df, save_df

setup_logger()
logger = logging.getLogger(__name__)


def analize_df(
    df: pd.DataFrame,
    cfg: Any,
) -> None:
    """Analyze dataframe and log basic statistics."""
    logger.info("Dataframe Analysis:")
    logger.info("General Info:")
    logger.info(f"Number of samples: {len(df)}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info("Missing values per column:")
    df = df.replace([np.inf, -np.inf], np.nan)
    logger.info(df.isnull().sum())
    logger.info("Data types:")
    logger.info(df.dtypes)

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = cfg.data.label_col

    logger.info("Numerical Columns Statistics:")
    logger.info(df[num_cols].describe().T)

    logger.info("Categorical Columns Statistics:")
    for col in cat_cols:
        logger.info(f"Column: {col}")
        logger.info(f"Number of unique values: {df[col].nunique()}")
        logger.info(df[col].value_counts())

    logger.info("Label Distribution:")
    logger.info(df[label_col].value_counts())


def sample_intra_class_distances(
    X, idx_c, n_pairs=200_000, metric="euclidean", rng=None
):
    """
    Sample distances between pairs of points within the same class.
    Args:
        X: np.ndarray of shape (n_samples, n_features)
        idx_c: indices of samples belonging to the same class
        n_pairs: number of distance pairs to sample
        metric: distance metric to use
        rng: optional numpy random generator
    """
    if rng is None:
        rng = np.random.default_rng(0)

    idx_c = np.asarray(idx_c)
    n = len(idx_c)
    if n < 2:
        return np.array([])

    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]

    Xi = X[idx_c[i]]
    Xj = X[idx_c[j]]

    d = paired_distances(Xi, Xj, metric=metric)
    return d


def sample_inter_class_distances(
    X, idx_c, idx_not_c, n_pairs=200_000, metric="euclidean", rng=None
):
    """
    Sample distances between pairs of points from different classes.
    Args:
        X: np.ndarray of shape (n_samples, n_features)
        idx_c: indices of samples belonging to the first class
        idx_not_c: indices of samples not belonging to the first class
        n_pairs: number of distance pairs to sample
        metric: distance metric to use
        rng: optional numpy random generator
    """
    if rng is None:
        rng = np.random.default_rng(0)

    idx_c = np.asarray(idx_c)
    idx_not_c = np.asarray(idx_not_c)

    if len(idx_c) == 0 or len(idx_not_c) == 0:
        return np.array([])

    i = rng.integers(0, len(idx_c), size=n_pairs)
    j = rng.integers(0, len(idx_not_c), size=n_pairs)

    Xi = X[idx_c[i]]
    Xj = X[idx_not_c[j]]
    d = paired_distances(Xi, Xj, metric=metric)
    return d


def summarize_distances(d):
    if d.size == 0:
        return {"mean": np.nan, "median": np.nan, "p90": np.nan, "p95": np.nan, "n": 0}
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "p90": float(np.quantile(d, 0.90)),
        "p95": float(np.quantile(d, 0.95)),
        "n": int(d.size),
    }


def run_separability_analysis(df: pd.DataFrame, label_col: str):
    y = df[label_col].astype(str).values
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    labels = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in labels}

    rows = []
    all_idx = np.arange(len(y))

    for c in labels:
        logger.info(f"Analyzing class {c}...")
        idx_c = class_indices[c]
        idx_not = all_idx[y != c]

        logger.info(f"  Computing intra-class distances.")
        intra = sample_intra_class_distances(Xs, idx_c)
        logger.info(f"  Computing inter-class distances.")
        inter = sample_inter_class_distances(Xs, idx_c, idx_not)

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
            s_c ≈ (b̄_c - ā_c) / max(ā_c, b̄_c)
            """
            if not np.isfinite(intra_mean) or not np.isfinite(inter_mean):
                return np.nan
            return (inter_mean - intra_mean) / max(intra_mean, inter_mean)

        class_sil_score = approximate_class_silhouette(s_intra["mean"], s_inter["mean"])
        logger.info(f"  Approximate silhouette score for class {c}: {class_sil_score}")

        rows.append(
            {
                "class": c,
                "n_samples": int(len(idx_c)),
                "intra_mean": s_intra["mean"],
                "intra_median": s_intra["median"],
                "intra_p90": s_intra["p90"],
                "intra_p95": s_intra["p95"],
                "inter_mean": s_inter["mean"],
                "inter_median": s_inter["median"],
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
        np.arange(len(Xs)),
        train_size=max_samples_for_silhouette,
        stratify=y,
        random_state=42,
    )
    Xs_sample = Xs[sample_idx]
    y_sample = y[sample_idx]

    sil_score = silhouette_score(Xs_sample, y_sample)

    return separability_df, sil_score


def main() -> None:
    """Main entry point for data preparation."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    features = num_cols + cat_cols
    label_col = cfg.data.label_col
    benign_tag = cfg.data.benign_tag

    # Load raw data
    logger.info("Loading raw data...")
    raw_data_path = Path(cfg.path.raw_data) / f"{cfg.data.file_name}.csv"
    df = load_df(str(raw_data_path))
    df = df[features + [label_col]]

    # Analyze data
    # logger.info("Analyzing data...")
    # analize_df(df, cfg)

    sep_df, sil_score = run_separability_analysis(df, label_col=label_col)

    logger.info("=== Separability (intra/inter) ===")
    logger.info(sep_df.head(11))

    logger.info("\n=== Silhouette Score ===")
    logger.info(sil_score)


if __name__ == "__main__":
    main()
