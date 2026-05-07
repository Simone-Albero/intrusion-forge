from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances
from sklearn.decomposition import PCA
from tqdm import tqdm


def _mean_distance_intra(
    X: np.ndarray,
    idx: np.ndarray,
    max_pairs: int | None,
    metric: str,
    chunk_size: int = 512,
) -> float:
    """Mean intra-class distance without materialising the full distance array.

    Iterates the upper-triangle in row chunks when the total number of pairs is
    within *max_pairs* (or max_pairs is None); otherwise draws random paired
    samples and accumulates a running sum.
    """
    n = len(idx)
    if n < 2:
        return np.nan

    total_pairs = n * (n - 1) // 2
    exact = max_pairs is None or total_pairs <= max_pairs

    running_sum = 0.0
    running_count = 0

    if exact:
        for start in range(0, n - 1, chunk_size):
            end = min(start + chunk_size, n - 1)
            D = pairwise_distances(
                X[idx[start:end]], X[idx[start + 1 :]], metric=metric
            )
            for li in range(end - start):
                row = D[li, li:]  # upper-triangle only
                running_sum += float(row.sum())
                running_count += len(row)
    else:
        size = int(max_pairs * 1.1) + 32
        i = np.random.randint(0, n, size=size)
        j = np.random.randint(0, n, size=size)
        mask = i != j
        i, j = i[mask][:max_pairs], j[mask][:max_pairs]
        for start in range(0, len(i), chunk_size):
            end = min(start + chunk_size, len(i))
            d = paired_distances(
                X[idx[i[start:end]]], X[idx[j[start:end]]], metric=metric
            )
            running_sum += float(d.sum())
            running_count += len(d)

    return running_sum / running_count if running_count > 0 else np.nan


def _mean_distance_inter(
    X: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    max_pairs: int | None,
    metric: str,
    chunk_size: int = 512,
) -> float:
    """Mean inter-class distance without materialising the full distance array."""
    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.nan

    total_pairs = len(idx_a) * len(idx_b)
    exact = max_pairs is None or total_pairs <= max_pairs

    running_sum = 0.0
    running_count = 0

    if exact:
        for start in range(0, len(idx_a), chunk_size):
            end = min(start + chunk_size, len(idx_a))
            D = pairwise_distances(X[idx_a[start:end]], X[idx_b], metric=metric)
            running_sum += float(D.sum())
            running_count += D.size
    else:
        n = max_pairs
        replace = n > min(len(idx_a), len(idx_b))
        i = np.random.choice(len(idx_a), size=n, replace=replace)
        j = np.random.choice(len(idx_b), size=n, replace=replace)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            d = paired_distances(
                X[idx_a[i[start:end]]], X[idx_b[j[start:end]]], metric=metric
            )
            running_sum += float(d.sum())
            running_count += len(d)

    return running_sum / running_count if running_count > 0 else np.nan


def compute_pairwise_separability(
    X: np.ndarray,
    y: np.ndarray,
    max_pairs: int | None = 50_000,
    metric: str = "cosine",
    pca_variance: float = 0.9,
) -> dict[str, dict[str, float]]:
    """Compute pairwise separability ratios between groups defined by *y*.

    For every pair of groups the ratio ``mean_intra / mean_inter`` is computed.
    Low values indicate well-separated groups.

    Distances are never collected into full arrays; sums are accumulated in
    chunk-sized batches so RAM usage stays O(chunk_size) instead of O(n²).

    Returns ``{group: {other_group: ratio, "_mean_ratio": float}}`` sorted by
    mean ratio ascending.
    """
    classes = np.unique(y)

    X = PCA(n_components=pca_variance, svd_solver="full", whiten=True).fit_transform(X)

    pair_metrics: dict[tuple, float] = {}
    n_pairs = len(classes) * (len(classes) - 1) // 2
    for class_a, class_b in tqdm(
        combinations(classes, 2), total=n_pairs, desc="class pairs"
    ):
        idx_a = np.where(y == class_a)[0]
        idx_b = np.where(y == class_b)[0]

        intra_a = _mean_distance_intra(X, idx_a, max_pairs=max_pairs, metric=metric)
        intra_b = _mean_distance_intra(X, idx_b, max_pairs=max_pairs, metric=metric)
        inter_ab = _mean_distance_inter(
            X, idx_a, idx_b, max_pairs=max_pairs, metric=metric
        )

        intra_mean = (
            float(np.mean([intra_a, intra_b]))
            if np.isfinite(intra_a) and np.isfinite(intra_b)
            else np.nan
        )
        ratio = (
            intra_mean / inter_ab
            if np.isfinite(intra_mean) and np.isfinite(inter_ab) and inter_ab != 0.0
            else np.nan
        )

        pair_metrics[(str(class_a), str(class_b))] = float(ratio)

    result: dict[str, dict[str, float]] = {}
    for cls in map(str, classes):
        pairs = {
            b if a == cls else a: ratio
            for (a, b), ratio in pair_metrics.items()
            if a == cls or b == cls
        }
        pairs = dict(
            sorted(
                pairs.items(), key=lambda x: x[1] if np.isfinite(x[1]) else float("inf")
            )
        )
        mean_ratio = float(np.nanmean(list(pairs.values()))) if pairs else np.nan
        result[cls] = {"_mean_ratio": mean_ratio, **pairs}

    return dict(sorted(result.items(), key=lambda x: x[1]["_mean_ratio"]))


def get_df_info(df: pd.DataFrame, label_col: str | None = None) -> dict:
    """Return basic information about a DataFrame."""
    info = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
        "memory_usage": int(df.memory_usage(deep=True).sum()),
        "feature_info": {
            col: {
                "dtype": str(df[col].dtype),
                "unique_count": int(df[col].nunique()),
            }
            for col in df.columns
        },
    }
    if label_col and label_col in df.columns:
        info["label_distribution"] = df[label_col].value_counts().to_dict()
    return info


def compute_df_metadata(
    splits: dict[str, pd.DataFrame],
    label_col: str,
    num_cols: list[str],
    cat_cols: list[str],
    benign_tag: str,
    label_mapping: dict | None = None,
    weights_key: str | None = None,
) -> dict:
    """Compute metadata dictionary for one or more named DataFrames.

    Args:
        splits: Mapping of ``tag → DataFrame`` (e.g. ``{"train": …, "val": …}``).
        weights_key: Tag of the split used for class-weight computation.
            Defaults to ``"train"`` if present, otherwise the first key.
    """
    if not splits:
        raise ValueError("splits must contain at least one DataFrame.")

    if weights_key is None:
        weights_key = "train" if "train" in splits else next(iter(splits))
    ref_df = splits[weights_key]

    class_counts = ref_df[label_col].value_counts().sort_index()
    class_weights = len(ref_df) / (len(class_counts) * class_counts)
    class_weights = np.log1p(class_weights) / np.log1p(class_weights).max()

    return {
        "label_mapping": label_mapping or {},
        "dataset_sizes": {tag: len(df) for tag, df in splits.items()},
        "samples_per_class": {
            tag: df[label_col].value_counts().to_dict() for tag, df in splits.items()
        },
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "benign_tag": benign_tag,
        "num_classes": ref_df[label_col].nunique(),
        "class_weights": class_weights.tolist(),
    }


def _approx_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 10_000,
    min_per_cluster: int = 30,
) -> np.ndarray | None:
    """Compute per-sample silhouette scores with stratified subsampling.

    Sampling is stratified by cluster: each cluster contributes at least
    *min_per_cluster* points (or all its points if smaller), so even rare
    clusters are represented.  Non-sampled points receive ``NaN``; per-cluster
    means remain valid.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None

    n = len(X)
    if n <= max_samples:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(42)
        idx_parts: list[np.ndarray] = []
        # guarantee min_per_cluster from each cluster
        for lbl in unique_labels:
            members = np.where(labels == lbl)[0]
            take = min(len(members), max(min_per_cluster, 1))
            idx_parts.append(rng.choice(members, size=take, replace=False))
        guaranteed = np.concatenate(idx_parts)
        remaining_budget = max_samples - len(guaranteed)
        if remaining_budget > 0:
            pool = np.setdiff1d(np.arange(n), guaranteed)
            extra = rng.choice(
                pool, size=min(remaining_budget, len(pool)), replace=False
            )
            idx = np.concatenate([guaranteed, extra])
        else:
            idx = guaranteed

    try:
        from sklearn.metrics import silhouette_samples

        scores = silhouette_samples(X[idx], labels[idx])
    except ValueError:
        return None
    full = np.full(n, np.nan)
    full[idx] = scores
    return full


def compute_clusters_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    cluster_col: str,
    centroids: dict,
) -> dict:
    """Aggregate cluster metadata across all splits.

    Returns {class_to_clusters, clusters_distribution, centroids}.
    """
    df_ = pd.concat([train_df, val_df, test_df], ignore_index=True)
    encoded_label_col = f"encoded_{label_col}"

    class_to_clusters = {
        str(cls): [
            str(v) for v in df_[df_[encoded_label_col] == cls][cluster_col].unique()
        ]
        for cls in df_[encoded_label_col].unique()
    }

    clusters_distribution = {
        str(k): v for k, v in df_[cluster_col].value_counts().to_dict().items()
    }

    return {
        "class_to_clusters": class_to_clusters,
        "clusters_distribution": clusters_distribution,
        "centroids": {str(k): v for k, v in centroids.items()},
    }
