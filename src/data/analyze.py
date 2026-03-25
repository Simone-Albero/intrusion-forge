from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_samples
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


def _majority_class_map(
    cluster_labels: np.ndarray, class_labels: np.ndarray
) -> dict[str, str]:
    """Map each cluster label to its majority class label."""
    mapping: dict[str, str] = {}
    for c in np.unique(cluster_labels):
        classes_in = class_labels[cluster_labels == c]
        vals, counts = np.unique(classes_in, return_counts=True)
        mapping[str(c)] = str(vals[np.argmax(counts)])
    return mapping


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
        scores = silhouette_samples(X[idx], labels[idx])
    except Exception:
        return None
    full = np.full(n, np.nan)
    full[idx] = scores
    return full


def _single_cluster_stats(
    samples: np.ndarray,
    centroid: np.ndarray,
    pairwise_dist_row: np.ndarray,
    idx_in_keys: int,
    cluster_to_class: dict[str, str],
    class_centroids: dict[str, np.ndarray],
    key: str,
    keys: list[str],
    sil_values: np.ndarray | None,
    mask_arr: np.ndarray,
    eps: float,
    metric: str,
) -> dict:
    """Compute statistics for a single cluster."""
    n = len(samples)
    if n == 0:
        return {
            "centroid": centroid.tolist(),
            "n_samples": 0,
            "n_unique": 0,
            "unique_ratio": None,
            "intra_dispersion": None,
            "std_dispersion": None,
            "median_dispersion": None,
            "max_dispersion": None,
            "density": None,
            "log_density": None,
            "dist_to_class_centroid": None,
            "dist_to_nearest_cluster": None,
            "dist_to_nearest_foreign_cluster": None,
            "nearest_separation_ratio": None,
            "foreign_separation_ratio": None,
            "silhouette": None,
        }

    dists = pairwise_distances(samples, centroid.reshape(1, -1), metric=metric).ravel()
    intra = float(np.mean(dists))
    n_unique = int(np.unique(samples, axis=0).shape[0])

    row = pairwise_dist_row.copy()
    row[idx_in_keys] = np.inf
    min_dist = np.min(row) if len(keys) > 1 else np.inf
    nearest = float(min_dist) if np.isfinite(min_dist) else None

    cls_key = cluster_to_class.get(key)
    foreign_dists = [
        pairwise_dist_row[j]
        for j, k in enumerate(keys)
        if j != idx_in_keys and cluster_to_class.get(k) != cls_key
    ]
    nearest_foreign = float(np.min(foreign_dists)) if foreign_dists else None

    return {
        "centroid": centroid.tolist(),
        "n_samples": n,
        "n_unique": n_unique,
        "unique_ratio": float(n_unique / n),
        "intra_dispersion": intra,
        "std_dispersion": float(np.std(dists)),
        "median_dispersion": float(np.median(dists)),
        "max_dispersion": float(np.max(dists)),
        "density": float(n / (intra + eps) ** 3),
        "log_density": float(np.log1p(n) - 3.0 * np.log(intra + eps)),
        "dist_to_class_centroid": (
            float(
                pairwise_distances(
                    centroid.reshape(1, -1),
                    class_centroids[cls_key].reshape(1, -1),
                    metric=metric,
                )[0, 0]
            )
            if cls_key in class_centroids
            else None
        ),
        "dist_to_nearest_cluster": nearest,
        "dist_to_nearest_foreign_cluster": nearest_foreign,
        "nearest_separation_ratio": (
            float(nearest / (intra + eps)) if nearest is not None else None
        ),
        "foreign_separation_ratio": (
            float(nearest_foreign / (intra + eps))
            if nearest_foreign is not None
            else None
        ),
        "silhouette": (
            float(np.nanmean(sil_values[mask_arr]))
            if sil_values is not None and np.any(mask_arr)
            else None
        ),
    }


def compute_cluster_stats(
    df_: pd.DataFrame,
    feature_cols: list[str],
    cluster_col: str,
    encoded_label_col: str,
    centroids: dict,
    eps: float = 1e-8,
    metric: str = "euclidean",
    compute_silhouette: bool = True,
    silhouette_max_samples: int = 10_000,
) -> dict:
    """Compute per-cluster statistics.

    Args:
        metric: Distance metric for intra-cluster dispersion and centroid
            distances. Supports ``"euclidean"`` and ``"cosine"``.
        silhouette_max_samples: Max points for the silhouette approximation.
            Set to 0 to use all samples (exact but O(n²)).
    """
    if df_.empty:
        return {}

    X = df_[feature_cols].to_numpy(dtype=float)
    cluster_labels = df_[cluster_col].to_numpy()
    class_labels = df_[encoded_label_col].to_numpy()

    class_centroids = {
        str(cls): X[class_labels == cls].mean(axis=0) for cls in np.unique(class_labels)
    }
    cluster_to_class = _majority_class_map(cluster_labels, class_labels)

    present = {str(c) for c in cluster_labels}
    keys = [str(k) for k in centroids if str(k) in present]
    if not keys:
        return {}

    centroid_matrix = np.stack([np.asarray(centroids[k], dtype=float) for k in keys])
    pairwise_dist = pairwise_distances(centroid_matrix, metric=metric)

    sil_values = None
    if compute_silhouette:
        max_s = silhouette_max_samples if silhouette_max_samples > 0 else len(X)
        sil_values = _approx_silhouette(X, cluster_labels, max_samples=max_s)

    cluster_stats: dict[str, dict] = {}
    for i, key in tqdm(enumerate(keys), total=len(keys), desc="cluster stats"):
        mask = df_[cluster_col].astype(str) == key
        samples = df_.loc[mask, feature_cols].to_numpy(dtype=float)
        cluster_stats[key] = _single_cluster_stats(
            samples=samples,
            centroid=centroid_matrix[i],
            pairwise_dist_row=pairwise_dist[i],
            idx_in_keys=i,
            cluster_to_class=cluster_to_class,
            class_centroids=class_centroids,
            key=key,
            keys=keys,
            sil_values=sil_values,
            mask_arr=mask.to_numpy(),
            eps=eps,
            metric=metric,
        )

    return cluster_stats


def compute_clusters_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    cluster_col: str,
    centroids: dict,
    feature_cols: list[str],
    metric: str = "euclidean",
) -> dict:
    """Aggregate cluster metadata across all splits."""
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

    cluster_stats = compute_cluster_stats(
        df_=df_,
        feature_cols=feature_cols,
        cluster_col=cluster_col,
        encoded_label_col=encoded_label_col,
        centroids=centroids,
        metric=metric,
    )

    return {
        "class_to_clusters": class_to_clusters,
        "clusters_distribution": clusters_distribution,
        "cluster_stats": cluster_stats,
    }


def _to_finite_float(value) -> float | None:
    """Convert to float, returning None for non-finite or unconvertible values."""
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def build_cluster_summary(
    class_to_clusters: dict,
    clusters_distribution: dict,
    cluster_stats: dict,
    cluster_errors: dict,
    separability: dict,
) -> dict:
    """Assemble a comprehensive per-cluster summary from all available sources.

    Merges cluster statistics (dispersion, density, silhouette, etc.),
    failure rates, and separability-based foreign/self ratios into a single
    dict keyed by cluster id.

    Args:
        class_to_clusters: ``{class_label: [cluster_id, ...]}``.
        clusters_distribution: ``{cluster_id: sample_count}``.
        cluster_stats: Per-cluster statistics dict (from ``compute_cluster_stats``),
            keyed by cluster id. Each entry may contain centroid, dispersion,
            density, silhouette, etc.
        cluster_errors: ``{cluster_id: {n_error, n_total, error_rate}}``.
        separability: Output of ``compute_pairwise_separability``.

    Returns:
        ``{cluster_id: {cluster_class, cluster_size, failure_rate, is_failed,
        <cluster_stats fields>, foreign_*_*, self_*, ...pairwise distances}}``.
    """
    # --- cluster → class mapping ---
    cluster_to_class = {
        str(c): cls for cls, clusters in class_to_clusters.items() for c in clusters
    }

    # --- per-cluster, per-class separability ratio aggregation ---
    class_cluster_sets = {
        str(cls): {str(c) for c in clusters}
        for cls, clusters in class_to_clusters.items()
    }

    per_cluster_class_ratios: dict[str, dict] = {}
    for cid, ratios in separability.items():
        peer_ratios = {
            str(k): fv
            for k, v in ratios.items()
            if k != "_mean_ratio" and (fv := _to_finite_float(v)) is not None
        }
        per_cluster_class_ratios[str(cid)] = {
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
            for cls, ids in class_cluster_sets.items()
        }

    # --- assemble per-cluster summary ---
    _STATS_KEYS = (
        "intra_dispersion",
        "std_dispersion",
        "median_dispersion",
        "max_dispersion",
        "density",
        "log_density",
        "dist_to_class_centroid",
        "dist_to_nearest_cluster",
        "dist_to_nearest_foreign_cluster",
        "nearest_separation_ratio",
        "foreign_separation_ratio",
        "silhouette",
    )

    results = {}
    for cid in clusters_distribution:
        cid = str(cid)
        cluster_class = cluster_to_class.get(cid)
        cluster_size = clusters_distribution.get(cid)

        # cluster_stats fields (exclude centroid — not useful in summary)
        stats_entry = cluster_stats.get(cid, {})
        stats_fields = {k: stats_entry.get(k) for k in _STATS_KEYS}

        # failure info
        error_entry = cluster_errors.get(cid, {})
        failure_rate = error_entry.get("error_rate")

        # separability: foreign vs self aggregation
        foreign_avgs, foreign_maxs = [], []
        self_avg = self_max = None

        for cls, ratios in per_cluster_class_ratios.get(cid, {}).items():
            if cls == cluster_class:
                self_avg = ratios.get("mean_ratio")
                self_max = ratios.get("max_ratio")
            else:
                if (v := ratios.get("mean_ratio")) is not None:
                    foreign_avgs.append(v)
                if (v := ratios.get("max_ratio")) is not None:
                    foreign_maxs.append(v)

        # pairwise distances to other clusters
        distances = {
            k: _to_finite_float(v)
            for k, v in separability.get(cid, {}).items()
            if k != "_mean_ratio"
        }
        distances[cid] = 1.0

        results[cid] = {
            "cluster_class": cluster_class,
            "cluster_size": cluster_size,
            "failure_rate": failure_rate,
            "is_failed": failure_rate is not None and failure_rate > 0.0,
            **stats_fields,
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
