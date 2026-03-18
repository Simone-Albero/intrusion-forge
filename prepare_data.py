import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_samples

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import save_to_json

from src.data.io import load_df, save_df
from src.data.preprocessing import (
    LogTransformer,
    TopNHashEncoder,
    drop_nans,
    encode_labels,
    ml_split,
    query_filter,
    rare_category_filter,
    random_undersample_df,
)

from src.ml.clustering import hdbscan_grid_search

setup_logger()
logger = logging.getLogger(__name__)


def get_df_info(df, label_col=None):
    """Return basic information about the dataframe."""
    info = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
        "memory_usage": int(df.memory_usage(deep=True).sum()),
        "feature_info": {},
    }

    for col in df.columns:
        info["feature_info"][col] = {
            "dtype": str(df[col].dtype),
            "unique_count": int(df[col].nunique()),
        }

    if label_col and label_col in df.columns:
        info["label_distribution"] = df[label_col].value_counts().to_dict()

    return info


def compute_splits_metadata(
    train_df,
    val_df,
    test_df,
    label_col,
    num_cols,
    cat_cols,
    benign_tag,
    label_mapping=None,
):
    """Compute and return metadata dictionary for the dataset splits."""
    class_counts = train_df[label_col].value_counts().sort_index()
    class_weights = len(train_df) / (len(class_counts) * class_counts)
    class_weights = np.log1p(class_weights) / np.log1p(class_weights).max()
    class_weights_list = class_weights.tolist()

    metadata = {
        "label_mapping": label_mapping or {},
        "dataset_sizes": {
            split: len(df_split)
            for split, df_split in zip(
                ["train", "val", "test"], [train_df, val_df, test_df]
            )
        },
        "samples_per_class": {
            split: df_split[label_col].value_counts().to_dict()
            for split, df_split in zip(
                ["train", "val", "test"], [train_df, val_df, test_df]
            )
        },
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "benign_tag": benign_tag,
        "num_classes": train_df[label_col].nunique(),
        "class_weights": class_weights_list,
    }

    return metadata


def compute_cluster_stats(
    df_: pd.DataFrame,
    feature_cols: list[str],
    cluster_col: str,
    encoded_label_col: str,
    centroids: dict,
    eps: float = 1e-8,
    compute_silhouette: bool = True,
) -> dict:
    """Compute robust per-cluster statistics."""
    if df_.empty:
        return {}

    X = df_[feature_cols].to_numpy(dtype=float)
    cluster_labels = df_[cluster_col].to_numpy()
    class_labels = df_[encoded_label_col].to_numpy()

    class_centroids = {
        str(cls): X[class_labels == cls].mean(axis=0) for cls in np.unique(class_labels)
    }

    cluster_to_class = {}
    for c in np.unique(cluster_labels):
        classes_in = class_labels[cluster_labels == c]
        vals, counts = np.unique(classes_in, return_counts=True)
        cluster_to_class[str(c)] = str(vals[np.argmax(counts)])

    present = {str(c) for c in cluster_labels}
    keys = [str(k) for k in centroids if str(k) in present]
    if not keys:
        return {}

    centroid_matrix = np.stack([np.asarray(centroids[k], dtype=float) for k in keys])
    pairwise_dist = np.linalg.norm(
        centroid_matrix[:, None] - centroid_matrix[None], axis=-1
    )

    sil_values = None
    if compute_silhouette and len(np.unique(cluster_labels)) >= 2:
        try:
            sil_values = silhouette_samples(X, cluster_labels)
        except Exception:
            pass

    cluster_stats = {}
    for i, key in tqdm(enumerate(keys), total=len(keys), desc="cluster stats"):
        centroid = centroid_matrix[i]
        mask = df_[cluster_col].astype(str) == key
        samples = df_.loc[mask, feature_cols].to_numpy(dtype=float)
        n = len(samples)

        if n == 0:
            cluster_stats[key] = {
                "centroid": centroid.tolist(),
                "n_samples": 0,
                "n_unique": 0,
                **{
                    k: None
                    for k in [
                        "unique_ratio",
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
                    ]
                },
            }
            continue

        dists = np.linalg.norm(samples - centroid, axis=1)
        intra = float(np.mean(dists))
        n_unique = int(np.unique(samples, axis=0).shape[0])

        row = pairwise_dist[i].copy()
        row[i] = np.inf
        min_dist = np.min(row) if len(keys) > 1 else np.inf
        nearest = float(min_dist) if np.isfinite(min_dist) else None

        cls_key = cluster_to_class.get(key)
        foreign_dists = [
            pairwise_dist[i, j]
            for j, k in enumerate(keys)
            if j != i and cluster_to_class.get(k) != cls_key
        ]
        nearest_foreign = float(np.min(foreign_dists)) if foreign_dists else None

        mask_arr = mask.to_numpy()
        cluster_stats[key] = {
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
                float(np.linalg.norm(centroid - class_centroids[cls_key]))
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
                float(np.mean(sil_values[mask_arr]))
                if sil_values is not None and np.any(mask_arr)
                else None
            ),
        }

    return cluster_stats


def compute_clusters_metadata(
    train_df,
    val_df,
    test_df,
    label_col,
    cluster_col,
    centroids,
    feature_cols,
):
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
    )

    return {
        "class_to_clusters": class_to_clusters,
        "clusters_distribution": clusters_distribution,
        "cluster_stats": cluster_stats,
    }


def build_preprocessor(num_cols, cat_cols, top_n, hash_buckets):
    """Build a ColumnTransformer for numerical and categorical features."""
    set_config(transform_output="pandas")
    transformers = []

    if num_cols:
        num_pipeline = Pipeline(
            [
                ("log_transformer", LogTransformer()),
                ("scaler", RobustScaler()),
            ]
        )
        logger.info(f"Numerical pipeline steps: {num_pipeline.steps}")
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline(
            [
                (
                    "top_n_encoder",
                    TopNHashEncoder(top_n=top_n, hash_buckets=hash_buckets),
                ),
            ]
        )
        logger.info(f"Categorical pipeline steps: {cat_pipeline.steps}")
        transformers.append(("cat", cat_pipeline, cat_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def preprocess_df(
    df,
    num_cols,
    cat_cols,
    label_col,
    filter_query,
    min_cat_count,
    train_frac,
    val_frac,
    test_frac,
    random_state,
    top_n,
    hash_buckets,
):
    """Preprocess dataframe: filter, encode, scale, and split."""
    df = drop_nans(df, num_cols + cat_cols + [label_col])
    df = query_filter(df, filter_query)
    df = rare_category_filter(df, [label_col], min_count=min_cat_count)

    train_df, val_df, test_df = ml_split(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
        label_col=label_col,
    )
    train_df = random_undersample_df(train_df, label_col, random_state)

    preprocessor = build_preprocessor(num_cols, cat_cols, top_n, hash_buckets)
    preprocessor.fit(train_df)
    train_df, val_df, test_df = (
        preprocessor.transform(split) for split in [train_df, val_df, test_df]
    )

    return train_df, val_df, test_df


def compute_clusters(df, feature_cols, label_col, classes=None, dst_col="cluster"):
    """Compute HDBSCAN clusters for specified classes, encoding noise (-1) as max_label + 1.

    Returns:
        df: DataFrame with cluster assignments
        centroids: dict mapping cluster_label -> centroid array (in original feature space)
    """
    df[dst_col] = -1
    centroids = {}
    offset = 0

    if classes is None:
        groups = [(df.index, df[feature_cols].values, "all data")]
    else:
        groups = [
            (
                df[df[label_col] == cls].index,
                df.loc[df[label_col] == cls, feature_cols].values,
                f"class '{cls}' (n_samples={df[df[label_col] == cls].shape[0]})",
            )
            for cls in classes
        ]

    for idx, values, log_label in groups:
        logger.info(f"Computing clusters for {log_label}...")
        _, labels, _, _ = hdbscan_grid_search(values)
        noise_label = int(labels.max()) + 1
        labels = np.where(labels == -1, noise_label, labels)

        df.loc[idx, dst_col] = labels + offset

        for lbl in np.unique(labels):
            centroids[str(lbl + offset)] = values[labels == lbl].mean(axis=0)

        offset += noise_label + 1

    return df, centroids


def clusters_over_splits(
    train_df, val_df, test_df, feature_cols, label_col, dst_col, classes
):
    train_df, train_centroids = compute_clusters(
        train_df, feature_cols, label_col, classes, dst_col=dst_col
    )
    val_df, val_centroids = compute_clusters(
        val_df, feature_cols, label_col, classes, dst_col=dst_col
    )
    test_df, test_centroids = compute_clusters(
        test_df, feature_cols, label_col, classes, dst_col=dst_col
    )

    centroids = {"train": train_centroids, "val": val_centroids, "test": test_centroids}
    return train_df, val_df, test_df, centroids


def clusters_over_all(
    train_df, val_df, test_df, feature_cols, label_col, dst_col, classes
):
    train_df["_split"] = "train"
    val_df["_split"] = "val"
    test_df["_split"] = "test"

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df, centroids = compute_clusters(
        combined_df, feature_cols, label_col, classes, dst_col=dst_col
    )

    train_df = (
        combined_df[combined_df["_split"] == "train"]
        .drop("_split", axis=1)
        .reset_index(drop=True)
    )
    val_df = (
        combined_df[combined_df["_split"] == "val"]
        .drop("_split", axis=1)
        .reset_index(drop=True)
    )
    test_df = (
        combined_df[combined_df["_split"] == "test"]
        .drop("_split", axis=1)
        .reset_index(drop=True)
    )

    return train_df, val_df, test_df, centroids


def run_clustering(
    train_df,
    val_df,
    test_df,
    feature_cols,
    label_col,
    clustering_type,
    cluster_classes,
    ignore_clusters,
    seed,
    cluster_col="cluster",
):
    """Apply the requested clustering strategy and return updated splits with centroids."""
    cluster_fn = {
        "over_splits": clusters_over_splits,
        "over_all": clusters_over_all,
    }.get(clustering_type)

    if cluster_fn is None:
        raise ValueError(f"Unknown clustering_type: '{clustering_type}'")

    if cluster_classes is not None and len(cluster_classes) == 0:
        cluster_classes = train_df[label_col].unique().tolist()
        logger.info(
            f"No cluster_classes specified, using all classes: {cluster_classes}"
        )

    train_df, val_df, test_df, centroids = cluster_fn(
        train_df,
        val_df,
        test_df,
        feature_cols=feature_cols,
        label_col=label_col,
        dst_col=cluster_col,
        classes=cluster_classes,
    )

    if ignore_clusters:
        for df in [val_df, test_df]:
            df.drop(
                df[df[cluster_col].isin(ignore_clusters)].index,
                inplace=True,
            )
            df.reset_index(drop=True, inplace=True)
        train_df = train_df[~train_df[cluster_col].isin(ignore_clusters)].reset_index(
            drop=True
        )
        train_df = random_undersample_df(train_df, label_col, seed)

    return train_df, val_df, test_df, centroids


def prepare(cfg):
    """Prepare data given a configuration object."""
    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    label_col = cfg.data.label_col

    raw_data_path = Path(cfg.path.raw_data)
    processed_data_path = Path(cfg.path.processed_data)
    json_logs_path = Path(cfg.path.json_logs)

    logger.info("Loading and preprocessing data...")
    df = load_df(str(raw_data_path))

    df_info = get_df_info(df, label_col=label_col)
    save_to_json(df_info, json_logs_path / "data/df_info.json")

    train_df, val_df, test_df = preprocess_df(
        df,
        num_cols,
        cat_cols,
        label_col,
        cfg.data.filter_query,
        cfg.data.min_cat_count,
        cfg.data.train_frac,
        cfg.data.val_frac,
        cfg.data.test_frac,
        cfg.seed,
        cfg.data.top_n,
        cfg.data.hash_buckets,
    )

    train_df, val_df, test_df = (
        df.reset_index(drop=True) for df in [train_df, val_df, test_df]
    )

    centroids = None
    if cfg.clustering_type is not None:
        train_df, val_df, test_df, centroids = run_clustering(
            train_df,
            val_df,
            test_df,
            feature_cols=num_cols + cat_cols,
            label_col=label_col,
            clustering_type=cfg.clustering_type,
            cluster_classes=cfg.cluster_classes,
            ignore_clusters=cfg.ignore_clusters,
            seed=cfg.seed,
        )

    train_df, val_df, test_df, label_mapping = encode_labels(
        train_df, val_df, test_df, label_col, dst_label_col=f"encoded_{label_col}"
    )

    logger.info("Saving processed data...")
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        save_df(split_df, processed_data_path / f"{split_name}.{cfg.data.extension}")

    logger.info("Computing and saving metadata...")
    metadata = compute_splits_metadata(
        train_df,
        val_df,
        test_df,
        label_col,
        num_cols,
        cat_cols,
        cfg.data.benign_tag,
        label_mapping,
    )
    save_to_json(metadata, json_logs_path / "data/df_meta.json")

    if cfg.clustering_type is not None and centroids is not None:
        clusters_metadata = compute_clusters_metadata(
            train_df,
            val_df,
            test_df,
            label_col,
            cluster_col="cluster",
            centroids=centroids,
            feature_cols=num_cols + cat_cols,
        )
        save_to_json(clusters_metadata, json_logs_path / "data/clusters_meta.json")

    return train_df, val_df, test_df, metadata


def main():
    """Main entry point for data preparation."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    prepare(cfg)


if __name__ == "__main__":
    main()
