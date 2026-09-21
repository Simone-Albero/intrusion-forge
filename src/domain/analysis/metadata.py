import numpy as np
import pandas as pd


def get_df_info(df: pd.DataFrame, *, label_col: str | None = None) -> dict:
    """Return basic information about a DataFrame."""
    info = {"shape": list(df.shape)}
    if label_col and label_col in df.columns:
        info["label_distribution"] = df[label_col].value_counts().to_dict()
    return info


def compute_df_metadata(
    splits: dict[str, pd.DataFrame],
    label_col: str,
    num_cols: list[str],
    cat_cols: list[str],
    benign_tag: str,
    *,
    label_mapping: dict | None = None,
) -> dict:
    """Metadata for the named splits, with class weights taken from the train split."""
    if not splits:
        raise ValueError("splits must contain at least one DataFrame.")

    ref_df = splits["train"] if "train" in splits else next(iter(splits.values()))

    class_counts = ref_df[label_col].value_counts().sort_index()
    class_weights = len(ref_df) / (len(class_counts) * class_counts)
    log_weights = np.log1p(class_weights)
    class_weights = log_weights / log_weights.max()

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


def compute_clusters_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cluster_col: str,
    *,
    noise_cluster_ids: list[int] | None = None,
) -> dict:
    """Aggregate cluster metadata across all splits."""
    df_ = pd.concat([train_df, val_df, test_df], ignore_index=True)
    clusters_distribution = {
        str(k): v for k, v in df_[cluster_col].value_counts().to_dict().items()
    }
    return {
        "clusters_distribution": clusters_distribution,
        "noise_cluster_ids": sorted(noise_cluster_ids) if noise_cluster_ids else [],
    }
