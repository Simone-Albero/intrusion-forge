import numpy as np
import pandas as pd


def get_df_info(df: pd.DataFrame, *, label_col: str) -> dict:
    """Size of the raw frame, plus one row per class."""
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "classes": [
            {"class_name": str(name), "n_rows": int(n)}
            for name, n in df[label_col].value_counts().items()
        ],
    }


def compute_df_metadata(
    splits: dict[str, pd.DataFrame],
    *,
    label_col: str,
    num_cols: list[str],
    cat_cols: list[str],
    benign_tag: str,
    label_mapping: dict[int, str],
) -> dict:
    """Run scalars plus split and class tables, read from the encoded `label_col`."""
    counts = {tag: df[label_col].value_counts() for tag, df in splits.items()}
    train_counts = counts["train"]
    weights = len(splits["train"]) / (len(train_counts) * train_counts)
    weights = np.log1p(weights) / np.log1p(weights).max()

    return {
        "benign_tag": benign_tag,
        "n_classes": int(splits["train"][label_col].nunique()),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "splits": [{"split": tag, "n_rows": len(df)} for tag, df in splits.items()],
        "classes": [
            {
                "class_id": int(class_id),
                "class_name": name,
                "weight": float(weights[class_id]),
                **{f"n_{tag}": int(c.get(class_id, 0)) for tag, c in counts.items()},
            }
            for class_id, name in sorted(label_mapping.items())
        ],
    }


def compute_clusters_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    cluster_col: str,
    noise_cluster_ids: list[int],
) -> dict:
    """Noise cluster ids, plus one row per cluster with its size across all splits."""
    sizes = pd.concat([train_df, val_df, test_df])[cluster_col].value_counts()
    return {
        "noise_cluster_ids": sorted(noise_cluster_ids),
        "clusters": [
            {"cluster_id": int(cid), "n_rows": int(n)}
            for cid, n in sorted(sizes.items())
        ],
    }
