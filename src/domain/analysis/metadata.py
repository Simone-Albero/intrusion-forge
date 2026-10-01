import pandas as pd


def build_meta(
    labels: dict[str, pd.Series],
    *,
    num_cols: list[str],
    cat_cols: list[str],
    label_mapping: dict[int, str],
    raw_classes: pd.Series,
    n_raw_rows: int,
    n_raw_columns: int,
) -> dict:
    """What the later stages need to know of the split: columns, classes and sizes."""
    counts = {split: series.value_counts() for split, series in labels.items()}
    return {
        "n_classes": len(label_mapping),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "n_raw_rows": n_raw_rows,
        "n_raw_columns": n_raw_columns,
        "splits": [
            {"split": split, "n_rows": len(series)} for split, series in labels.items()
        ],
        "classes": [
            {
                "class_id": int(class_id),
                "class_name": name,
                **{
                    f"n_{split}": int(c.get(class_id, 0)) for split, c in counts.items()
                },
            }
            for class_id, name in sorted(label_mapping.items())
        ],
        # Before the rare-class filter: which classes it dropped stays visible.
        "raw_classes": [
            {"class_name": str(name), "n_rows": int(n)}
            for name, n in raw_classes.items()
        ],
    }
