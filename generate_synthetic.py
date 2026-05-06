"""Generates a synthetic dataset for ML pipeline testing.

Produces configurable-size data with 20 numerical features (num_1..num_20),
2 categorical features (cat_1, cat_2), and a label column with 11 classes.

Feature layout:
- num_1..num_15: per-class discriminating features
- num_16..num_20: intra-class sub-cluster features (40% of each class shifted to
  N(20, 1)), enabling per-class HDBSCAN to find ≥2 sub-clusters

Pipeline test behaviors:
- class_2 / class_3: identical numericals, differ only in cat_2 (hard separation)
- class_7 / class_8: same center on num_3,4,11,12; class_8 has wider variance (overlap)
- class_10: slight shift from class_1 on num_1,2,9 only (partial overlap)
- class_11: always 500 samples → filtered by rare_category_filter (min_count=3000)
- ~500 rows with NaN/Inf values → removed by drop_nans

See docs/synthetic_dataset.md for the full per-class specification.

Usage (from project root):
    python generate_synthetic.py [--rows N]
    make generate [ROWS=N]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path("resources/raw_data/synthetic/synthetic_test.csv")

_DEFAULT_CLASS_SIZES: dict[str, int] = {
    "class_1": 35_000,
    "class_2": 10_000,
    "class_3": 10_000,
    "class_4": 8_000,
    "class_5": 7_000,
    "class_6": 6_000,
    "class_7": 8_000,
    "class_8": 6_000,
    "class_9": 5_000,
    "class_10": 7_000,
    "class_11": 500,
}

_DEFAULT_TOTAL = sum(_DEFAULT_CLASS_SIZES.values())  # 102_500
_RARE_CLASS_SIZE = 500  # class_11 always kept rare (< min_cat_count=3000)

_CAT1_VALUES = ["A", "B", "C"]
_CAT2_VALUES = ["X", "Y", "Z", "W"]

RNG = np.random.default_rng(42)


def _normal(mu: float, sigma: float, n: int) -> np.ndarray:
    return RNG.normal(loc=mu, scale=sigma, size=n)


def _uniform(low: float, high: float, n: int) -> np.ndarray:
    return RNG.uniform(low=low, high=high, size=n)


def _cat(values: list, weights: list, n: int) -> np.ndarray:
    return RNG.choice(values, size=n, p=weights)


def _base_features(n: int) -> dict:
    """Baseline: all 20 features drawn from N(5, 1)."""
    return {f"num_{i}": _normal(5.0, 1.0, n) for i in range(1, 21)}


def _add_sub_cluster(f: dict, n: int) -> dict:
    """Shift num_16..num_20 in the last 40% of rows to N(20, 1).

    These 5 features are reserved solely for intra-class sub-cluster structure
    and are not used as per-class discriminating dimensions (num_1..num_15).
    5 shifted features survive PCA+preprocessing and give HDBSCAN two clear
    density peaks even for the smallest classes (around 4000 train samples).
    """
    n_b = int(n * 0.40)
    f = {k: v.copy() for k, v in f.items()}
    for col in ["num_16", "num_17", "num_18", "num_19", "num_20"]:
        f[col][-n_b:] = _normal(20.0, 1.0, n_b)
    return f


def _make_df(
    features: dict, cat1_w: list, cat2_w: list, label: str, n: int
) -> pd.DataFrame:
    df = pd.DataFrame(features)
    df["cat_1"] = _cat(_CAT1_VALUES, cat1_w, n)
    df["cat_2"] = _cat(_CAT2_VALUES, cat2_w, n)
    df["label"] = label
    return df


def generate_class_1(n: int) -> pd.DataFrame:
    """Majority baseline: all features N(5, 1)."""
    f = _add_sub_cluster(_base_features(n), n)
    return _make_df(f, [0.6, 0.3, 0.1], [0.4, 0.3, 0.15, 0.15], "class_1", n)


def generate_class_2(n: int) -> pd.DataFrame:
    """Hard-pair A: distinct block (num_1,2,5,6,11,12,13), cat_2=W exclusively."""
    f = _base_features(n)
    f["num_1"] = _normal(20.0, 1.0, n)
    f["num_2"] = _normal(18.0, 1.0, n)
    f["num_5"] = _normal(1.0, 0.3, n).clip(0)
    f["num_6"] = _normal(1.0, 0.3, n).clip(0)
    f["num_11"] = _uniform(1.0, 3.0, n)
    f["num_12"] = _uniform(1.0, 3.0, n)
    f["num_13"] = _normal(18.0, 0.8, n)
    return _make_df(
        _add_sub_cluster(f, n), [0.8, 0.15, 0.05], [0.0, 0.0, 0.0, 1.0], "class_2", n
    )


def generate_class_3(n: int) -> pd.DataFrame:
    """Hard-pair B: identical numericals to class_2, cat_2=Z exclusively."""
    f = _base_features(n)
    f["num_1"] = _normal(20.0, 1.0, n)
    f["num_2"] = _normal(18.0, 1.0, n)
    f["num_5"] = _normal(1.0, 0.3, n).clip(0)
    f["num_6"] = _normal(1.0, 0.3, n).clip(0)
    f["num_11"] = _uniform(1.0, 3.0, n)
    f["num_12"] = _uniform(1.0, 3.0, n)
    f["num_13"] = _normal(18.0, 0.8, n)
    return _make_df(
        _add_sub_cluster(f, n), [0.8, 0.15, 0.05], [0.0, 0.0, 1.0, 0.0], "class_3", n
    )


def generate_class_4(n: int) -> pd.DataFrame:
    """Block A: num_1,2,7,14,15 high; num_5,6 near-zero."""
    f = _base_features(n)
    f["num_1"] = _normal(30.0, 1.5, n)
    f["num_2"] = _normal(28.0, 1.5, n)
    f["num_7"] = _normal(25.0, 1.5, n)
    f["num_14"] = _normal(22.0, 1.0, n)
    f["num_15"] = _normal(20.0, 1.0, n)
    f["num_5"] = _uniform(0.0, 0.5, n)
    f["num_6"] = _uniform(0.0, 0.5, n)
    return _make_df(
        _add_sub_cluster(f, n), [0.5, 0.4, 0.1], [0.5, 0.4, 0.05, 0.05], "class_4", n
    )


def generate_class_5(n: int) -> pd.DataFrame:
    """Block B: num_3,4,8,9,10 high; num_10 tightly concentrated."""
    f = _base_features(n)
    f["num_3"] = _normal(28.0, 1.0, n)
    f["num_4"] = _normal(25.0, 1.0, n)
    f["num_8"] = _normal(22.0, 1.0, n)
    f["num_9"] = _normal(20.0, 1.0, n)
    f["num_10"] = _normal(35.0, 0.3, n)
    return _make_df(
        _add_sub_cluster(f, n), [0.4, 0.5, 0.1], [0.3, 0.5, 0.1, 0.1], "class_5", n
    )


def generate_class_6(n: int) -> pd.DataFrame:
    """Block C: num_5,6,9 very high; num_1,7 near-zero."""
    f = _base_features(n)
    f["num_5"] = _normal(35.0, 2.0, n)
    f["num_6"] = _normal(30.0, 2.0, n)
    f["num_9"] = _normal(28.0, 2.0, n)
    f["num_1"] = _uniform(0.0, 0.5, n)
    f["num_7"] = _normal(0.3, 0.1, n).clip(0)
    return _make_df(
        _add_sub_cluster(f, n), [0.7, 0.2, 0.1], [0.6, 0.1, 0.1, 0.2], "class_6", n
    )


def generate_class_7(n: int) -> pd.DataFrame:
    """Overlap-pair A: moderate shift on num_3,4,11,12."""
    f = _base_features(n)
    f["num_3"] = _normal(14.0, 1.5, n)
    f["num_4"] = _normal(13.0, 1.5, n)
    f["num_11"] = _normal(15.0, 1.5, n)
    f["num_12"] = _normal(13.0, 1.0, n)
    return _make_df(
        _add_sub_cluster(f, n), [0.55, 0.35, 0.1], [0.35, 0.25, 0.2, 0.2], "class_7", n
    )


def generate_class_8(n: int) -> pd.DataFrame:
    """Overlap-pair B: same center as class_7, wider variance → intentional overlap."""
    f = _base_features(n)
    f["num_3"] = _normal(14.0, 4.0, n).clip(0)
    f["num_4"] = _normal(13.0, 4.0, n).clip(0)
    f["num_11"] = _normal(15.0, 1.5, n)
    f["num_12"] = _normal(13.0, 2.5, n)
    return _make_df(
        _add_sub_cluster(f, n), [0.55, 0.35, 0.1], [0.3, 0.3, 0.2, 0.2], "class_8", n
    )


def generate_class_9(n: int) -> pd.DataFrame:
    """Block D: num_5,6 extreme; num_1,2 near-zero."""
    f = _base_features(n)
    f["num_5"] = _normal(50.0, 4.0, n)
    f["num_6"] = _normal(55.0, 5.0, n)
    f["num_1"] = _uniform(0.0, 0.3, n)
    f["num_2"] = _normal(0.3, 0.1, n).clip(0)
    return _make_df(
        _add_sub_cluster(f, n), [0.7, 0.2, 0.1], [0.5, 0.1, 0.1, 0.3], "class_9", n
    )


def generate_class_10(n: int) -> pd.DataFrame:
    """Partial overlap with class_1: slight shift on num_1,2,9 only."""
    f = _base_features(n)
    f["num_1"] = _normal(7.0, 1.2, n)
    f["num_2"] = _normal(7.5, 1.2, n)
    f["num_9"] = _normal(7.0, 1.0, n)
    return _make_df(
        _add_sub_cluster(f, n), [0.6, 0.3, 0.1], [0.7, 0.1, 0.05, 0.15], "class_10", n
    )


def generate_class_11(n: int) -> pd.DataFrame:
    """Rare class: filtered by rare_category_filter (n < min_cat_count=3000)."""
    f = _base_features(n)
    return _make_df(f, [0.33, 0.34, 0.33], [0.25, 0.25, 0.25, 0.25], "class_11", n)


def inject_edge_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Inject ~300 Inf and ~200 NaN values into random numerical positions."""
    num_cols = [c for c in df.columns if c.startswith("num_")]
    df = df.copy()

    inf_idx = RNG.choice(df.index, size=300, replace=False)
    inf_cols = RNG.choice(["num_7", "num_8"], size=300)
    for idx, col in zip(inf_idx, inf_cols):
        df.at[idx, col] = np.inf

    nan_idx = RNG.choice(df.index, size=200, replace=False)
    nan_cols = RNG.choice(num_cols, size=200)
    for idx, col in zip(nan_idx, nan_cols):
        df.at[idx, col] = np.nan

    return df


def _compute_sizes(total_rows: int) -> dict[str, int]:
    """Scale class sizes proportionally to total_rows, keeping class_11 rare."""
    scalable_total = _DEFAULT_TOTAL - _RARE_CLASS_SIZE
    scale = (total_rows - _RARE_CLASS_SIZE) / scalable_total
    return {
        cls: _RARE_CLASS_SIZE if cls == "class_11" else max(1, round(n * scale))
        for cls, n in _DEFAULT_CLASS_SIZES.items()
    }


_GENERATORS = [
    generate_class_1,
    generate_class_2,
    generate_class_3,
    generate_class_4,
    generate_class_5,
    generate_class_6,
    generate_class_7,
    generate_class_8,
    generate_class_9,
    generate_class_10,
    generate_class_11,
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic test dataset.")
    parser.add_argument(
        "--rows",
        type=int,
        default=_DEFAULT_TOTAL,
        help=f"Target total row count (default: {_DEFAULT_TOTAL:,}). "
        "class_11 is always kept at 500 rows to preserve the rare-class filter test.",
    )
    args = parser.parse_args()

    sizes = _compute_sizes(args.rows)
    print(f"Generating synthetic dataset ({sum(sizes.values()):,} rows)...")

    dfs = [gen(sizes[f"class_{i + 1}"]) for i, gen in enumerate(_GENERATORS)]
    df = pd.concat(dfs, ignore_index=True)
    df = inject_edge_cases(df)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    num_cols = [c for c in df.columns if c.startswith("num_")]
    inf_count = int((df[num_cols] == np.inf).sum().sum())

    print(f"Saved:     {OUTPUT_PATH}")
    print(f"Shape:     {df.shape}")
    print(f"NaN count: {df.isna().sum().sum()}")
    print(f"Inf count: {inf_count}")
    print("\nClass distribution:")
    print(df["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
