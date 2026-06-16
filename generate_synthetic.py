import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path("resources/raw_data/synthetic/synthetic_test.csv")

_DEFAULT_CLASS_SIZES: dict[str, int] = {
    "class_1": 22_000,
    "class_2": 9_000,
    "class_3": 7_000,
    "class_4": 6_000,
    "class_5": 5_000,
    "class_6": 5_000,
    "class_7": 5_000,
    "class_8": 5_000,
    "class_9": 5_000,
    "class_10": 500,
}

_DEFAULT_TOTAL = sum(_DEFAULT_CLASS_SIZES.values())  # 69_500
_RARE_CLASS_SIZE = 500

_CAT1_VALUES = ["A", "B", "C"]
_CAT2_VALUES = ["X", "Y", "Z", "W"]

RNG = np.random.default_rng(42)


def _n(mu: float, sigma: float, n: int) -> np.ndarray:
    return RNG.normal(loc=mu, scale=sigma, size=n)


def _cat(values: list, weights: list, n: int) -> np.ndarray:
    return RNG.choice(values, size=n, p=weights)


def _base(n: int) -> dict:
    """All 20 features at the benign baseline N(5, 1)."""
    return {f"num_{i}": _n(5.0, 1.0, n) for i in range(1, 21)}


def _interp(
    class_mu: dict[str, float], adv_mu: dict[str, float], alpha: float
) -> dict[str, float]:
    """Interpolate feature means: (1 - alpha) * class + alpha * adversary.

    Features not listed in class_mu default to 5.0 (benign baseline).
    alpha=0 → canonical (at class center); alpha=1 → at adversary center.
    """
    keys = set(class_mu) | set(adv_mu)
    return {
        k: (1 - alpha) * class_mu.get(k, 5.0) + alpha * adv_mu.get(k, 5.0) for k in keys
    }


def _subgroup(
    n: int,
    class_mu: dict[str, float],
    adv_mu: dict[str, float],
    alpha: float,
    sigma: float,
) -> dict:
    """Generate features for one sub-group.

    Non-discriminating features stay at baseline N(5, 1); discriminating features
    are drawn at the interpolated center with the given sigma.
    """
    f = _base(n)
    for col, mu in _interp(class_mu, adv_mu, alpha).items():
        f[col] = _n(mu, sigma, n)
    return f


def _assemble(
    label: str, parts: list[tuple[dict, str]], cat1_w: list, cat2_w: list
) -> pd.DataFrame:
    """Concatenate sub-group feature dicts, attach categoricals and label."""
    dfs = []
    for features, sg in parts:
        df = pd.DataFrame(features)
        df["true_subgroup"] = sg
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    n = len(df)
    df["cat_1"] = _cat(_CAT1_VALUES, cat1_w, n)
    df["cat_2"] = _cat(_CAT2_VALUES, cat2_w, n)
    df["label"] = label
    return df


# ── Canonical class centers (discriminating features only; baseline = 5.0) ──────────

# class_2: attack A — num_1, num_2, num_3
_MU2 = {"num_1": 25.0, "num_2": 22.0, "num_3": 20.0}
# class_3: attack B, variant — shares num_1, num_2 with class_2; uses num_4 not num_3
_MU3 = {"num_1": 24.0, "num_2": 20.0, "num_4": 22.0}
# class_4: attack C — distinct feature block
_MU4 = {"num_5": 30.0, "num_6": 28.0, "num_7": 25.0}
# class_5 / class_6: overlap pair — adjacent clusters in num_8/num_9 space
_MU5 = {"num_8": 20.0, "num_9": 18.0, "num_10": 22.0}
_MU6 = {"num_8": 22.0, "num_9": 14.0, "num_11": 20.0}
# class_7: attack D — another separate block
_MU7 = {"num_12": 28.0, "num_13": 25.0, "num_14": 22.0}
# class_9: stealth — uses num_1/num_2 (partial overlap with class_2/3) + num_15
_MU9 = {"num_1": 18.0, "num_2": 16.0, "num_15": 15.0}
# Benign baseline center (all at 5.0)
_MU1: dict[str, float] = {}


def generate_class_1(n: int) -> pd.DataFrame:
    """Majority benign: canonical at baseline + near-attack sub-group toward class_2."""
    n_a, n_b = int(n * 0.80), n - int(n * 0.80)
    f_a = _base(n_a)
    f_b = _subgroup(n_b, _MU1, _MU2, alpha=0.30, sigma=1.2)
    return _assemble(
        "class_1",
        [(f_a, "canonical"), (f_b, "evasive")],
        [0.6, 0.3, 0.1],
        [0.4, 0.35, 0.15, 0.10],
    )


def generate_class_2(n: int) -> pd.DataFrame:
    """Attack A (clear): full difficulty gradient toward class_1."""
    n_a, n_b = int(n * 0.50), int(n * 0.30)
    n_c = n - n_a - n_b
    return _assemble(
        "class_2",
        [
            (_subgroup(n_a, _MU2, _MU1, alpha=0.00, sigma=1.5), "canonical"),
            (_subgroup(n_b, _MU2, _MU1, alpha=0.50, sigma=1.5), "medium"),
            (_subgroup(n_c, _MU2, _MU1, alpha=0.80, sigma=1.5), "evasive"),
        ],
        [0.8, 0.15, 0.05],
        [0.05, 0.10, 0.10, 0.75],
    )


def generate_class_3(n: int) -> pd.DataFrame:
    """Attack B (variant of A): shares num_1/num_2 with class_2; evasive toward class_1."""
    n_a, n_b = int(n * 0.50), int(n * 0.30)
    n_c = n - n_a - n_b
    return _assemble(
        "class_3",
        [
            (_subgroup(n_a, _MU3, _MU1, alpha=0.00, sigma=1.5), "canonical"),
            (_subgroup(n_b, _MU3, _MU1, alpha=0.50, sigma=1.5), "medium"),
            (_subgroup(n_c, _MU3, _MU1, alpha=0.80, sigma=1.5), "evasive"),
        ],
        [0.75, 0.20, 0.05],
        [0.05, 0.10, 0.35, 0.50],
    )


def generate_class_4(n: int) -> pd.DataFrame:
    """Attack C (well separated): distinct feature block; evasive toward class_1."""
    n_a, n_b = int(n * 0.50), int(n * 0.30)
    n_c = n - n_a - n_b
    return _assemble(
        "class_4",
        [
            (_subgroup(n_a, _MU4, _MU1, alpha=0.00, sigma=1.5), "canonical"),
            (_subgroup(n_b, _MU4, _MU1, alpha=0.50, sigma=1.5), "medium"),
            (_subgroup(n_c, _MU4, _MU1, alpha=0.80, sigma=1.5), "evasive"),
        ],
        [0.5, 0.4, 0.1],
        [0.5, 0.35, 0.10, 0.05],
    )


def generate_class_5(n: int) -> pd.DataFrame:
    """Overlap-pair A: evasive sub-group falls deep into class_6 territory."""
    n_a, n_b = int(n * 0.50), int(n * 0.30)
    n_c = n - n_a - n_b
    return _assemble(
        "class_5",
        [
            (_subgroup(n_a, _MU5, _MU6, alpha=0.00, sigma=1.5), "canonical"),
            (_subgroup(n_b, _MU5, _MU6, alpha=0.50, sigma=1.5), "medium"),
            (_subgroup(n_c, _MU5, _MU6, alpha=0.80, sigma=1.5), "evasive"),
        ],
        [0.4, 0.5, 0.1],
        [0.3, 0.5, 0.1, 0.1],
    )


def generate_class_6(n: int) -> pd.DataFrame:
    """Overlap-pair B: wider variance; evasive sub-group overlaps class_5 deeply."""
    n_a, n_b = int(n * 0.50), int(n * 0.30)
    n_c = n - n_a - n_b
    return _assemble(
        "class_6",
        [
            (_subgroup(n_a, _MU6, _MU5, alpha=0.00, sigma=2.0), "canonical"),
            (_subgroup(n_b, _MU6, _MU5, alpha=0.50, sigma=2.0), "medium"),
            (_subgroup(n_c, _MU6, _MU5, alpha=0.80, sigma=2.0), "evasive"),
        ],
        [0.35, 0.50, 0.15],
        [0.3, 0.5, 0.1, 0.1],
    )


def generate_class_7(n: int) -> pd.DataFrame:
    """Attack D (well separated 2): different feature block from class_4; evasive toward class_1."""
    n_a, n_b = int(n * 0.50), int(n * 0.30)
    n_c = n - n_a - n_b
    return _assemble(
        "class_7",
        [
            (_subgroup(n_a, _MU7, _MU1, alpha=0.00, sigma=1.5), "canonical"),
            (_subgroup(n_b, _MU7, _MU1, alpha=0.50, sigma=1.5), "medium"),
            (_subgroup(n_c, _MU7, _MU1, alpha=0.80, sigma=1.5), "evasive"),
        ],
        [0.55, 0.35, 0.10],
        [0.35, 0.30, 0.20, 0.15],
    )


def generate_class_8(n: int) -> pd.DataFrame:
    """Categorical: numerically near class_1; cat_2=W exclusively is the separator.

    Two sub-groups via num_15 drift: canonical at baseline, medium shifted to N(10, 1.5).
    This gives HDBSCAN two distinguishable density peaks within the class.
    Evasive sub-group omitted: moving numerically toward class_1 is impossible here
    (the class IS numerically at class_1); the challenge is purely categorical.
    """
    n_a, n_b = int(n * 0.60), n - int(n * 0.60)
    f_a = _base(n_a)
    f_b = _base(n_b)
    f_b["num_15"] = _n(10.0, 1.5, n_b)
    return _assemble(
        "class_8",
        [(f_a, "canonical"), (f_b, "medium")],
        [0.7, 0.2, 0.1],
        [0.0, 0.0, 0.0, 1.0],
    )


def generate_class_9(n: int) -> pd.DataFrame:
    """Stealth: majority evasive — most samples are near-benign; canonical is exceptional."""
    n_a, n_b = int(n * 0.20), int(n * 0.40)
    n_c = n - n_a - n_b
    return _assemble(
        "class_9",
        [
            (_subgroup(n_a, _MU9, _MU1, alpha=0.00, sigma=1.5), "canonical"),
            (_subgroup(n_b, _MU9, _MU1, alpha=0.50, sigma=1.2), "medium"),
            (_subgroup(n_c, _MU9, _MU1, alpha=0.80, sigma=1.0), "evasive"),
        ],
        [0.6, 0.3, 0.1],
        [0.7, 0.1, 0.1, 0.1],
    )


def generate_class_10(n: int) -> pd.DataFrame:
    """Rare class: always 500 rows → removed by rare_category_filter (< min_cat_count=3000)."""
    df = pd.DataFrame(_base(n))
    df["cat_1"] = _cat(_CAT1_VALUES, [0.33, 0.34, 0.33], n)
    df["cat_2"] = _cat(_CAT2_VALUES, [0.25, 0.25, 0.25, 0.25], n)
    df["label"] = "class_10"
    df["true_subgroup"] = "canonical"
    return df


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
    """Scale class sizes proportionally to total_rows, keeping class_10 rare."""
    scalable_total = _DEFAULT_TOTAL - _RARE_CLASS_SIZE
    scale = (total_rows - _RARE_CLASS_SIZE) / scalable_total
    return {
        cls: _RARE_CLASS_SIZE if cls == "class_10" else max(1, round(n * scale))
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
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic test dataset.")
    parser.add_argument(
        "--rows",
        type=int,
        default=_DEFAULT_TOTAL,
        help=f"Target total row count (default: {_DEFAULT_TOTAL:,}). "
        "class_10 is always kept at 500 rows to preserve the rare-class filter test.",
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
    print("\nSub-group distribution:")
    print(df.groupby("label")["true_subgroup"].value_counts().to_string())


if __name__ == "__main__":
    main()
