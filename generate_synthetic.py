import argparse
from dataclasses import dataclass, field
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

_DEFAULT_TOTAL = sum(_DEFAULT_CLASS_SIZES.values())
_RARE_CLASS_SIZE = 500

_CAT1_VALUES = ["A", "B", "C"]
_CAT2_VALUES = ["X", "Y", "Z", "W"]

_N_FEATURES = 20
_BASE_MU = 5.0
_BASE_SIGMA = 1.0
_MIX_SIGMA = 1.5

RNG = np.random.default_rng(42)


@dataclass(frozen=True)
class ClassSpec:
    """Core feature means, core spread and categorical profiles of one class."""

    mu: dict[str, float] = field(default_factory=dict)
    sigma: float = _BASE_SIGMA
    cat_1: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)
    cat_2: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)


@dataclass(frozen=True)
class Rung:
    """One step of the difficulty ladder: mass share, margin in sigmas, categorical blend."""

    name: str
    share: float
    margin: float
    blend: float


_CANONICAL_SHARE = 0.34

_LADDER = (
    Rung("clear", 0.28, 1.80, 0.25),
    Rung("evasive", 0.24, 0.90, 0.65),
    Rung("mimicry", 0.14, 0.25, 1.00),
)

_CLASSES: dict[str, ClassSpec] = {
    "class_1": ClassSpec(
        mu={"num_18": 24.0, "num_19": 20.0, "num_20": 18.0},
        cat_1=(0.60, 0.30, 0.10),
        cat_2=(0.40, 0.35, 0.15, 0.10),
    ),
    "class_2": ClassSpec(
        mu={"num_1": 25.0, "num_2": 22.0, "num_3": 20.0},
        cat_1=(0.80, 0.15, 0.05),
        cat_2=(0.05, 0.10, 0.10, 0.75),
    ),
    "class_3": ClassSpec(
        mu={"num_1": 24.0, "num_2": 20.0, "num_4": 22.0},
        cat_1=(0.75, 0.20, 0.05),
        cat_2=(0.05, 0.10, 0.35, 0.50),
    ),
    "class_4": ClassSpec(
        mu={"num_5": 30.0, "num_6": 28.0, "num_7": 25.0},
        cat_1=(0.50, 0.40, 0.10),
        cat_2=(0.50, 0.35, 0.10, 0.05),
    ),
    "class_5": ClassSpec(
        mu={"num_8": 20.0, "num_9": 18.0, "num_10": 22.0},
        cat_1=(0.40, 0.50, 0.10),
        cat_2=(0.30, 0.50, 0.10, 0.10),
    ),
    "class_6": ClassSpec(
        mu={"num_8": 22.0, "num_9": 14.0, "num_11": 20.0},
        sigma=2.0,
        cat_1=(0.35, 0.50, 0.15),
        cat_2=(0.30, 0.50, 0.10, 0.10),
    ),
    "class_7": ClassSpec(
        mu={"num_12": 28.0, "num_13": 25.0, "num_14": 22.0},
        cat_1=(0.55, 0.35, 0.10),
        cat_2=(0.35, 0.30, 0.20, 0.15),
    ),
    "class_8": ClassSpec(
        mu={"num_16": 11.0, "num_17": 10.0},
        cat_1=(0.70, 0.20, 0.10),
        cat_2=(0.02, 0.03, 0.05, 0.90),
    ),
    "class_9": ClassSpec(
        mu={"num_1": 18.0, "num_2": 16.0, "num_15": 15.0},
        cat_1=(0.60, 0.30, 0.10),
        cat_2=(0.70, 0.10, 0.10, 0.10),
    ),
    "class_10": ClassSpec(),
}

_OVERLAPS = (
    ("class_1", "class_2"),
    ("class_1", "class_3"),
    ("class_1", "class_4"),
    ("class_1", "class_7"),
    ("class_1", "class_8"),
    ("class_1", "class_9"),
    ("class_2", "class_3"),
    ("class_5", "class_6"),
)


def _n(mu: float, sigma: float, n: int) -> np.ndarray:
    """Draw n normal samples."""
    return RNG.normal(loc=mu, scale=sigma, size=n)


def _cat(values: list, weights: tuple[float, ...], n: int) -> np.ndarray:
    """Draw n categorical samples with the given weights."""
    return RNG.choice(values, size=n, p=weights)


def _base(n: int) -> dict:
    """All numerical features at the shared baseline N(5, 1)."""
    return {f"num_{i}": _n(_BASE_MU, _BASE_SIGMA, n) for i in range(1, _N_FEATURES + 1)}


def _adversaries(label: str) -> list[str]:
    """Classes sharing a difficulty corridor with `label`, in declaration order."""
    return [b if a == label else a for a, b in _OVERLAPS if label in (a, b)]


def _rung_center(
    mu: dict[str, float], adv_mu: dict[str, float], margin: float
) -> dict[str, float]:
    """Means placed `margin` mixing sigmas from the pair midpoint, on the own-class side."""
    keys = sorted(set(mu) | set(adv_mu))
    own = np.array([mu.get(k, _BASE_MU) for k in keys])
    adv = np.array([adv_mu.get(k, _BASE_MU) for k in keys])
    direction = own - adv
    norm = float(np.linalg.norm(direction))
    offset = margin * _MIX_SIGMA * direction / norm if norm > 0 else 0.0
    return dict(zip(keys, (own + adv) / 2 + offset))


def _blend_weights(
    weights: tuple[float, ...], adv_weights: tuple[float, ...], blend: float
) -> tuple[float, ...]:
    """Categorical profile moved `blend` of the way to the pair's shared 50/50 mixture."""
    mixed = [
        (1 - blend / 2) * w + (blend / 2) * a for w, a in zip(weights, adv_weights)
    ]
    total = sum(mixed)
    return tuple(w / total for w in mixed)


def _split_counts(n: int, shares: list[float]) -> list[int]:
    """Split n rows across shares, giving the rounding remainder to the first group."""
    total = sum(shares)
    counts = [int(n * share / total) for share in shares]
    counts[0] += n - sum(counts)
    return counts


def _subgroup_frame(
    label: str,
    subgroup: str,
    n: int,
    center: dict[str, float],
    sigma: float,
    cat_1: tuple[float, ...],
    cat_2: tuple[float, ...],
) -> pd.DataFrame:
    """One sub-group: baseline features with `center` overridden, plus categoricals and label."""
    features = _base(n)
    for col, mu in center.items():
        features[col] = _n(mu, sigma, n)
    df = pd.DataFrame(features)
    df["cat_1"] = _cat(_CAT1_VALUES, cat_1, n)
    df["cat_2"] = _cat(_CAT2_VALUES, cat_2, n)
    df["label"] = label
    df["true_subgroup"] = subgroup
    return df


def generate_class(label: str, n: int) -> pd.DataFrame:
    """Canonical core plus one margin-graded ladder toward every overlapping class."""
    spec = _CLASSES[label]
    adversaries = _adversaries(label)
    if not adversaries:
        return _subgroup_frame(
            label, "canonical", n, spec.mu, spec.sigma, spec.cat_1, spec.cat_2
        )

    plan = [
        (
            "canonical",
            _CANONICAL_SHARE,
            spec.mu,
            spec.sigma,
            spec.cat_1,
            spec.cat_2,
        )
    ]
    for adversary in adversaries:
        adv = _CLASSES[adversary]
        for rung in _LADDER:
            plan.append(
                (
                    rung.name,
                    rung.share / len(adversaries),
                    _rung_center(spec.mu, adv.mu, rung.margin),
                    _MIX_SIGMA,
                    _blend_weights(spec.cat_1, adv.cat_1, rung.blend),
                    _blend_weights(spec.cat_2, adv.cat_2, rung.blend),
                )
            )

    counts = _split_counts(n, [share for _, share, *_ in plan])
    return pd.concat(
        [
            _subgroup_frame(label, name, count, center, sigma, cat_1, cat_2)
            for (name, _, center, sigma, cat_1, cat_2), count in zip(plan, counts)
            if count > 0
        ],
        ignore_index=True,
    )


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


def build_dataset(total_rows: int) -> pd.DataFrame:
    """Build the full shuffled dataset with edge cases injected."""
    sizes = _compute_sizes(total_rows)
    df = pd.concat(
        [generate_class(label, sizes[label]) for label in _CLASSES],
        ignore_index=True,
    )
    df = inject_edge_cases(df)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main() -> None:
    """Entry point: build the synthetic dataset and write it to `OUTPUT_PATH`."""
    parser = argparse.ArgumentParser(description="Generate synthetic test dataset.")
    parser.add_argument(
        "--rows",
        type=int,
        default=_DEFAULT_TOTAL,
        help=f"Target total row count (default: {_DEFAULT_TOTAL:,}). "
        "class_10 is always kept at 500 rows to preserve the rare-class filter test.",
    )
    args = parser.parse_args()

    print(f"Generating synthetic dataset ({args.rows:,} rows)...")
    df = build_dataset(args.rows)

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
