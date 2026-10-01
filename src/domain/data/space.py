from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Space:
    """The euclidean space geometry lives in: numerics as prepared, one-hot categoricals."""

    num_cols: list[str]
    # Per categorical column, the train codes with a slot of their own; the rest share one.
    kept: dict[str, list[int]]
    # Squared distance between two rows differing on one column. Numerics are IQR-scaled,
    # so 1 weighs a mismatch like one IQR step.
    cat_cost: float

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        *,
        num_cols: list[str],
        cat_cols: list[str],
        top_k: int,
        cat_cost: float,
    ) -> "Space":
        """The `top_k` most frequent codes of every categorical column of train."""
        kept = {
            col: [int(code) for code in train[col].value_counts().index[:top_k]]
            for col in cat_cols
        }
        return cls(list(num_cols), kept, cat_cost)

    @classmethod
    def from_record(cls, record: dict, *, num_cols: list[str]) -> "Space":
        """The space a stage saved with `to_record`."""
        kept = {row["column"]: row["codes"] for row in record["categories"]}
        return cls(list(num_cols), kept, record["cat_cost"])

    def to_record(self) -> dict:
        """What a stage saves of the space: a table of kept codes per column."""
        return {
            "cat_cost": self.cat_cost,
            "categories": [
                {"column": col, "codes": codes} for col, codes in self.kept.items()
            ],
        }

    def columns(self) -> list[str]:
        """Names of the coordinates `embed` returns, in order."""
        columns = list(self.num_cols)
        for col, codes in self.kept.items():
            columns += [f"{col}={code}" for code in codes] + [f"{col}=other"]
        return columns

    def embed(self, df: pd.DataFrame) -> np.ndarray:
        """The rows as points of the space."""
        numerics = df[self.num_cols].to_numpy(dtype=np.float64)
        if not self.kept:
            return numerics
        weight = np.sqrt(self.cat_cost / 2.0)
        # float32: the one-hot blocks are most of the matrix, and a class of millions of
        # rows has to fit in memory next to its copies.
        blocks = [numerics.astype(np.float32)]
        for col, codes in self.kept.items():
            slot = (
                df[col]
                .map({code: i for i, code in enumerate(codes)})
                .fillna(len(codes))
                .to_numpy(dtype=np.int64)
            )
            block = np.zeros((len(df), len(codes) + 1), dtype=np.float32)
            block[np.arange(len(df)), slot] = weight
            blocks.append(block)
        return np.hstack(blocks)
