from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.core.paths import OutputPaths
from src.domain.data.preprocessing import random_undersample_df


@dataclass
class _Split:
    """One split: the training rows and the universe positions it evaluates."""

    train_df: pd.DataFrame
    fold_dir: Path
    eval_idx: np.ndarray


def _oof_splits(base: pd.DataFrame, label_col: str, k: int, seed: int) -> list:
    """Deterministic stratified OOF folds over `base`; K capped to the rarest class."""
    y = base[label_col].to_numpy()
    k = min(k, int(np.unique(y, return_counts=True)[1].min()))
    if k < 2:
        raise ValueError(f"k-fold OOF needs >=2 samples per class, got k={k}.")
    return list(
        StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(base, y)
    )


def _build_universe_and_splits(
    cfg,
    paths: OutputPaths,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
) -> tuple[pd.DataFrame, list[_Split], str]:
    """Build the evaluation universe and its splits: the single test split, or k-fold OOF."""
    universe = pd.concat([train_df, test_df], ignore_index=True)

    if not cfg.kfold:
        eval_idx = np.arange(len(train_df), len(universe))
        return universe, [_Split(train_df, paths.models, eval_idx)], "single_split"

    splits = []
    for f, (tr_idx, te_idx) in enumerate(
        _oof_splits(universe, label_col, cfg.kfold_splits, cfg.seed)
    ):
        fold_train = universe.iloc[tr_idx]
        if cfg.balance == "undersample":
            fold_train = random_undersample_df(
                fold_train, label_col, random_state=cfg.seed
            )
        splits.append(_Split(fold_train, paths.models / f"fold_{f}", te_idx))
    return universe, splits, "oof_kfold"
