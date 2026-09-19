from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class ComponentSpec:
    """Registered component name plus the params it is built with."""

    name: str | None
    params: dict[str, Any] = field(default_factory=dict)


class Trainer(Protocol):
    """Fits, persists and runs a classifier of one kind."""

    num_cols: list[str]
    cat_cols: list[str]

    def features(self, df: pd.DataFrame, feat_cols: list[str]) -> Any:
        """Shape the feature side of `df` the way this trainer's predict expects it."""

    def prepare(
        self, df: pd.DataFrame, feat_cols: list[str], label_col: str
    ) -> tuple[Any, Any]:
        """Shape (X, y) the way this trainer's fit expects them."""

    def fit(
        self,
        name: str,
        params: dict,
        X: Any,
        y: Any,
        *,
        X_val: pd.DataFrame,
        save_dir: Path,
    ) -> tuple[Any, dict]:
        """Fit a classifier on (X, y) and return (model, summary)."""

    def grid_search(
        self,
        name: str,
        params: dict,
        grid: dict,
        X: Any,
        y: Any,
        *,
        scoring: str,
        cv: int,
        max_samples: int | None,
        random_state: int,
    ) -> tuple[Any, dict]:
        """Cross-validated grid search returning (best model, summary)."""

    def predict(self, model: Any, X: Any, *, return_embedding: bool = False) -> tuple:
        """Predict `X` → (y_pred, y_proba), plus the latent embedding on request."""

    def save(self, model: Any, path: Path, *, name: str, params: dict) -> None:
        """Persist `model` under `path`."""

    def load(self, path: Path) -> Any:
        """Load the model persisted under `path`."""
