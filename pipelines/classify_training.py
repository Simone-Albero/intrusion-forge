import importlib
import inspect
import logging
from pathlib import Path
from types import ModuleType

import pandas as pd
import torch
from omegaconf import OmegaConf

from pipelines.classify_splits import _Split
from src.core.log import LogBundle, LogDispatcher
from src.core.paths import OutputPaths
from src.core.utils import timed
from src.domain.plot.base import Plot
from src.domain.plot.primitives import line_plot
from src.engine.ml.model import MLClassifierFactory

logger = logging.getLogger(__name__)


def _supports_random_state(clf_cls: type) -> bool:
    """True if the estimator accepts a `random_state` parameter."""
    if "random_state" in inspect.signature(clf_cls.__init__).parameters:
        return True
    try:
        return "random_state" in clf_cls().get_params()
    except Exception:
        return False


def _training_history_figures(history: dict[str, list[float]]) -> dict[str, Plot]:
    """One line plot per scalar in the per-step DL training history."""
    return {
        f"figure/training/{name}_curve": line_plot(
            {name: values},
            y_label=name,
            show_legend=False,
        )
        for name, values in history.items()
        if values
    }


_TRAINING_MODULES = {
    "ml": "src.domain.training.ml",
    "dl": "src.domain.training.dl",
}


def _resolve_training_module(kind: str) -> ModuleType:
    """Return the training module matching the classifier kind."""
    if kind not in _TRAINING_MODULES:
        raise ValueError(
            f"Unknown classifier kind: {kind!r}. "
            f"Expected one of {sorted(_TRAINING_MODULES)}."
        )
    return importlib.import_module(_TRAINING_MODULES[kind])


def _build_dl_context(
    cfg,
    paths: OutputPaths,
    df_meta: dict,
    num_cols: list[str],
    cat_cols: list[str],
    label_col: str,
) -> dict:
    """DL training context."""
    return {
        "device": torch.device(cfg.device),
        "df_meta": df_meta,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "label_col": label_col,
        "loss_cfg": cfg.loss,
        "optimizer_cfg": cfg.optimizer,
        "scheduler_cfg": cfg.scheduler,
        "loops_cfg": cfg.loops,
        "models_path": paths.models,
    }


def _prepare_train_payload(
    kind: str,
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
) -> tuple[object, object]:
    """Shape (X, y) for the training module: a feature slice for ML, the full frame for DL."""
    if kind == "ml":
        return df[feat_cols], df[label_col].to_numpy()
    return df, None


def _build_ml_context(num_cols: list[str], cat_cols: list[str]) -> dict:
    """ML training context."""
    return {"num_cols": num_cols, "cat_cols": cat_cols}


def _build_context(
    cfg,
    paths: OutputPaths,
    df_meta: dict,
    num_cols: list[str],
    cat_cols: list[str],
    label_col: str,
) -> dict:
    """Build the training context matching the classifier kind."""
    if cfg.classifier.kind == "dl":
        return _build_dl_context(cfg, paths, df_meta, num_cols, cat_cols, label_col)
    return _build_ml_context(num_cols, cat_cols)


def _resolve_dl_params(
    name: str,
    params: dict,
    num_cols: list[str],
    cat_cols: list[str],
    num_classes: int,
) -> dict:
    """Inject the data-shape params a DL classifier needs, keeping them out of the YAML."""
    out = dict(params)
    out["num_classes"] = num_classes
    if name == "numerical":
        out["in_features"] = len(num_cols)
    elif name == "categorical":
        out["num_features"] = len(cat_cols)
    elif name == "tabular":
        out["num_numerical_features"] = len(num_cols)
        out["num_categorical_features"] = len(cat_cols)
    return out


def _resolve_fit_params(
    cfg, kind: str, num_cols: list[str], cat_cols: list[str], df_meta: dict
) -> dict:
    """Resolve the classifier `params` (DL shape injection / ML random_state)."""
    params = (
        OmegaConf.to_container(cfg.classifier.params, resolve=True)
        if cfg.classifier.params is not None
        else {}
    )
    if kind == "dl":
        params = _resolve_dl_params(
            cfg.classifier.name, params, num_cols, cat_cols, df_meta["num_classes"]
        )
    elif _supports_random_state(MLClassifierFactory.get(cfg.classifier.name)):
        params.setdefault("random_state", cfg.seed)
    return params


def _fit_model(
    training_mod: ModuleType,
    name: str,
    params: dict,
    X,
    y,
    X_val,
    y_val,
    context: dict,
    save_dir: Path,
) -> tuple[object, dict]:
    """Fit one classifier on (X, y) and save it under `save_dir`."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model, summary = training_mod.fit_classifier(
        name=name, params=params, X=X, y=y, X_val=X_val, y_val=y_val, context=context
    )
    training_mod.save_model(model, save_dir, name=name, params=params)
    return model, summary


def _predict_model(
    training_mod: ModuleType,
    model_dir: Path,
    df: pd.DataFrame,
    feat_cols: list[str],
    kind: str,
    context: dict,
    return_embedding: bool = False,
) -> tuple:
    """Load the model in `model_dir` and predict `df` → (y_pred, y_proba[, embedding])."""
    model = training_mod.load_model(model_dir, context=context)
    X = df[feat_cols] if kind == "ml" else df
    return training_mod.predict_with_proba(
        model, X, context=context, return_embedding=return_embedding
    )


@timed
def _fit_splits(
    cfg,
    paths: OutputPaths,
    splits: list[_Split],
    val_df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    df_meta: dict,
    num_cols: list[str],
    cat_cols: list[str],
    bus: LogDispatcher,
) -> None:
    """Fit one model per split and save it under its own directory.

    Grid search and training-curve figures only apply to a single split: under k-fold
    they would run/publish once per fold, which today's k-fold path never did.
    """
    kind = cfg.classifier.kind
    training_mod = _resolve_training_module(kind)
    context = _build_context(cfg, paths, df_meta, num_cols, cat_cols, label_col)
    params = _resolve_fit_params(cfg, kind, num_cols, cat_cols, df_meta)
    X_val, y_val = _prepare_train_payload(kind, val_df, feat_cols, label_col)

    is_kfold = len(splits) > 1
    has_grid = "grid" in cfg.classifier and len(cfg.classifier.grid) > 0

    for split in splits:
        X, y = _prepare_train_payload(kind, split.train_df, feat_cols, label_col)
        fold_ctx = (
            {**context, "models_path": split.fold_dir} if kind == "dl" else context
        )

        if has_grid and not is_kfold:
            if kind != "ml":
                raise NotImplementedError(
                    f"Grid search is only implemented for ML classifiers; got kind={kind!r}."
                )
            logger.info(
                "Grid search for %s — scoring=%s, cv=%d",
                cfg.classifier.name,
                cfg.grid_search.scoring,
                cfg.grid_search.cv,
            )
            model, summary = training_mod.grid_search_classifier(
                name=cfg.classifier.name,
                params=params,
                grid=dict(cfg.classifier.grid),
                X=X,
                y=y,
                scoring=cfg.grid_search.scoring,
                cv=cfg.grid_search.cv,
                context=fold_ctx,
                max_samples=cfg.grid_search.max_samples,
                random_state=cfg.seed,
            )
            logger.info(
                "Best params: %s | Best score (%s): %.4f",
                summary["best_params"],
                summary["scoring"],
                summary["best_score"],
            )
            bus.publish(LogBundle.from_dict({"json/training/grid_search": summary}))
            training_mod.save_model(
                model, split.fold_dir, name=cfg.classifier.name, params=params
            )
        else:
            logger.info("Training %s ...", cfg.classifier.name)
            _, fit_summary = _fit_model(
                training_mod,
                cfg.classifier.name,
                params,
                X,
                y,
                X_val,
                y_val,
                fold_ctx,
                split.fold_dir,
            )
            history = fit_summary.get("history", {})
            if history and not is_kfold:
                bus.publish(LogBundle.from_dict(_training_history_figures(history)))

    if is_kfold:
        logger.info(
            "k-fold OOF: trained %d fold models under %s", len(splits), paths.models
        )
    else:
        logger.info("Model saved under %s", paths.models)
