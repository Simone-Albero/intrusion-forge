import inspect
import logging
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from pipelines.classify_splits import _Split
from src.core.config import to_container
from src.core.log import LogBundle, LogDispatcher
from src.core.paths import OutputPaths
from src.core.utils import timed
from src.domain.plot.base import Plot
from src.domain.plot.primitives import line_plot
from src.domain.training.base import ComponentSpec, Trainer
from src.domain.training.dl import DLTrainer
from src.domain.training.ml import MLTrainer
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


def _training_history_figures(
    history: dict[str, list[float]], fold_prefix: str = ""
) -> dict[str, Plot]:
    """One line plot per scalar in the per-step DL training history."""
    return {
        f"figure/training/{fold_prefix}{name}_curve": line_plot(
            {name: values},
            y_label=name,
            show_legend=False,
        )
        for name, values in history.items()
        if values
    }


def _component(node) -> ComponentSpec:
    """Resolve a `{name, params}` config node into plain Python values."""
    params = to_container(node.params) if node.params is not None else {}
    return ComponentSpec(name=node.name, params=params)


def build_trainer(
    cfg,
    df_meta: dict,
    num_cols: list[str],
    cat_cols: list[str],
    label_col: str,
) -> Trainer:
    """Build the trainer matching the classifier kind, resolving cfg into plain values."""
    kind = cfg.classifier.kind
    if kind == "ml":
        return MLTrainer(num_cols, cat_cols)
    if kind != "dl":
        raise ValueError(f"Unknown classifier kind: {kind!r}. Expected 'ml' or 'dl'.")

    loops = cfg.loops
    return DLTrainer(
        device=torch.device(cfg.device),
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=label_col,
        class_weights=df_meta["class_weights"],
        loss=_component(cfg.loss),
        optimizer=_component(cfg.optimizer),
        scheduler=_component(cfg.scheduler),
        epochs=loops.training.epochs,
        max_grad_norm=loops.training.max_grad_norm,
        patience=loops.training.early_stopping.patience,
        min_delta=loops.training.early_stopping.min_delta,
        train_loader_params=to_container(loops.training.dataloader),
        val_loader_params=to_container(loops.validation.dataloader),
    )


def _resolve_dl_params(
    params: dict,
    num_cols: list[str],
    cat_cols: list[str],
    num_classes: int,
    cardinality: int,
) -> dict:
    """Inject the data-shape params the tabular DL classifier needs, keeping them out of the YAML."""
    out = dict(params)
    out["num_classes"] = num_classes
    out["num_numerical_features"] = len(num_cols)
    out["cardinalities"] = [cardinality] * len(cat_cols)
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
        cardinality = cfg.data.top_n + cfg.data.hash_buckets
        params = _resolve_dl_params(
            params, num_cols, cat_cols, df_meta["num_classes"], cardinality
        )
    elif _supports_random_state(MLClassifierFactory.get(cfg.classifier.name)):
        params.setdefault("random_state", cfg.seed)
    return params


def _fit_model(
    trainer: Trainer,
    name: str,
    params: dict,
    X,
    y,
    X_val,
    save_dir: Path,
) -> tuple[object, dict]:
    """Fit one classifier on (X, y) and save it under `save_dir`."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model, summary = trainer.fit(name, params, X, y, X_val=X_val, save_dir=save_dir)
    trainer.save(model, save_dir, name=name, params=params)
    return model, summary


def predict_split(
    trainer: Trainer,
    model_dir: Path,
    df: pd.DataFrame,
    feat_cols: list[str],
    *,
    return_embedding: bool = False,
) -> tuple:
    """Load the model in `model_dir` and predict `df` → (y_pred, y_proba[, embedding])."""
    model = trainer.load(model_dir)
    return trainer.predict(
        model, trainer.features(df, feat_cols), return_embedding=return_embedding
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

    Grid search and training-curve figures are published for every split (nested CV
    under k-fold), under a fold-scoped key when there is more than one split.
    """
    trainer = build_trainer(cfg, df_meta, num_cols, cat_cols, label_col)
    params = _resolve_fit_params(cfg, cfg.classifier.kind, num_cols, cat_cols, df_meta)
    X_val = trainer.features(val_df, feat_cols)

    is_kfold = len(splits) > 1
    has_grid = "grid" in cfg.classifier and len(cfg.classifier.grid) > 0
    fold_records = []

    for f, split in enumerate(splits):
        X, y = trainer.prepare(split.train_df, feat_cols, label_col)
        fold_record = {
            "fold": f,
            "n_train": len(split.train_df),
            "n_eval": len(split.eval_idx),
        }
        fold_records.append(fold_record)

        if has_grid:
            cv = cfg.grid_search.nested_cv if is_kfold else cfg.grid_search.cv
            logger.info(
                "Grid search for %s%s — scoring=%s, cv=%d",
                cfg.classifier.name,
                f" (fold {f + 1}/{len(splits)})" if is_kfold else "",
                cfg.grid_search.scoring,
                cv,
            )
            model, summary = trainer.grid_search(
                cfg.classifier.name,
                params,
                dict(cfg.classifier.grid),
                X,
                y,
                scoring=cfg.grid_search.scoring,
                cv=cv,
                max_samples=cfg.grid_search.max_samples,
                random_state=cfg.seed,
            )
            logger.info(
                "Best params: %s | Best score (%s): %.4f",
                summary["best_params"],
                summary["scoring"],
                summary["best_score"],
            )
            fold_record["best_params"] = summary["best_params"]
            fold_record["best_score"] = summary["best_score"]
            bus.publish(
                LogBundle.from_dict(
                    {f"json/training/{split.fold_prefix}grid_search": summary}
                )
            )
            trainer.save(model, split.fold_dir, name=cfg.classifier.name, params=params)
        else:
            logger.info("Training %s ...", cfg.classifier.name)
            _, fit_summary = _fit_model(
                trainer,
                cfg.classifier.name,
                params,
                X,
                y,
                X_val,
                split.fold_dir,
            )
            history = fit_summary.get("history", {})
            if history:
                bus.publish(
                    LogBundle.from_dict(
                        _training_history_figures(history, split.fold_prefix)
                    )
                )

    if is_kfold:
        logger.info(
            "k-fold OOF: trained %d fold models under %s", len(splits), paths.models
        )
        bus.publish(
            LogBundle.from_dict(
                {
                    "json/training/kfold_summary": {
                        "k_requested": cfg.kfold_splits,
                        "k_effective": len(splits),
                        "seed": cfg.seed,
                        "balance": cfg.balance,
                        "n_samples": cfg.n_samples,
                        "folds": fold_records,
                    }
                }
            )
        )
    else:
        logger.info("Model saved under %s", paths.models)
