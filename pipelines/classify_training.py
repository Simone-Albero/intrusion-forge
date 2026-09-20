import inspect
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold

from src.core.config import to_container
from src.core.log import LogBundle, LogDispatcher
from src.core.paths import OutputPaths
from src.core.utils import load_from_json, timed
from src.domain.data.preprocessing import random_undersample_df, subsample_df
from src.domain.plot.base import Plot
from src.domain.plot.primitives import line_plot
from src.domain.training.base import ComponentSpec, Trainer
from src.domain.training.dl import DLTrainer
from src.domain.training.ml import MLTrainer
from src.engine.ml.model import MLClassifierFactory

logger = logging.getLogger(__name__)


@dataclass
class Split:
    """One split: the training rows and the universe positions it evaluates."""

    train_df: pd.DataFrame
    fold_dir: Path
    eval_idx: np.ndarray
    fold_prefix: str = ""


@dataclass
class SplitPlan:
    """The evaluation universe and the splits drawn from it."""

    universe: pd.DataFrame
    splits: list[Split]

    @property
    def is_kfold(self) -> bool:
        """True when more than one split covers the universe out of fold."""
        return len(self.splits) > 1

    @property
    def mode(self) -> str:
        """Evaluation mode recorded in the published artifacts."""
        return "oof_kfold" if self.is_kfold else "single_split"


@dataclass
class SplitPredictions:
    """Out-of-fold predictions over the universe, plus each split's latent embedding."""

    y_pred: np.ndarray
    y_proba: np.ndarray
    covered: np.ndarray
    embeddings: list[np.ndarray | None]


@dataclass
class ClassifyContext:
    """What both passes of the classify stage share."""

    cfg: object
    paths: OutputPaths
    trainer: Trainer
    feat_cols: list[str]
    label_col: str
    df_meta: dict
    bus: LogDispatcher


def _oof_splits(base: pd.DataFrame, label_col: str, k: int, seed: int) -> list:
    """Deterministic stratified OOF folds over `base`; K capped to the rarest class."""
    y = base[label_col].to_numpy()
    k = min(k, int(np.unique(y, return_counts=True)[1].min()))
    if k < 2:
        raise ValueError(f"k-fold OOF needs >=2 samples per class, got k={k}.")
    return list(
        StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(base, y)
    )


def build_splits(
    cfg,
    paths: OutputPaths,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
) -> SplitPlan:
    """Build the evaluation universe and its splits: the single test split, or k-fold OOF."""
    universe = pd.concat([train_df, test_df], ignore_index=True)

    if not cfg.kfold:
        eval_idx = np.arange(len(train_df), len(universe))
        return SplitPlan(universe, [Split(train_df, paths.models, eval_idx)])

    splits = []
    for f, (tr_idx, te_idx) in enumerate(
        _oof_splits(universe, label_col, cfg.kfold_splits, cfg.seed)
    ):
        fold_train = universe.iloc[tr_idx]
        if cfg.balance == "undersample":
            fold_train = random_undersample_df(
                fold_train, label_col, random_state=cfg.seed
            )
        if cfg.n_samples is not None:
            fold_train = subsample_df(
                fold_train, cfg.n_samples, random_state=cfg.seed, label_col=label_col
            )
        splits.append(
            Split(fold_train, paths.models / f"fold_{f}", te_idx, f"fold_{f}/")
        )
    return SplitPlan(universe, splits)


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


def _fingerprint(
    cfg,
    params: dict,
    num_cols: list[str],
    cat_cols: list[str],
    label_col: str,
    df_meta: dict,
) -> dict:
    """Everything that determines the trained models, so a mismatch rules out reuse.

    `df_meta` stands in for the prepared data itself: its split sizes and per-class counts
    move whenever the data is regenerated or `prepare` is reconfigured, which the dataset
    name alone would not catch. It also carries the `class_weights` the DL loss is built
    from. `device` is left out on purpose — it does change the weights, but reusing a model
    trained on another device is the point, not an accident. The dataloader's
    `num_workers`/`pin_memory` are left out because nothing in the dataset is random.
    """
    fingerprint = {
        "classifier": cfg.classifier.name,
        "kind": cfg.classifier.kind,
        "params": params,
        "grid": to_container(cfg.classifier.grid) if "grid" in cfg.classifier else None,
        "grid_search": to_container(cfg.grid_search),
        "seed": cfg.seed,
        "balance": cfg.balance,
        "n_samples": cfg.n_samples,
        "kfold": cfg.kfold,
        "kfold_splits": cfg.kfold_splits,
        "dataset": cfg.data.file_name,
        "extension": cfg.data.extension,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "label_col": label_col,
        "data_meta": df_meta,
    }
    if cfg.classifier.kind == "dl":
        training = cfg.loops.training
        fingerprint["dl_training"] = {
            "loss": to_container(cfg.loss),
            "optimizer": to_container(cfg.optimizer),
            "scheduler": to_container(cfg.scheduler),
            "epochs": training.epochs,
            "max_grad_norm": training.max_grad_norm,
            "early_stopping": to_container(training.early_stopping),
            "batch_size": training.dataloader.batch_size,
            "shuffle": training.dataloader.shuffle,
        }
    return fingerprint


def _first_difference(previous: dict, current: dict) -> str | None:
    """Name a field that differs between two fingerprints, or None when they match."""
    for key in sorted(set(previous) | set(current)):
        if previous.get(key) != current.get(key):
            return key
    return None


def _fingerprint_path(paths: OutputPaths) -> Path:
    """Where the fingerprint of the models currently on disk lives."""
    return paths.outputs / "training/fingerprint.json"


def _invalidate_fingerprint(paths: OutputPaths) -> None:
    """Drop the fingerprint before retraining.

    Models are overwritten one split at a time, so a run interrupted mid-loop leaves a
    mix of old and new models on disk. Without this, the surviving fingerprint would
    still describe the old ones and the next run would reuse that mix.
    """
    _fingerprint_path(paths).unlink(missing_ok=True)


def _can_reuse(context: ClassifyContext, plan: SplitPlan, fingerprint: dict) -> bool:
    """True when every split already has a model trained for this exact configuration."""
    trainer = context.trainer
    if context.cfg.force:
        return False

    previous_path = _fingerprint_path(context.paths)
    if not previous_path.exists():
        return False

    changed = _first_difference(load_from_json(previous_path), fingerprint)
    if changed is not None:
        logger.info("[RETRAIN] training config changed (%s) — retraining.", changed)
        return False

    missing = [s for s in plan.splits if not trainer.has_model(s.fold_dir)]
    if missing:
        logger.info(
            "[RETRAIN] %d of %d model(s) missing on disk — retraining all.",
            len(missing),
            len(plan.splits),
        )
        return False

    logger.info(
        "[STAGE-SKIP] Reusing %d trained model(s) — pass force=true to retrain.",
        len(plan.splits),
    )
    return True


def _train_split(
    context: ClassifyContext,
    plan: SplitPlan,
    split: Split,
    fold: int,
    params: dict,
    X_val,
) -> tuple[object, dict]:
    """Fit one split's model, publishing that split's training artifacts."""
    cfg, trainer, bus = context.cfg, context.trainer, context.bus
    record = {
        "fold": fold,
        "n_train": len(split.train_df),
        "n_eval": len(split.eval_idx),
    }
    X, y = trainer.prepare(split.train_df, context.feat_cols, context.label_col)

    if "grid" in cfg.classifier and len(cfg.classifier.grid) > 0:
        cv = cfg.grid_search.nested_cv if plan.is_kfold else cfg.grid_search.cv
        logger.info(
            "Grid search for %s%s — scoring=%s, cv=%d",
            cfg.classifier.name,
            f" (fold {fold + 1}/{len(plan.splits)})" if plan.is_kfold else "",
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
        record["best_params"] = summary["best_params"]
        record["best_score"] = summary["best_score"]
        bus.publish(
            LogBundle.from_dict(
                {f"json/training/{split.fold_prefix}grid_search": summary}
            )
        )
        trainer.save(model, split.fold_dir, name=cfg.classifier.name, params=params)
        return model, record

    logger.info("Training %s ...", cfg.classifier.name)
    split.fold_dir.mkdir(parents=True, exist_ok=True)
    model, summary = trainer.fit(
        cfg.classifier.name, params, X, y, X_val=X_val, save_dir=split.fold_dir
    )
    trainer.save(model, split.fold_dir, name=cfg.classifier.name, params=params)
    history = summary.get("history", {})
    if history:
        bus.publish(
            LogBundle.from_dict(_training_history_figures(history, split.fold_prefix))
        )
    return model, record


def _publish_training_summary(
    context: ClassifyContext, plan: SplitPlan, fingerprint: dict, fold_records: list
) -> None:
    """Publish the fingerprint identifying these models, plus the k-fold training record."""
    cfg = context.cfg
    artifacts = {"json/training/fingerprint": fingerprint}
    if plan.is_kfold:
        logger.info(
            "k-fold OOF: trained %d fold models under %s",
            len(plan.splits),
            context.paths.models,
        )
        artifacts["json/training/kfold_summary"] = {
            "k_requested": cfg.kfold_splits,
            "k_effective": len(plan.splits),
            "seed": cfg.seed,
            "balance": cfg.balance,
            "n_samples": cfg.n_samples,
            "folds": fold_records,
        }
    else:
        logger.info("Model saved under %s", context.paths.models)
    context.bus.publish(LogBundle.from_dict(artifacts))


@timed
def train_splits(
    context: ClassifyContext, plan: SplitPlan, val_df: pd.DataFrame
) -> SplitPredictions:
    """Obtain a model per split and predict the universe rows that split holds out.

    Each model is trained, or loaded when one already exists for this exact
    configuration, then used for prediction while still in memory and dropped: one model
    is held at a time. Reused models keep the training artifacts of the run that produced
    them, which describe them exactly.
    """
    cfg, trainer = context.cfg, context.trainer
    params = _resolve_fit_params(
        cfg, cfg.classifier.kind, trainer.num_cols, trainer.cat_cols, context.df_meta
    )
    fingerprint = _fingerprint(
        cfg,
        params,
        trainer.num_cols,
        trainer.cat_cols,
        context.label_col,
        context.df_meta,
    )
    reuse = _can_reuse(context, plan, fingerprint)
    if not reuse:
        _invalidate_fingerprint(context.paths)
    X_val = None if reuse else trainer.features(val_df, context.feat_cols)

    universe = plan.universe
    y_pred = np.empty(len(universe), dtype=universe[context.label_col].to_numpy().dtype)
    y_proba = np.zeros((len(universe), context.df_meta["num_classes"]))
    covered = np.zeros(len(universe), dtype=bool)
    embeddings: list[np.ndarray | None] = []
    fold_records = []

    for fold, split in enumerate(plan.splits):
        if reuse:
            model = trainer.load(split.fold_dir)
        else:
            model, record = _train_split(context, plan, split, fold, params, X_val)
            fold_records.append(record)

        eval_df = universe.iloc[split.eval_idx]
        fold_pred, fold_proba, embedding = trainer.predict(
            model,
            trainer.features(eval_df, context.feat_cols),
            return_embedding=True,
        )
        y_pred[split.eval_idx] = fold_pred
        y_proba[split.eval_idx] = fold_proba
        covered[split.eval_idx] = True
        embeddings.append(embedding)

    if not reuse:
        _publish_training_summary(context, plan, fingerprint, fold_records)

    return SplitPredictions(y_pred, y_proba, covered, embeddings)
