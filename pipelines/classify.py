import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix

from pipelines import load_prepared_metadata, paths_from_cfg
from src.core.config import load_config, save_config, to_container
from src.core.io import load_df, save_df
from src.core.log import (
    FilesystemFigureSubscriber,
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from src.core.paths import OutputPaths
from src.core.utils import first_difference, flush_timing, load_from_json, timed
from src.domain.analysis.classification import (
    compute_classification_metrics,
    evaluate_predictions,
    per_sample_scores,
)
from src.domain.analysis.confidence import mcp_risk
from src.domain.data.preprocessing import (
    oof_splits,
    random_undersample_df,
    subsample_df,
)
from src.domain.plot.base import set_figure_format
from src.domain.plot.classify_charts import (
    build_test_figures,
    latent_figures,
    training_history_figures,
)
from src.domain.plot.style import apply_plot_style
from src.domain.training.base import ComponentSpec, Trainer
from src.domain.training.dl import DLTrainer
from src.domain.training.ml import MLTrainer
from src.engine.ml.model import MLClassifierFactory
from src.engine.ml.preprocessing import supports_random_state

setup_logger()
apply_plot_style()
logger = logging.getLogger(__name__)


@dataclass
class Fold:
    """One model: the rows it trains on and the evaluation rows it predicts."""

    name: str  # "" for the single split, "fold_<k>" under k-fold
    train_df: pd.DataFrame
    eval_idx: np.ndarray  # positions in the evaluation frame

    @property
    def artifact_prefix(self) -> str:
        """Prefix of the fold's figures, empty for the single split."""
        return f"{self.name}/" if self.name else ""


@dataclass
class ClassifyContext:
    """What both passes of the classify stage share."""

    cfg: object
    paths: OutputPaths
    trainer: Trainer
    label_col: str
    df_meta: dict
    bus: LogDispatcher


def _eval_mode(cfg) -> str:
    return "oof_kfold" if cfg.fit.kfold else "single_split"


def build_folds(
    cfg, train_df: pd.DataFrame, test_df: pd.DataFrame, *, label_col: str
) -> tuple[pd.DataFrame, list[Fold]]:
    """Eval frame (test, or train+test under k-fold) and the folds partitioning it."""

    def balanced(df: pd.DataFrame) -> pd.DataFrame:
        if cfg.fit.balance == "undersample":
            df = random_undersample_df(df, label_col, random_state=cfg.seed)
        if cfg.fit.n_samples is not None:
            df = subsample_df(
                df, cfg.fit.n_samples, random_state=cfg.seed, label_col=label_col
            )
        return df

    if not cfg.fit.kfold:
        return test_df, [Fold("", balanced(train_df), np.arange(len(test_df)))]

    universe = pd.concat([train_df, test_df], ignore_index=True)
    folds = [
        Fold(f"fold_{k}", balanced(universe.iloc[train_idx]), eval_idx)
        for k, (train_idx, eval_idx) in enumerate(
            oof_splits(universe, label_col, cfg.fit.kfold_splits, random_state=cfg.seed)
        )
    ]
    return universe, folds


def _component(node) -> ComponentSpec:
    """Resolve a `{name, params}` config node into plain Python values."""
    params = to_container(node.params) if node.params is not None else {}
    return ComponentSpec(name=node.name, params=params)


_INERT_LOADER_KEYS = ("num_workers", "pin_memory")


def _loader_fingerprint(node) -> dict:
    """A dataloader config, minus the keys that can't change what it produces."""
    return {k: v for k, v in to_container(node).items() if k not in _INERT_LOADER_KEYS}


def build_trainer(
    cfg,
    *,
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

    class_weight = cfg.loss.params.class_weight
    if not (class_weight in ("auto", None) or OmegaConf.is_list(class_weight)):
        raise ValueError(
            f"Unknown loss class_weight: {class_weight!r}. "
            "Expected 'auto', null or a list with one weight per class."
        )

    fit_cfg = cfg.fit
    return DLTrainer(
        device=torch.device(fit_cfg.device),
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=label_col,
        # The weights correct the original distribution; `balance` and `n_samples` both
        # flatten it already, and weighting on top would correct the imbalance twice.
        class_weights=(
            [
                c["weight"]
                for c in sorted(df_meta["classes"], key=lambda c: c["class_id"])
            ]
            if fit_cfg.balance == "none" and fit_cfg.n_samples is None
            else None
        ),
        loss=_component(cfg.loss),
        optimizer=_component(cfg.optimizer),
        scheduler=_component(cfg.scheduler),
        epochs=fit_cfg.training.epochs,
        max_grad_norm=fit_cfg.training.max_grad_norm,
        patience=fit_cfg.training.early_stopping.patience,
        min_delta=fit_cfg.training.early_stopping.min_delta,
        train_loader_params=to_container(fit_cfg.training.dataloader),
        val_loader_params=to_container(fit_cfg.validation.dataloader),
    )


def _resolve_classifier_params(
    cfg, *, num_cols: list[str], cat_cols: list[str], df_meta: dict
) -> dict:
    """Resolve the classifier `params` (DL shape injection / ML random_state)."""
    params = (
        OmegaConf.to_container(cfg.classifier.params, resolve=True)
        if cfg.classifier.params is not None
        else {}
    )
    if cfg.classifier.kind == "dl":
        # The data shape, derived here rather than written in the YAML.
        params["num_classes"] = df_meta["n_classes"]
        params["num_numerical_features"] = len(num_cols)
        params["cardinalities"] = [cfg.data.top_n + cfg.data.hash_buckets] * len(
            cat_cols
        )
    elif supports_random_state(MLClassifierFactory.get(cfg.classifier.name)):
        params.setdefault("random_state", cfg.seed)
    return params


def _fingerprint(
    cfg,
    *,
    params: dict,
    num_cols: list[str],
    cat_cols: list[str],
    label_col: str,
    df_meta: dict,
) -> dict:
    """Everything that determines the trained models, so a mismatch rules out reuse.

    `df_meta` stands in for the prepared data itself: its split sizes and per-class counts
    move whenever the data is regenerated or `prepare` is reconfigured, which the dataset
    name alone would not catch. It also carries the class weights that a
    `class_weight: auto` loss is built from when the training split keeps its original
    distribution. `device` is left out on purpose — it does change the weights, but
    reusing a model trained on another device is the point, not an accident. Both
    dataloaders are fingerprinted wholesale via `_loader_fingerprint`, minus
    `num_workers`/`pin_memory` — nothing in the dataset is random, but every other key
    (`batch_size`, `shuffle`, `drop_last`, ...) can shift training or the early-stopping
    metric Ignite computes as an average of per-batch means, so enumerating fields by
    hand would leave the same hole open for the next key added.
    """
    fingerprint = {
        "classifier": cfg.classifier.name,
        "kind": cfg.classifier.kind,
        "params": params,
        "grid": to_container(cfg.classifier.grid) if "grid" in cfg.classifier else None,
        "grid_search": to_container(cfg.grid_search),
        "seed": cfg.seed,
        "balance": cfg.fit.balance,
        "n_samples": cfg.fit.n_samples,
        "kfold": cfg.fit.kfold,
        "kfold_splits": cfg.fit.kfold_splits,
        "dataset": cfg.data.file_name,
        "extension": cfg.data.extension,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "label_col": label_col,
        "data_meta": df_meta,
    }
    if cfg.classifier.kind == "dl":
        training = cfg.fit.training
        fingerprint["dl_training"] = {
            # Bumped when a value in here changes meaning: older records never match.
            "schema": 2,
            "loss": to_container(cfg.loss),
            "optimizer": to_container(cfg.optimizer),
            "scheduler": to_container(cfg.scheduler),
            "epochs": training.epochs,
            "max_grad_norm": training.max_grad_norm,
            "early_stopping": to_container(training.early_stopping),
            "train_loader": _loader_fingerprint(training.dataloader),
            "val_loader": _loader_fingerprint(cfg.fit.validation.dataloader),
        }
    return fingerprint


def _training_record_path(paths: OutputPaths) -> Path:
    return paths.outputs / "training/folds.json"


def _can_reuse(context: ClassifyContext, folds: list[Fold], fingerprint: dict) -> bool:
    """True when every fold already has a model trained for this exact configuration."""
    trainer = context.trainer
    if context.cfg.force:
        return False

    previous_path = _training_record_path(context.paths)
    if not previous_path.exists():
        return False

    previous = load_from_json(previous_path).get("fingerprint", {})
    changed = first_difference(previous, fingerprint)
    if changed is not None:
        logger.info("[RETRAIN] training config changed (%s) — retraining.", changed)
        return False

    missing = [f for f in folds if not trainer.has_model(context.paths.models / f.name)]
    if missing:
        logger.info(
            "[RETRAIN] %d of %d model(s) missing on disk — retraining all.",
            len(missing),
            len(folds),
        )
        return False

    logger.info(
        "[STAGE-SKIP] Reusing %d trained model(s) — pass force=true to retrain.",
        len(folds),
    )
    return True


def _grid_cv(cfg) -> int:
    """Inner CV of the grid search: smaller under k-fold, which already resamples."""
    return cfg.grid_search.nested_cv if cfg.fit.kfold else cfg.grid_search.cv


def _train_fold(
    context: ClassifyContext,
    fold: Fold,
    *,
    index: int,
    n_folds: int,
    params: dict,
    X_val,
) -> tuple[object, dict, list]:
    """Fit one fold's model, returning it with its fold record and grid-search rows."""
    cfg, trainer, bus = context.cfg, context.trainer, context.bus
    model_dir = context.paths.models / fold.name
    record = {
        "fold": index,
        "n_train": len(fold.train_df),
        "n_eval": len(fold.eval_idx),
    }
    X, y = trainer.prepare(fold.train_df, context.label_col)

    if "grid" in cfg.classifier and len(cfg.classifier.grid) > 0:
        cv = _grid_cv(cfg)
        logger.info(
            "Grid search for %s%s — scoring=%s, cv=%d",
            cfg.classifier.name,
            f" (fold {index + 1}/{n_folds})" if cfg.fit.kfold else "",
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
        # Flat rows: the grid's parameter names are the same for every combination in a
        # run, and the `param_` prefix keeps them from colliding with the score columns.
        record.update({f"param_{k}": v for k, v in summary["best_params"].items()})
        record["best_score"] = summary["best_score"]
        grid_rows = [
            {
                "fold": index,
                **{f"param_{k}": v for k, v in combination["params"].items()},
                "mean_test_score": combination["mean_test_score"],
                "std_test_score": combination["std_test_score"],
            }
            for combination in summary["cv_results"]
        ]
    else:
        logger.info("Training %s ...", cfg.classifier.name)
        model, summary = trainer.fit(
            cfg.classifier.name, params, X, y, X_val=X_val, save_dir=model_dir
        )
        history = summary.get("history", {})
        if history:
            bus.publish(
                LogBundle.from_dict(
                    {
                        f"figure/training/{fold.artifact_prefix}{key}": plot
                        for key, plot in training_history_figures(history).items()
                    }
                )
            )
        grid_rows = []

    trainer.save(model, model_dir, name=cfg.classifier.name, params=params)
    return model, record, grid_rows


def _publish_training_record(
    context: ClassifyContext,
    *,
    folds: list[Fold],
    fingerprint: dict,
    fold_records: list,
    grid_rows: list,
) -> None:
    """Publish the one record of what was trained: run scalars plus two tables."""
    cfg = context.cfg
    logger.info("Trained %d model(s) under %s", len(folds), context.paths.models)
    context.bus.publish(
        LogBundle.from_dict(
            {
                "json/training/folds": {
                    "mode": _eval_mode(cfg),
                    "k_requested": cfg.fit.kfold_splits if cfg.fit.kfold else 1,
                    "k_effective": len(folds),
                    "seed": cfg.seed,
                    "balance": cfg.fit.balance,
                    "n_samples": cfg.fit.n_samples,
                    "scoring": cfg.grid_search.scoring if grid_rows else None,
                    "cv": _grid_cv(cfg) if grid_rows else None,
                    "fingerprint": fingerprint,
                    "folds": fold_records,
                    "grid_search": grid_rows,
                }
            }
        )
    )


@timed
def train_folds(
    context: ClassifyContext,
    eval_df: pd.DataFrame,
    folds: list[Fold],
    *,
    val_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray | None]]:
    """Predict every eval row with the model of the fold that holds it out."""
    cfg, trainer = context.cfg, context.trainer
    params = _resolve_classifier_params(
        cfg,
        num_cols=trainer.num_cols,
        cat_cols=trainer.cat_cols,
        df_meta=context.df_meta,
    )
    fingerprint = _fingerprint(
        cfg,
        params=params,
        num_cols=trainer.num_cols,
        cat_cols=trainer.cat_cols,
        label_col=context.label_col,
        df_meta=context.df_meta,
    )
    # All or nothing: reused models keep the training artifacts of their own run.
    reuse = _can_reuse(context, folds, fingerprint)
    if not reuse:
        # Dropped first: an interrupted retrain leaves old and new models mixed.
        _training_record_path(context.paths).unlink(missing_ok=True)
    X_val = None if reuse else trainer.features(val_df)

    y_pred = np.empty(len(eval_df), dtype=eval_df[context.label_col].to_numpy().dtype)
    y_proba = np.zeros((len(eval_df), context.df_meta["n_classes"]))
    embeddings: list[np.ndarray | None] = []
    fold_records: list[dict] = []
    grid_rows: list[dict] = []

    for index, fold in enumerate(folds):
        if reuse:
            model = trainer.load(context.paths.models / fold.name)
        else:
            model, record, fold_grid = _train_fold(
                context,
                fold,
                index=index,
                n_folds=len(folds),
                params=params,
                X_val=X_val,
            )
            fold_records.append(record)
            grid_rows.extend(fold_grid)

        fold_pred, fold_proba, embedding = trainer.predict(
            model,
            trainer.features(eval_df.iloc[fold.eval_idx]),
            return_embedding=True,
        )
        y_pred[fold.eval_idx] = fold_pred
        y_proba[fold.eval_idx] = fold_proba
        embeddings.append(embedding)

    if not reuse:
        _publish_training_record(
            context,
            folds=folds,
            fingerprint=fingerprint,
            fold_records=fold_records,
            grid_rows=grid_rows,
        )

    return y_pred, y_proba, embeddings


@timed
def publish_evaluation(
    context: ClassifyContext,
    eval_df: pd.DataFrame,
    folds: list[Fold],
    *,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    embeddings: list[np.ndarray | None],
) -> None:
    """Turn the merged predictions into metrics, figures and per-sample dumps."""
    label_col, df_meta = context.label_col, context.df_meta
    class_names = {c["class_id"]: c["class_name"] for c in df_meta["classes"]}
    mode = _eval_mode(context.cfg)

    y_true = eval_df[label_col].to_numpy()
    clusters = eval_df["cluster"].to_numpy()

    # Every class, not only the observed ones: a prediction into a class the evaluated
    # rows never contain stays visible, and row k is class id k.
    all_classes = np.arange(df_meta["n_classes"])
    cm = confusion_matrix(y_true, y_pred, labels=all_classes, normalize="true")
    mcp = mcp_risk(y_proba)

    full_metrics = {**compute_classification_metrics(y_true, y_pred), "eval_mode": mode}
    pred_infos = {
        **evaluate_predictions(y_true, y_pred, mcp, clusters),
        "eval_mode": mode,
    }
    raw_figures = {
        **build_test_figures(
            eval_df,
            context.trainer.num_cols + context.trainer.cat_cols,
            y_true=y_true,
            y_pred=y_pred,
            cm=cm,
            cm_classes=all_classes,
            class_names=class_names,
        ),
        **latent_figures(
            [
                (fold.artifact_prefix, fold.eval_idx, embedding)
                for fold, embedding in zip(folds, embeddings)
            ],
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
        ),
    }
    figures = {f"figure/testing/{name}": plot for name, plot in raw_figures.items()}
    save_df(
        per_sample_scores(y_true, y_pred, mcp, clusters),
        context.paths.outputs / "analysis/predictions/oof_samples.parquet",
    )

    context.bus.publish(
        LogBundle.from_dict(
            {
                **figures,
                "json/testing/summary": full_metrics,
                "json/analysis/predictions/clusters": pred_infos,
            }
        )
    )
    if context.cfg.fit.kfold:
        logger.info(
            "k-fold OOF evaluation: %d samples over %d folds", len(eval_df), len(folds)
        )


@timed
def classify(cfg) -> None:
    """Run the supervised classification pipeline for a single classifier."""
    if cfg.fit.balance not in ("undersample", "none"):
        raise ValueError(
            f"Unknown balance: {cfg.fit.balance!r}. Valid: 'undersample', 'none'."
        )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    set_figure_format(cfg.figure_format)
    paths = paths_from_cfg(cfg)

    df_meta_path = paths.shared / "metadata/df_meta.json"
    if not df_meta_path.exists():
        raise FileNotFoundError(f"Missing {df_meta_path}. Run `make prepare` first.")
    df_meta = load_prepared_metadata(df_meta_path)

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = "encoded_" + cfg.data.label_col

    train_df, val_df, test_df = (
        load_df(paths.processed_data / f"{split}.{cfg.data.extension}")
        for split in ("train", "val", "test")
    )
    for split, df in (("train", train_df), ("test", test_df)):
        if "cluster" not in df.columns:
            raise ValueError(
                f"The {split} split has no `cluster` column: "
                "re-run `make prepare FORCE=1`."
            )
    logger.info(
        "Data loaded — train: %d, val: %d, test: %d samples",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info("Classifier: %s (kind=%s)", cfg.classifier.name, cfg.classifier.kind)

    bus = LogDispatcher()
    bus.subscribe(JSONSubscriber(paths.outputs))
    bus.subscribe(FilesystemFigureSubscriber(paths.figures))

    context = ClassifyContext(
        cfg=cfg,
        paths=paths,
        trainer=build_trainer(
            cfg,
            df_meta=df_meta,
            num_cols=num_cols,
            cat_cols=cat_cols,
            label_col=label_col,
        ),
        label_col=label_col,
        df_meta=df_meta,
        bus=bus,
    )
    eval_df, folds = build_folds(cfg, train_df, test_df, label_col=label_col)
    # Only eval_df and the folds' balanced copies are needed from here: on a single
    # split the full, unbalanced train frame would otherwise stay in memory throughout.
    del train_df, test_df
    y_pred, y_proba, embeddings = train_folds(context, eval_df, folds, val_df=val_df)
    publish_evaluation(
        context, eval_df, folds, y_pred=y_pred, y_proba=y_proba, embeddings=embeddings
    )

    logger.info("All stages completed.")


def main() -> None:
    """Entry point for the supervised classification stage."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    classify(cfg)
    flush_timing(Path(cfg.path.outputs) / "timing.json")
    save_config(cfg, Path(cfg.path.configs) / "config_composed_classify.json")


if __name__ == "__main__":
    main()
