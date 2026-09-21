import inspect
import logging
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from pipelines import paths_from_cfg
from src.core.config import load_config, save_config, to_container
from src.core.io import load_listed_dfs, save_df
from src.core.log import (
    FilesystemFigureSubscriber,
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    PickleSubscriber,
    setup_logger,
)
from src.core.paths import OutputPaths
from src.core.utils import flush_timing, load_from_json, timed
from src.domain.analysis.classification import (
    compute_classification_metrics,
    evaluate_predictions,
    per_sample_scores,
)
from src.domain.analysis.confidence import mcp_risk
from src.domain.data.preprocessing import random_undersample_df, subsample_df
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

setup_logger()
apply_plot_style()
logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    """Seed the random, numpy and torch generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class DataConfig:
    """Shared data parameters across stages."""

    processed_data_path: Path
    extension: str
    label_col: str
    n_samples: int | None
    balance: str = "undersample"


def _load_data(
    data: DataConfig, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits, balancing and subsampling the training set only."""
    train_df, val_df, test_df = load_listed_dfs(
        data.processed_data_path,
        [
            f"train.{data.extension}",
            f"val.{data.extension}",
            f"test.{data.extension}",
        ],
    )
    if data.balance == "undersample":
        train_df = random_undersample_df(
            train_df, data.label_col, random_state=random_state
        )
    if data.n_samples is not None:
        train_df = subsample_df(
            train_df,
            data.n_samples,
            random_state=random_state,
            label_col=data.label_col,
        )
    return train_df, val_df, test_df


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
        # The weights come from the original, imbalanced distribution, so they only apply
        # to a split that still has it. Both `balance` and `n_samples` flatten it —
        # subsample_df caps every class at the same size — and weighting on top of either
        # would correct the same imbalance twice, over-shooting towards the rare classes.
        class_weights=(
            df_meta["class_weights"]
            if cfg.balance == "none" and cfg.n_samples is None
            else None
        ),
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
    """Inject the data-shape params the DL classifier needs, keeping them out of the YAML."""
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
    from when the training split keeps its original distribution. `device` is left out on
    purpose — it does change the weights, but reusing a model trained on another device is
    the point, not an accident. Both dataloaders are fingerprinted wholesale via
    `_loader_fingerprint`, minus `num_workers`/`pin_memory` — nothing in the dataset is
    random, but every other key (`batch_size`, `shuffle`, `drop_last`, ...) can shift
    training or the early-stopping metric Ignite computes as an average of per-batch means,
    so enumerating fields by hand would leave the same hole open for the next key added.
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
            "train_loader": _loader_fingerprint(training.dataloader),
            "val_loader": _loader_fingerprint(cfg.loops.validation.dataloader),
        }
    return fingerprint


def _training_record_path(paths: OutputPaths) -> Path:
    """Where the record of the models currently on disk lives."""
    return paths.outputs / "training/folds.json"


def _invalidate_training_record(paths: OutputPaths) -> None:
    """Drop the training record before retraining.

    Models are overwritten one split at a time, so a run interrupted mid-loop leaves a
    mix of old and new models on disk. Without this, the surviving record would still
    describe the old ones and the next run would reuse that mix.
    """
    _training_record_path(paths).unlink(missing_ok=True)


def _first_difference(previous: dict, current: dict) -> str | None:
    """Name a field that differs between two fingerprints, or None when they match."""
    for key in sorted(set(previous) | set(current)):
        if previous.get(key) != current.get(key):
            return key
    return None


def _can_reuse(context: ClassifyContext, plan: SplitPlan, fingerprint: dict) -> bool:
    """True when every split already has a model trained for this exact configuration."""
    trainer = context.trainer
    if context.cfg.force:
        return False

    previous_path = _training_record_path(context.paths)
    if not previous_path.exists():
        return False

    previous = load_from_json(previous_path).get("fingerprint", {})
    changed = _first_difference(previous, fingerprint)
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


def _grid_cv(cfg, plan: SplitPlan) -> int:
    """Inner CV of the grid search: smaller under k-fold, which already resamples."""
    return cfg.grid_search.nested_cv if plan.is_kfold else cfg.grid_search.cv


def _train_split(
    context: ClassifyContext,
    plan: SplitPlan,
    split: Split,
    fold: int,
    params: dict,
    X_val,
) -> tuple[object, dict, list]:
    """Fit one split's model, returning it with its fold record and grid-search rows."""
    cfg, trainer, bus = context.cfg, context.trainer, context.bus
    record = {
        "fold": fold,
        "n_train": len(split.train_df),
        "n_eval": len(split.eval_idx),
    }
    X, y = trainer.prepare(split.train_df, context.feat_cols, context.label_col)

    if "grid" in cfg.classifier and len(cfg.classifier.grid) > 0:
        cv = _grid_cv(cfg, plan)
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
        # Flat rows: the grid's parameter names are the same for every combination in a
        # run, and the `param_` prefix keeps them from colliding with the score columns.
        grid_rows = [
            {
                "fold": fold,
                **{f"param_{k}": v for k, v in combination["params"].items()},
                "mean_test_score": combination["mean_test_score"],
                "std_test_score": combination["std_test_score"],
            }
            for combination in summary["cv_results"]
        ]
        trainer.save(model, split.fold_dir, name=cfg.classifier.name, params=params)
        return model, record, grid_rows

    logger.info("Training %s ...", cfg.classifier.name)
    split.fold_dir.mkdir(parents=True, exist_ok=True)
    model, summary = trainer.fit(
        cfg.classifier.name, params, X, y, X_val=X_val, save_dir=split.fold_dir
    )
    trainer.save(model, split.fold_dir, name=cfg.classifier.name, params=params)
    history = summary.get("history", {})
    if history:
        bus.publish(
            LogBundle.from_dict(
                {
                    f"figure/training/{split.fold_prefix}{key}": plot
                    for key, plot in training_history_figures(history).items()
                }
            )
        )
    return model, record, []


def _publish_training_record(
    context: ClassifyContext,
    plan: SplitPlan,
    fingerprint: dict,
    fold_records: list,
    grid_rows: list,
) -> None:
    """Publish the one record of what was trained: run scalars plus two tables.

    Identical in shape whether or not k-fold ran, so nothing has to know which mode
    produced it to read it.
    """
    cfg = context.cfg
    logger.info("Trained %d model(s) under %s", len(plan.splits), context.paths.models)
    context.bus.publish(
        LogBundle.from_dict(
            {
                "json/training/folds": {
                    "mode": plan.mode,
                    "k_requested": cfg.kfold_splits if plan.is_kfold else 1,
                    "k_effective": len(plan.splits),
                    "seed": cfg.seed,
                    "balance": cfg.balance,
                    "n_samples": cfg.n_samples,
                    "scoring": cfg.grid_search.scoring if grid_rows else None,
                    "cv": _grid_cv(cfg, plan) if grid_rows else None,
                    "fingerprint": fingerprint,
                    "folds": fold_records,
                    "grid_search": grid_rows,
                }
            }
        )
    )


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
        _invalidate_training_record(context.paths)
    X_val = None if reuse else trainer.features(val_df, context.feat_cols)

    universe = plan.universe
    y_pred = np.empty(len(universe), dtype=universe[context.label_col].to_numpy().dtype)
    y_proba = np.zeros((len(universe), context.df_meta["num_classes"]))
    covered = np.zeros(len(universe), dtype=bool)
    embeddings: list[np.ndarray | None] = []
    fold_records: list[dict] = []
    grid_rows: list[dict] = []

    for fold, split in enumerate(plan.splits):
        if reuse:
            model = trainer.load(split.fold_dir)
        else:
            model, record, fold_grid = _train_split(
                context, plan, split, fold, params, X_val
            )
            fold_records.append(record)
            grid_rows.extend(fold_grid)

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
        _publish_training_record(context, plan, fingerprint, fold_records, grid_rows)

    return SplitPredictions(y_pred, y_proba, covered, embeddings)


@timed
def publish_evaluation(
    context: ClassifyContext, plan: SplitPlan, predictions: SplitPredictions
) -> None:
    """Turn the merged out-of-fold predictions into metrics, figures and per-sample dumps.

    A single split's `eval_idx` covers only the test rows, so the merged evaluation is
    test-only; k-fold's `eval_idx` values partition the whole universe, so it is not.
    """
    label_col, df_meta = context.label_col, context.df_meta
    label_mapping = df_meta["label_mapping"]

    eval_pos = np.flatnonzero(predictions.covered)
    eval_universe = plan.universe.iloc[eval_pos]
    y_true = eval_universe[label_col].to_numpy()
    y_pred = predictions.y_pred[eval_pos]
    y_proba = predictions.y_proba[eval_pos]
    clusters = (
        eval_universe["cluster"].to_numpy()
        if "cluster" in eval_universe.columns
        else None
    )

    # np.unique(y_true), not unique_labels(y_true, y_pred): compute_classification_metrics
    # uses the latter internally (it must, to include classes only ever predicted), so this
    # stays under its own name to keep the two orderings from being reached for interchangeably.
    observed_classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=observed_classes, normalize="true")
    mcp = mcp_risk(y_proba)

    full_metrics = {
        **compute_classification_metrics(y_true, y_pred),
        "eval_mode": plan.mode,
    }
    pred_infos = {
        **evaluate_predictions(y_true, y_pred, mcp, clusters),
        "eval_mode": plan.mode,
    }
    raw_figures = {
        **build_test_figures(
            eval_universe,
            context.feat_cols,
            y_true=y_true,
            y_pred=y_pred,
            cm=cm,
            classes=observed_classes,
            label_mapping=label_mapping,
        ),
        **latent_figures(
            [
                (s.fold_prefix, s.eval_idx, e)
                for s, e in zip(plan.splits, predictions.embeddings)
            ],
            universe_labels=plan.universe[label_col].to_numpy(),
            universe_y_pred=predictions.y_pred,
            label_mapping=label_mapping,
        ),
    }
    figures = {f"figure/testing/{name}": plot for name, plot in raw_figures.items()}
    if clusters is not None:
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
                "pickle/analysis/confusion_matrices/testing": cm,
            }
        )
    )
    if plan.is_kfold:
        logger.info(
            "k-fold OOF evaluation: %d samples over %d folds",
            len(eval_pos),
            len(plan.splits),
        )


@timed
def classify(cfg) -> None:
    """Run the supervised classification pipeline for a single classifier."""
    if cfg.balance not in ("undersample", "none"):
        raise ValueError(
            f"Unknown balance: {cfg.balance!r}. Valid: 'undersample', 'none'."
        )

    _seed_everything(cfg.seed)
    set_figure_format(cfg.figure_format)
    paths = paths_from_cfg(cfg)

    df_meta_path = paths.shared / "metadata/df_meta.json"
    if not df_meta_path.exists():
        raise FileNotFoundError(f"Missing {df_meta_path}. Run `make prepare` first.")
    df_meta = load_from_json(df_meta_path)
    save_config(cfg, paths.configs / "config_composed.json")

    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    label_col = "encoded_" + cfg.data.label_col
    feat_cols = num_cols + cat_cols

    data = DataConfig(
        processed_data_path=paths.processed_data,
        extension=cfg.data.extension,
        label_col=label_col,
        n_samples=cfg.n_samples,
        balance=cfg.balance,
    )

    use_kfold = cfg.kfold
    load_cfg = replace(data, balance="none", n_samples=None) if use_kfold else data
    train_df, val_df, test_df = _load_data(load_cfg, cfg.seed)
    logger.info(
        "Data loaded — train: %d, val: %d, test: %d samples",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info("Classifier: %s (kind=%s)", cfg.classifier.name, cfg.classifier.kind)

    bus = LogDispatcher()
    bus.subscribe(JSONSubscriber(paths.outputs))
    bus.subscribe(PickleSubscriber(paths.pickle))
    bus.subscribe(FilesystemFigureSubscriber(paths.figures))

    context = ClassifyContext(
        cfg=cfg,
        paths=paths,
        trainer=build_trainer(cfg, df_meta, num_cols, cat_cols, label_col),
        feat_cols=feat_cols,
        label_col=label_col,
        df_meta=df_meta,
        bus=bus,
    )
    plan = build_splits(cfg, paths, train_df, test_df, label_col)
    predictions = train_splits(context, plan, val_df)
    publish_evaluation(context, plan, predictions)

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


if __name__ == "__main__":
    main()
