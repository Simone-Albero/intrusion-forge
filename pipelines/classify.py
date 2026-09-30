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
from src.domain.data.digest import digest_frames
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
from src.domain.training.weighting import compute_class_weights
from src.engine.ml.model import MLClassifierFactory
from src.engine.ml.preprocessing import supports_random_state

setup_logger()
apply_plot_style()
logger = logging.getLogger(__name__)


@dataclass
class ClassifyContext:
    """What building, training and evaluating the classifier share."""

    cfg: object
    paths: OutputPaths
    trainer: Trainer
    label_col: str
    df_meta: dict
    data_digest: str
    routed_digest: str
    bus: LogDispatcher


def _balance_train(cfg, train_df: pd.DataFrame, *, label_col: str) -> pd.DataFrame:
    """Apply this run's `fit.balance` and `fit.n_samples` cap to the training split."""
    if cfg.fit.balance == "undersample":
        train_df = random_undersample_df(train_df, label_col, random_state=cfg.seed)
    if cfg.fit.n_samples is not None:
        train_df = subsample_df(
            train_df, cfg.fit.n_samples, random_state=cfg.seed, label_col=label_col
        )
    return train_df


def _component(node) -> ComponentSpec:
    """Resolve a `{name, params}` config node into plain Python values."""
    params = to_container(node.params) if node.params is not None else {}
    return ComponentSpec(name=node.name, params=params)


def build_trainer(
    cfg,
    *,
    df_meta: dict,
    train_df: pd.DataFrame,
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
    class_ids = sorted(c["class_id"] for c in df_meta["classes"])
    weight_by_class = compute_class_weights(train_df[label_col])
    return DLTrainer(
        device=torch.device(fit_cfg.device),
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=label_col,
        # The weights correct the original distribution; `balance` and `n_samples` both
        # flatten it already, and weighting on top would correct the imbalance twice.
        class_weights=(
            [weight_by_class[cid] for cid in class_ids]
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
        seed=cfg.seed,
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


# Config the model cannot depend on: this stage's switches and outputs, and the groups
# only other stages read. Every other key is in, so a key added later retrains rather
# than reuses wrongly.
_NOT_TRAINING = (
    "force",
    "figure_format",
    "path",
    "name",
    "distance",
    "clustering",
    "complexity",
    "failure_regressor",
)
_INERT_LOADER_KEYS = ("num_workers", "pin_memory")


def _fingerprint(cfg, *, data_digest: str) -> dict:
    """The config the model is trained under, plus the digest of the data it sees."""
    config = to_container(cfg)
    for key in _NOT_TRAINING:
        del config[key]
    # Where and how fast a model trains, not on what: reusing a model trained on another
    # device or with other parallelism is the point, even where float rounding differs.
    del config["fit"]["device"]
    if config["classifier"]["params"] is not None:
        config["classifier"]["params"].pop("n_jobs", None)
    for loop in ("training", "validation"):
        for key in _INERT_LOADER_KEYS:
            del config["fit"][loop]["dataloader"][key]
    # Bumped when the code changes what a config trains: older records then never match.
    return {"schema": 3, **config, "data_digest": data_digest}


def _training_record_path(paths: OutputPaths) -> Path:
    return paths.outputs / "training/record.json"


def _can_reuse(context: ClassifyContext, fingerprint: dict) -> bool:
    """True when a model already exists trained under this exact configuration."""
    if context.cfg.force:
        return False
    model_dir = context.paths.models
    # The record first: the model itself depends on the config, so a changed config is
    # the cause and a missing model only its symptom.
    record_path = _training_record_path(context.paths)
    if not record_path.exists():
        if context.trainer.has_model(model_dir):
            logger.info("[RETRAIN] no record of what the model on disk was trained on.")
        return False
    changed = first_difference(load_from_json(record_path)["fingerprint"], fingerprint)
    if changed is not None:
        logger.info("[RETRAIN] training inputs changed (%s) — retraining.", changed)
        return False
    if not context.trainer.has_model(model_dir):
        logger.info("[RETRAIN] no model on disk — retraining.")
        return False

    logger.info("[CACHED] Reusing the trained model — pass force=true to retrain.")
    return True


def _fit_classifier(
    context: ClassifyContext,
    train_df: pd.DataFrame,
    *,
    params: dict,
    X_val,
) -> tuple[object, dict, list]:
    """Fit the classifier; return it with its grid-search summary and candidate rows."""
    cfg, trainer, bus = context.cfg, context.trainer, context.bus
    model_dir = context.paths.models
    X, y = trainer.prepare(train_df, context.label_col)

    if "grid" in cfg.classifier and len(cfg.classifier.grid) > 0:
        cv = cfg.grid_search.cv
        logger.info(
            "Grid search for %s — scoring=%s, cv=%d",
            cfg.classifier.name,
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
        # Flat: the grid's parameter names are the same for every candidate, and the
        # `param_` prefix keeps them from colliding with the score columns.
        summary_row = {
            **{f"param_{k}": v for k, v in summary["best_params"].items()},
            "best_score": summary["best_score"],
        }
        grid_rows = [
            {
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
                        f"figure/training/{key}": plot
                        for key, plot in training_history_figures(history).items()
                    }
                )
            )
        summary_row, grid_rows = {}, []

    trainer.save(model, model_dir, name=cfg.classifier.name, params=params)
    return model, summary_row, grid_rows


def _publish_training_record(
    context: ClassifyContext,
    *,
    fingerprint: dict,
    row: dict,
    grid_rows: list,
) -> None:
    """Publish the one record of what was trained: run scalars and the grid table."""
    cfg = context.cfg
    logger.info("Trained 1 model under %s", context.paths.models)
    context.bus.publish(
        LogBundle.from_dict(
            {
                "json/training/record": {
                    "seed": cfg.seed,
                    "balance": cfg.fit.balance,
                    "n_samples": cfg.fit.n_samples,
                    "scoring": cfg.grid_search.scoring if grid_rows else None,
                    "cv": cfg.grid_search.cv if grid_rows else None,
                    **row,
                    "fingerprint": fingerprint,
                    "grid_search": grid_rows,
                }
            }
        )
    )


@timed
def train_model(
    context: ClassifyContext,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    val_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Fit or reuse the classifier and predict every test row."""
    cfg, trainer = context.cfg, context.trainer
    params = _resolve_classifier_params(
        cfg,
        num_cols=trainer.num_cols,
        cat_cols=trainer.cat_cols,
        df_meta=context.df_meta,
    )
    fingerprint = _fingerprint(cfg, data_digest=context.data_digest)
    reuse = _can_reuse(context, fingerprint)
    if not reuse:
        # Dropped first: a crash between saving the new model and publishing its record
        # would otherwise leave the old record pointing at a model it never described.
        _training_record_path(context.paths).unlink(missing_ok=True)
    X_val = None if reuse else trainer.features(val_df)

    if reuse:
        model = trainer.load(context.paths.models)
    else:
        model, grid_extra, grid_rows = _fit_classifier(
            context, train_df, params=params, X_val=X_val
        )
        row = {"n_train": len(train_df), "n_eval": len(test_df), **grid_extra}
        _publish_training_record(
            context, fingerprint=fingerprint, row=row, grid_rows=grid_rows
        )

    y_pred, y_proba, embedding = trainer.predict(
        model, trainer.features(test_df), return_embedding=True
    )
    return y_pred, y_proba, embedding


@timed
def publish_evaluation(
    context: ClassifyContext,
    test_df: pd.DataFrame,
    *,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    embedding: np.ndarray | None,
) -> None:
    """Turn the test predictions into metrics, figures and per-sample dumps."""
    label_col, df_meta = context.label_col, context.df_meta
    class_names = {c["class_id"]: c["class_name"] for c in df_meta["classes"]}

    y_true = test_df[label_col].to_numpy()
    clusters = test_df["routed_cluster"].to_numpy()

    # Every class, not only the observed ones: a prediction into a class the test rows
    # never contain stays visible, and row k is class id k.
    all_classes = np.arange(df_meta["n_classes"])
    cm = confusion_matrix(y_true, y_pred, labels=all_classes, normalize="true")
    mcp = mcp_risk(y_proba)

    full_metrics = compute_classification_metrics(y_true, y_pred)
    pred_infos = {
        **evaluate_predictions(y_true, y_pred, mcp, clusters),
        # What these rates were measured on: the regressor checks both against the data
        # prepare holds when it runs.
        "data_digest": context.data_digest,
        "routed_digest": context.routed_digest,
    }
    raw_figures = {
        **build_test_figures(
            test_df,
            context.trainer.num_cols + context.trainer.cat_cols,
            y_true=y_true,
            y_pred=y_pred,
            cm=cm,
            cm_classes=all_classes,
            class_names=class_names,
        ),
        **latent_figures(
            embedding, y_true=y_true, y_pred=y_pred, class_names=class_names
        ),
    }
    figures = {f"figure/testing/{name}": plot for name, plot in raw_figures.items()}
    save_df(
        per_sample_scores(y_true, y_pred, mcp, clusters),
        context.paths.outputs / "analysis/predictions/eval_samples.parquet",
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


@timed
def classify(cfg) -> None:
    """Run the supervised classification pipeline for a single classifier."""
    if cfg.fit.balance not in ("undersample", "none"):
        raise ValueError(
            f"Unknown balance: {cfg.fit.balance!r}. Valid: 'undersample', 'none'."
        )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    set_figure_format(cfg.figure_format)
    paths = paths_from_cfg(cfg)

    # prepare writes its record last: without it, the splits and df_meta may disagree.
    prepared_path = paths.shared / "prepare_fingerprint.json"
    if not prepared_path.exists():
        raise FileNotFoundError(f"Missing {prepared_path}: run `make prepare` first.")
    # The splits and regions on disk were sized for this: a changed train/test fraction
    # (or any other data.* key) would otherwise be evaluated with no error at all.
    changed = first_difference(
        load_from_json(prepared_path)["data"], to_container(cfg.data)
    )
    if changed is not None:
        raise ValueError(
            f"data.{changed} differs from what `make prepare` last ran with: re-run "
            "`make prepare` to size the splits and regions for the current config."
        )
    df_meta = load_prepared_metadata(paths.shared / "metadata/df_meta.json")

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = "encoded_" + cfg.data.label_col

    train_df, val_df, test_df = (
        load_df(paths.processed_data / f"{split}.{cfg.data.extension}")
        for split in ("train", "val", "test")
    )
    for split, df in (("train", train_df), ("test", test_df)):
        if "routed_cluster" not in df.columns:
            raise ValueError(
                f"The {split} split has no `routed_cluster`: re-run `make prepare`."
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
            train_df=train_df,
            num_cols=num_cols,
            cat_cols=cat_cols,
            label_col=label_col,
        ),
        label_col=label_col,
        df_meta=df_meta,
        # The splits as loaded, feature and label columns only: `cluster` stays out, so
        # re-clustering a dataset reuses its trained model.
        data_digest=digest_frames(
            {"train": train_df, "val": val_df, "test": test_df},
            num_cols + cat_cols + [label_col],
        ),
        # The regions every rate here is counted on, which the label never chose.
        routed_digest=digest_frames(
            {"train": train_df, "val": val_df, "test": test_df}, ["routed_cluster"]
        ),
        bus=bus,
    )
    # Reassigned, not copied: the full, unbalanced train frame the trainer above already
    # used would otherwise stay in memory throughout, alongside its balanced copy.
    train_df = _balance_train(cfg, train_df, label_col=label_col)
    y_pred, y_proba, embedding = train_model(context, train_df, test_df, val_df=val_df)
    if not np.isfinite(y_proba).all():
        raise ValueError(
            "The model predicted non-finite probabilities: every confidence-based "
            "score downstream would be NaN."
        )
    publish_evaluation(
        context, test_df, y_pred=y_pred, y_proba=y_proba, embedding=embedding
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
