import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix

from src.core.config import to_container
from src.core.io import save_df, save_figures
from src.core.log import setup_logger
from src.core.record import RECORD, clear_dir, is_current, write_record
from src.core.utils import flush_timing, load_from_json, save_to_json, timed
from src.domain.analysis.classification import compute_classification_metrics
from src.domain.analysis.confidence import mcp_risk
from src.domain.data.preprocessing import random_undersample_df, subsample_df
from src.domain.plot.base import Plot, set_figure_format
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
from stages import (
    SPLITS,
    load_cli_config,
    load_split,
    paths_from_cfg,
    stage_config,
    upstream_ids,
)

setup_logger()
apply_plot_style()
logger = logging.getLogger(__name__)

# Bumped when the code changes what a config trains: older records never match.
SCHEMA = 1
# What a reused model keeps: the model, how it was trained, and the figures of that
# training; its record stays too, since a crash while predicting does not make it stale.
MODEL_FILES = ("model", "training.json", "figures", RECORD)
LABEL = "label"


def _balance_train(cfg, train_df: pd.DataFrame) -> pd.DataFrame:
    """Apply this run's `fit.balance` and `fit.n_samples` cap to the training split."""
    if cfg.fit.balance == "undersample":
        train_df = random_undersample_df(train_df, LABEL, random_state=cfg.seed)
    if cfg.fit.n_samples is not None:
        train_df = subsample_df(
            train_df, cfg.fit.n_samples, random_state=cfg.seed, label_col=LABEL
        )
    return train_df


def _component(node) -> ComponentSpec:
    """Resolve a `{name, params}` config node into plain Python values."""
    params = to_container(node.params) if node.params is not None else {}
    return ComponentSpec(name=node.name, params=params)


def build_trainer(
    cfg,
    *,
    meta: dict,
    train_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
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
    class_ids = sorted(c["class_id"] for c in meta["classes"])
    weight_by_class = compute_class_weights(train_df[LABEL])
    return DLTrainer(
        device=torch.device(fit_cfg.device),
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=LABEL,
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
    cfg, *, num_cols: list[str], cat_cols: list[str], meta: dict
) -> dict:
    """Resolve the classifier `params` (DL shape injection / ML random_state)."""
    params = (
        OmegaConf.to_container(cfg.classifier.params, resolve=True)
        if cfg.classifier.params is not None
        else {}
    )
    if cfg.classifier.kind == "dl":
        # The data shape, derived here rather than written in the YAML.
        params["num_classes"] = meta["n_classes"]
        params["num_numerical_features"] = len(num_cols)
        params["cardinalities"] = [cfg.data.top_n + cfg.data.hash_buckets] * len(
            cat_cols
        )
    elif supports_random_state(MLClassifierFactory.get(cfg.classifier.name)):
        params.setdefault("random_state", cfg.seed)
    return params


@timed
def _fit_classifier(
    cfg,
    trainer: Trainer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    params: dict,
    model_dir: Path,
) -> tuple[object, dict, dict[str, Plot]]:
    """Fit and save the classifier; return it with its training record and figures."""
    X, y = trainer.prepare(train_df, LABEL)
    figures: dict[str, Plot] = {}

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
        best = {
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
        search = {"scoring": cfg.grid_search.scoring, "cv": cfg.grid_search.cv}
    else:
        logger.info("Training %s ...", cfg.classifier.name)
        model, summary = trainer.fit(
            cfg.classifier.name,
            params,
            X,
            y,
            X_val=trainer.features(val_df),
            save_dir=model_dir,
        )
        figures = {
            f"training_{key}": plot
            for key, plot in training_history_figures(
                summary.get("history", {})
            ).items()
        }
        best, grid_rows, search = {}, [], {"scoring": None, "cv": None}

    trainer.save(model, model_dir, name=cfg.classifier.name, params=params)
    logger.info("Trained 1 model under %s", model_dir)
    training = {
        "seed": cfg.seed,
        "balance": cfg.fit.balance,
        "n_samples": cfg.fit.n_samples,
        **search,
        "n_train": len(train_df),
        **best,
        "grid_search": grid_rows,
    }
    return model, training, figures


def _predict(
    trainer: Trainer, model, df: pd.DataFrame, split: str, *, embed: bool = False
):
    """Predict every row of a split; the probabilities must be finite."""
    y_pred, y_proba, *embedding = trainer.predict(
        model, trainer.features(df), return_embedding=embed
    )
    if not np.isfinite(y_proba).all():
        raise ValueError(
            f"The model predicted non-finite probabilities on {split}: every "
            "confidence-based score downstream would be NaN."
        )
    return y_pred, y_proba, embedding[0] if embed else None


def _predictions_table(
    predicted: dict[str, tuple[np.ndarray, np.ndarray]], *, fit_rows: pd.Index
) -> pd.DataFrame:
    """One row per row of every split; `in_fit` marks the train rows the model was fitted on."""
    table = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": split,
                    "row": np.arange(len(y_pred), dtype=np.int32),
                    "y_pred": y_pred.astype(np.int32),
                    "mcp_risk": mcp_risk(y_proba).astype(np.float32),
                    "in_fit": (
                        np.isin(np.arange(len(y_pred)), fit_rows)
                        if split == "train"
                        else np.zeros(len(y_pred), dtype=bool)
                    ),
                }
            )
            for split, (y_pred, y_proba) in predicted.items()
        ],
        ignore_index=True,
    )
    table["split"] = pd.Categorical(table["split"], categories=list(predicted))
    return table


@timed
def write_evaluation(
    stage_dir: Path,
    test_df: pd.DataFrame,
    predicted: tuple[np.ndarray, np.ndarray, np.ndarray | None],
    *,
    meta: dict,
    extra_figures: dict[str, Plot],
) -> None:
    """Metrics and figures of the test predictions."""
    y_pred, _, embedding = predicted
    class_names = {c["class_id"]: c["class_name"] for c in meta["classes"]}
    y_true = test_df[LABEL].to_numpy()

    # Every class, not only the observed ones: a prediction into a class the test rows
    # never contain stays visible, and row k is class id k.
    all_classes = np.arange(meta["n_classes"])
    cm = confusion_matrix(y_true, y_pred, labels=all_classes, normalize="true")
    figures = {
        **build_test_figures(
            test_df,
            meta["num_cols"] + meta["cat_cols"],
            y_true=y_true,
            y_pred=y_pred,
            cm=cm,
            cm_classes=all_classes,
            class_names=class_names,
        ),
        **latent_figures(
            embedding, y_true=y_true, y_pred=y_pred, class_names=class_names
        ),
        **extra_figures,
    }
    save_figures(figures, stage_dir / "figures")
    save_to_json(
        compute_classification_metrics(y_true, y_pred), stage_dir / "metrics.json"
    )


def classify(cfg) -> None:
    """Train or reuse the classifier, predict every split, and write the evaluation."""
    if cfg.fit.balance not in ("undersample", "none"):
        raise ValueError(
            f"Unknown balance: {cfg.fit.balance!r}. Valid: 'undersample', 'none'."
        )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    set_figure_format(cfg.figure_format)

    paths = paths_from_cfg(cfg)
    stage_dir = paths.of("classify")
    model_dir = stage_dir / "model"
    config = stage_config(cfg, "classify")
    inputs = upstream_ids(cfg, paths, "classify")
    meta = load_from_json(paths.of("split") / "meta.json")
    num_cols, cat_cols = meta["num_cols"], meta["cat_cols"]
    train_df, val_df, test_df = (
        load_split(paths, name) for name in ("train", "val", "test")
    )
    logger.info(
        "Data loaded — train: %d, val: %d, test: %d samples",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info("Classifier: %s (kind=%s)", cfg.classifier.name, cfg.classifier.kind)

    trainer = build_trainer(
        cfg, meta=meta, train_df=train_df, num_cols=num_cols, cat_cols=cat_cols
    )
    reuse = trainer.has_model(model_dir) and is_current(
        stage_dir,
        ("training.json",),
        schema=SCHEMA,
        config=config,
        inputs=inputs,
        force=cfg.force,
    )
    # A new model clears the record with the rest: a crash between saving it and writing
    # its record would otherwise leave the old record vouching for a model it never saw.
    clear_dir(stage_dir, keep=MODEL_FILES if reuse else ())

    figures: dict[str, Plot] = {}
    if reuse:
        logger.info("[CACHED] Reusing the trained model — pass force=true to retrain.")
        model = trainer.load(model_dir)
        fit_rows = _balance_train(cfg, train_df).index
    else:
        # Reassigned, not copied: the full, unbalanced train frame would otherwise stay
        # in memory throughout, alongside its balanced copy.
        train_df = _balance_train(cfg, train_df)
        fit_rows = train_df.index
        params = _resolve_classifier_params(
            cfg, num_cols=num_cols, cat_cols=cat_cols, meta=meta
        )
        model, training, figures = _fit_classifier(
            cfg, trainer, train_df, val_df, params=params, model_dir=model_dir
        )
        save_to_json({**training, "n_eval": len(test_df)}, stage_dir / "training.json")
        # The balanced copy is gone from here on: every train row gets a prediction.
        train_df = load_split(paths, "train")

    predicted = {
        name: _predict(trainer, model, df, name, embed=name == "test")
        for name, df in zip(SPLITS, (train_df, val_df, test_df))
    }
    write_evaluation(
        stage_dir, test_df, predicted["test"], meta=meta, extra_figures=figures
    )
    save_df(
        _predictions_table(
            {name: p[:2] for name, p in predicted.items()}, fit_rows=fit_rows
        ),
        stage_dir / "predictions.parquet",
    )
    flush_timing(stage_dir / "timing.json")
    write_record(stage_dir, schema=SCHEMA, config=config, inputs=inputs)
    logger.info("All stages completed.")


def main() -> None:
    """Entry point for the classify stage."""
    classify(load_cli_config())


if __name__ == "__main__":
    main()
