import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.core.config import to_container
from src.core.io import save_df
from src.core.log import setup_logger
from src.core.record import clear_dir, is_current, write_record
from src.core.utils import flush_timing, load_from_json, save_to_json, timed
from src.domain.analysis.classification import compute_classification_metrics
from src.domain.data.preprocessing import random_undersample_df, subsample_df
from src.domain.projection import TSNE_MAX_SAMPLES
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
logger = logging.getLogger(__name__)

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
) -> tuple[object, dict]:
    """Fit and save the classifier; return it with its training record."""
    X, y = trainer.prepare(train_df, LABEL)
    best, grid_rows, history = {}, [], []

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
        history = [
            {"step": step, "loss": loss}
            for step, loss in enumerate(summary.get("history", {}).get("loss", []))
        ]

    trainer.save(model, model_dir, name=cfg.classifier.name, params=params)
    logger.info("Trained 1 model under %s", model_dir)
    return model, {**best, "grid_search": grid_rows, "history": history}


def _predict(
    trainer: Trainer,
    model,
    df: pd.DataFrame,
    split: str,
    *,
    n_classes: int,
    embed: bool = False,
):
    """Predict every row of a split; one finite probability per class."""
    y_pred, y_proba, *embedding = trainer.predict(
        model, trainer.features(df), return_embedding=embed
    )
    if y_proba.shape[1] != n_classes:
        raise ValueError(
            f"The model gave {y_proba.shape[1]} probabilities per row on {split} for "
            f"{n_classes} classes: column k of predictions would not be class k."
        )
    if not np.isfinite(y_proba).all():
        raise ValueError(
            f"The model predicted non-finite probabilities on {split}: every "
            "confidence-based score downstream would be NaN."
        )
    return y_pred, y_proba, embedding[0] if embed else None


def _predictions_table(
    predicted: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    fit_rows: pd.Index,
    n_classes: int,
) -> pd.DataFrame:
    """One row per split row: `y_pred`, `proba_<class_id>` and `in_fit`, the train rows
    the model was fitted on."""
    table = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": split,
                    "row": np.arange(len(y_pred), dtype=np.int32),
                    "y_pred": y_pred.astype(np.int32),
                    **{
                        f"proba_{c}": y_proba[:, c].astype(np.float32)
                        for c in range(n_classes)
                    },
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


def _latent_table(
    y_true: np.ndarray, embedding: np.ndarray, *, random_state: int
) -> pd.DataFrame:
    """Up to `TSNE_MAX_SAMPLES` test rows of every class, with their embedding."""
    rng = np.random.default_rng(random_state)
    rows = np.sort(
        np.concatenate(
            [
                rng.choice(
                    members, size=min(TSNE_MAX_SAMPLES, len(members)), replace=False
                )
                for members in (np.flatnonzero(y_true == c) for c in np.unique(y_true))
            ]
        )
    )
    table = pd.DataFrame(
        embedding[rows].astype(np.float32),
        columns=[f"z_{i}" for i in range(embedding.shape[1])],
    )
    table.insert(0, "row", rows.astype(np.int32))
    return table


def classify(cfg) -> None:
    """Train the classifier, predict every split and write its metrics."""
    if cfg.fit.balance not in ("undersample", "none"):
        raise ValueError(
            f"Unknown balance: {cfg.fit.balance!r}. Valid: 'undersample', 'none'."
        )
    paths = paths_from_cfg(cfg)
    stage_dir = paths.of("classify")
    config = stage_config(cfg, "classify")
    inputs = upstream_ids(cfg, paths, "classify")
    if is_current(stage_dir, config=config, inputs=inputs, force=cfg.force):
        return

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    clear_dir(stage_dir)
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
    fit_df = _balance_train(cfg, train_df)
    model, training = _fit_classifier(
        cfg,
        trainer,
        fit_df,
        val_df,
        params=_resolve_classifier_params(
            cfg, num_cols=num_cols, cat_cols=cat_cols, meta=meta
        ),
        model_dir=stage_dir / "model",
    )
    save_to_json(training, stage_dir / "training.json")

    n_classes = meta["n_classes"]
    predicted = {
        name: _predict(
            trainer, model, df, name, n_classes=n_classes, embed=name == "test"
        )
        for name, df in zip(SPLITS, (train_df, val_df, test_df))
    }
    y_true = test_df[LABEL].to_numpy()
    y_pred, _, embedding = predicted["test"]
    save_to_json(
        compute_classification_metrics(y_true, y_pred), stage_dir / "metrics.json"
    )
    if embedding is not None:
        save_df(
            _latent_table(y_true, embedding, random_state=cfg.seed),
            stage_dir / "latent.parquet",
        )
    save_df(
        _predictions_table(
            {name: p[:2] for name, p in predicted.items()},
            fit_rows=fit_df.index,
            n_classes=n_classes,
        ),
        stage_dir / "predictions.parquet",
    )
    flush_timing(stage_dir / "timing.json")
    write_record(stage_dir, config=config, inputs=inputs)


def main() -> None:
    """Entry point for the classify stage."""
    classify(load_cli_config())


if __name__ == "__main__":
    main()
