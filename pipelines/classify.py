import importlib
import inspect
import logging
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from pipelines import paths_from_cfg
from src.core.config import load_config, save_config
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
from src.domain.analysis.selective_prediction import entropy_risk, margin_risk, mcp_risk
from src.domain.data.preprocessing import random_undersample_df, subsample_df
from src.domain.plot.base import Plot, set_figure_format
from src.domain.plot.charts import bar_plot, line_plot, scatter_plot
from src.domain.plot.metrics import confusion_matrix_plot
from src.domain.plot.style import apply_plot_style, extended_palette
from src.domain.projection import stratified_subsample, tsne_projection
from src.engine.ml.model import MLClassifierFactory

setup_logger(log_file="resources/logs.txt")
apply_plot_style()
logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    """Seed the random, numpy and torch generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _supports_random_state(clf_cls: type) -> bool:
    """True if the estimator accepts a `random_state` parameter."""
    if "random_state" in inspect.signature(clf_cls.__init__).parameters:
        return True
    try:
        return "random_state" in clf_cls().get_params()
    except Exception:
        return False


@dataclass
class DataConfig:
    """Shared data parameters across stages."""

    processed_data_path: Path
    extension: str
    num_cols: list[str]
    cat_cols: list[str]
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


_METRIC_FNS: list[tuple[str, Callable]] = [
    ("precision", precision_score),
    ("recall", recall_score),
    ("f1", f1_score),
]


def _compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Overall accuracy, macro/weighted precision/recall/F1, plus per-class arrays."""
    full: dict = {"accuracy": float(accuracy_score(y_true, y_pred))}

    for avg in ("macro", "weighted"):
        for name, fn in _METRIC_FNS:
            full[f"{name}_{avg}"] = float(
                fn(y_true, y_pred, average=avg, zero_division=0)
            )

    for name, fn in _METRIC_FNS:
        full[f"{name}_per_class"] = fn(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()

    return full


def _cluster_error_rates(
    clusters: np.ndarray,
    error_mask: np.ndarray,
    extra_scores: dict[str, np.ndarray] | None = None,
) -> dict[str, dict]:
    """Per-cluster error counts, rate and mean extra scores, sorted by rate descending."""
    failed = clusters[error_mask]
    extra_scores = extra_scores or {}
    stats: dict[str, dict] = {}
    for c in np.unique(clusters):
        mask = clusters == c
        n_total = int(mask.sum())
        n_error = int((failed == c).sum())
        stats[str(c)] = {
            "n_error": n_error,
            "n_total": n_total,
            "error_rate": (n_error / n_total) if n_total > 0 else None,
            **{
                name: float(scores[mask].mean()) if n_total > 0 else None
                for name, scores in extra_scores.items()
            },
        }
    return dict(
        sorted(stats.items(), key=lambda x: x[1]["error_rate"] or 0.0, reverse=True)
    )


def _evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    clusters: np.ndarray | None = None,
) -> dict:
    """Per-class prediction quality plus per-cluster error rates and mean risk scores."""
    y_proba = np.asarray(y_proba)
    mcp, margin, entropy = (
        mcp_risk(y_proba),
        margin_risk(y_proba),
        entropy_risk(y_proba),
    )
    confidences = 1.0 - mcp

    has_cluster = clusters is not None
    global_error_mask = y_true != y_pred

    cluster_errors_total = (
        _cluster_error_rates(
            clusters,
            global_error_mask,
            extra_scores={
                "mcp_risk": mcp,
                "margin_risk": margin,
                "entropy_risk": entropy,
            },
        )
        if has_cluster
        else None
    )
    cluster_errors_by_class: dict[str, dict] | None = {} if has_cluster else None

    classes: dict[str, dict] = {}
    for label in np.unique(y_true):
        mask = y_true == label
        n_total = int(mask.sum())
        n_errors = int((y_true[mask] != y_pred[mask]).sum())
        error_mask = mask & global_error_mask

        if has_cluster:
            wrong_preds = y_pred[error_mask]
            wrong_clusters = clusters[error_mask]
            cluster_in_fn = {
                str(cls): np.unique(wrong_clusters[wrong_preds == cls]).tolist()
                for cls in np.unique(wrong_preds)
            }
            tp_clusters = clusters[mask & ~global_error_mask]
            cluster_in_tp = np.unique(tp_clusters).tolist()

            class_clusters = clusters[mask]
            cluster_errors_by_class[str(label)] = _cluster_error_rates(
                class_clusters, error_mask[mask]
            )
        else:
            cluster_in_fn = cluster_in_tp = None

        classes[str(label)] = {
            "tot_failures": n_errors,
            "tot_samples": n_total,
            "failure_rate": n_errors / n_total if n_total > 0 else None,
            "mean_confidence": (
                float(confidences[mask].mean()) if n_total > 0 else None
            ),
            "cluster_in_fn": cluster_in_fn,
            "cluster_in_tp": cluster_in_tp,
        }

    classes = dict(
        sorted(
            classes.items(),
            key=lambda x: x[1]["failure_rate"] or 0.0,
            reverse=True,
        )
    )

    return {
        "classes": classes,
        "clusters": {
            "global": cluster_errors_total,
            "by_class": cluster_errors_by_class,
        },
    }


def _per_sample_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    clusters: np.ndarray,
) -> pd.DataFrame:
    """Per-sample risk table (label-free scores plus labels) for the instance baselines."""
    yp = np.asarray(y_proba)
    return pd.DataFrame(
        {
            "cluster": np.asarray(clusters),
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "mcp_risk": mcp_risk(yp),
            "margin_risk": margin_risk(yp),
            "entropy_risk": entropy_risk(yp),
        }
    ).astype(
        {"mcp_risk": "float32", "margin_risk": "float32", "entropy_risk": "float32"}
    )


def _build_test_figures(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_mapping: dict,
    n_samples: int = 2000,
    embedding: np.ndarray | None = None,
) -> dict[str, Plot]:
    """Confusion matrix, per-class F1 bar and t-SNE scatter of raw features and embedding."""
    figures: dict[str, Plot] = {}

    classes = np.unique(y_true)
    class_names = [label_mapping.get(str(int(c)), str(c)) for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    figures["figure/testing/confusion_matrix"] = confusion_matrix_plot(
        cm, class_names=class_names, normalize=None
    )

    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_dict = {
        label_mapping.get(str(int(c)), str(c)): float(v)
        for c, v in zip(classes, f1_per_class)
    }
    figures["figure/testing/f1_per_class"] = bar_plot(
        list(f1_dict.keys()),
        list(f1_dict.values()),
        orientation="v",
        color=extended_palette(len(f1_dict)),
        sort=None,
        ylim=(0, 1),
    )

    correct = y_pred == y_true

    mis = ~correct
    total_mis = int(mis.sum())
    mis_per_class = {int(c): int((mis & (y_true == c)).sum()) for c in classes}
    keep_classes: list[int] = []
    cumulative = 0
    for c in sorted(mis_per_class, key=mis_per_class.get, reverse=True):
        if len(keep_classes) >= 2 and total_mis and cumulative >= 0.9 * total_mis:
            break
        keep_classes.append(c)
        cumulative += mis_per_class[c]
    if not keep_classes:
        keep_classes = [int(c) for c in classes]

    names = {c: label_mapping.get(str(c), str(c)) for c in keep_classes}
    prob_pos = np.flatnonzero(np.isin(y_true, keep_classes))
    sub = stratified_subsample(y_true[prob_pos], n_samples=n_samples, stratify=False)
    vis_idx = prob_pos[sub]

    def _projection(space: np.ndarray) -> Plot | None:
        return scatter_plot(
            tsne_projection(space[vis_idx], n_components=2),
            y_true[vis_idx],
            highlight_mask=~correct[vis_idx],
            names=names,
            marker_size=35.0,
            marker_alpha=0.85,
            legend_on_top=True,
        )

    figures["figure/testing/raw"] = _projection(X)
    if embedding is not None:
        figures["figure/testing/latent"] = _projection(embedding)
    return figures


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


def _publish_evaluation(
    bus: LogDispatcher,
    df_meta: dict,
    X_np: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    clusters: np.ndarray | None,
    *,
    eval_mode: str,
    embedding: np.ndarray | None = None,
    predictions_dir: Path | None = None,
) -> None:
    """Build metrics, confusion matrix, figures and per-sample dumps, then publish them."""
    full_metrics = {
        **_compute_classification_metrics(y_true, y_pred),
        "eval_mode": eval_mode,
    }
    pred_infos = {
        **_evaluate_predictions(y_true, y_pred, y_proba, clusters),
        "eval_mode": eval_mode,
    }
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true), normalize="true")
    figures = _build_test_figures(
        X_np, y_true, y_pred, df_meta["label_mapping"], embedding=embedding
    )
    if predictions_dir is not None and clusters is not None:
        save_df(
            _per_sample_scores(y_true, y_pred, y_proba, clusters),
            predictions_dir / "test_samples.parquet",
        )
    bus.publish(
        LogBundle.from_dict(
            {
                **figures,
                "json/testing/summary": full_metrics,
                "json/analysis/predictions/test": pred_infos,
                "pickle/analysis/confusion_matrices/test": cm,
            }
        )
    )


@timed
def _train_stage(
    cfg,
    paths: OutputPaths,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    df_meta: dict,
    num_cols: list[str],
    cat_cols: list[str],
    bus: LogDispatcher,
) -> None:
    """Train one model on the train split (optionally grid search) and save it."""
    kind = cfg.classifier.kind
    training_mod = _resolve_training_module(kind)
    context = _build_context(cfg, paths, df_meta, num_cols, cat_cols, label_col)
    params = _resolve_fit_params(cfg, kind, num_cols, cat_cols, df_meta)
    X, y = _prepare_train_payload(kind, train_df, feat_cols, label_col)
    X_val, y_val = _prepare_train_payload(kind, val_df, feat_cols, label_col)

    has_grid = "grid" in cfg.classifier and len(cfg.classifier.grid) > 0
    if has_grid:
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
            context=context,
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
            model, paths.models, name=cfg.classifier.name, params=params
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
            context,
            paths.models,
        )
        history = fit_summary.get("history", {})
        if history:
            bus.publish(LogBundle.from_dict(_training_history_figures(history)))
    logger.info("Model saved under %s", paths.models)


@timed
def _evaluate_stage(
    cfg,
    paths: OutputPaths,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    df_meta: dict,
    num_cols: list[str],
    cat_cols: list[str],
    bus: LogDispatcher,
) -> None:
    """Load the trained model, predict the test split, publish metrics + figures + dumps."""
    kind = cfg.classifier.kind
    training_mod = _resolve_training_module(kind)
    context = _build_context(cfg, paths, df_meta, num_cols, cat_cols, label_col)

    logger.info("Loading model from %s ...", paths.models)
    y_pred, y_proba, embedding = _predict_model(
        training_mod,
        paths.models,
        test_df,
        feat_cols,
        kind,
        context,
        return_embedding=True,
    )
    y_true = test_df[label_col].to_numpy()
    clusters = test_df["cluster"].to_numpy() if "cluster" in test_df.columns else None

    _publish_evaluation(
        bus,
        df_meta,
        test_df[feat_cols].to_numpy(),
        y_true,
        y_pred,
        y_proba,
        clusters,
        eval_mode="single_split",
        embedding=embedding,
        predictions_dir=paths.outputs / "analysis/predictions",
    )


def _oof_splits(base: pd.DataFrame, label_col: str, k: int, seed: int) -> list:
    """Deterministic stratified OOF folds over `base`; K capped to the rarest class."""
    y = base[label_col].to_numpy()
    k = min(k, int(np.unique(y, return_counts=True)[1].min()))
    if k < 2:
        raise ValueError(f"k-fold OOF needs >=2 samples per class, got k={k}.")
    return list(
        StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(base, y)
    )


@timed
def _train_kfold_stage(
    cfg,
    paths: OutputPaths,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    df_meta: dict,
    num_cols: list[str],
    cat_cols: list[str],
) -> None:
    """Train one classifier per OOF fold over train+test, each saved under `fold_{f}/`."""
    kind = cfg.classifier.kind
    training_mod = _resolve_training_module(kind)
    context = _build_context(cfg, paths, df_meta, num_cols, cat_cols, label_col)
    params = _resolve_fit_params(cfg, kind, num_cols, cat_cols, df_meta)
    X_val, y_val = _prepare_train_payload(kind, val_df, feat_cols, label_col)

    base = pd.concat([train_df, test_df], ignore_index=True)
    for f, (tr_idx, _) in enumerate(
        _oof_splits(base, label_col, cfg.kfold_splits, cfg.seed)
    ):
        fold_dir = paths.models / f"fold_{f}"
        fold_train = base.iloc[tr_idx]
        if cfg.balance == "undersample":
            fold_train = random_undersample_df(
                fold_train, label_col, random_state=cfg.seed
            )
        X, y = _prepare_train_payload(kind, fold_train, feat_cols, label_col)
        fold_ctx = {**context, "models_path": fold_dir} if kind == "dl" else context
        _fit_model(
            training_mod,
            cfg.classifier.name,
            params,
            X,
            y,
            X_val,
            y_val,
            fold_ctx,
            fold_dir,
        )
    logger.info("k-fold OOF: trained %d fold models under %s", f + 1, paths.models)


@timed
def _evaluate_kfold_stage(
    cfg,
    paths: OutputPaths,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    df_meta: dict,
    num_cols: list[str],
    cat_cols: list[str],
    bus: LogDispatcher,
) -> None:
    """Assemble leakage-free OOF predictions from the fold models and publish them."""
    kind = cfg.classifier.kind
    training_mod = _resolve_training_module(kind)
    context = _build_context(cfg, paths, df_meta, num_cols, cat_cols, label_col)

    base = pd.concat([train_df, test_df], ignore_index=True)
    y_true = base[label_col].to_numpy()
    clusters = base["cluster"].to_numpy() if "cluster" in base.columns else None
    y_pred = np.empty(len(base), dtype=y_true.dtype)
    y_proba = np.zeros((len(base), df_meta["num_classes"]))

    for f, (_, te_idx) in enumerate(
        _oof_splits(base, label_col, cfg.kfold_splits, cfg.seed)
    ):
        y_pred[te_idx], y_proba[te_idx] = _predict_model(
            training_mod,
            paths.models / f"fold_{f}",
            base.iloc[te_idx],
            feat_cols,
            kind,
            context,
        )

    _publish_evaluation(
        bus,
        df_meta,
        base[feat_cols].to_numpy(),
        y_true,
        y_pred,
        y_proba,
        clusters,
        eval_mode="oof_kfold",
        predictions_dir=paths.outputs / "analysis/predictions",
    )
    logger.info("k-fold OOF evaluation: %d samples over %d folds", len(base), f + 1)


@timed
def classify(cfg) -> None:
    """Run the supervised classification pipeline for a single classifier."""
    if cfg.balance not in ("undersample", "none"):
        raise ValueError(
            f"Unknown balance: {cfg.balance!r}. Valid: 'undersample', 'none'."
        )

    _seed_everything(cfg.seed)
    set_figure_format(cfg.plots.format)
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
        num_cols=num_cols,
        cat_cols=cat_cols,
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

    if use_kfold:
        _train_kfold_stage(
            cfg,
            paths,
            train_df,
            val_df,
            test_df,
            feat_cols,
            label_col,
            df_meta,
            num_cols,
            cat_cols,
        )
    else:
        _train_stage(
            cfg,
            paths,
            train_df,
            val_df,
            feat_cols,
            label_col,
            df_meta,
            num_cols,
            cat_cols,
            bus,
        )

    if use_kfold:
        _evaluate_kfold_stage(
            cfg,
            paths,
            train_df,
            test_df,
            feat_cols,
            label_col,
            df_meta,
            num_cols,
            cat_cols,
            bus,
        )
    else:
        _evaluate_stage(
            cfg,
            paths,
            test_df,
            feat_cols,
            label_col,
            df_meta,
            num_cols,
            cat_cols,
            bus,
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


if __name__ == "__main__":
    main()
