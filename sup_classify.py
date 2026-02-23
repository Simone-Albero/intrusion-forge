import logging
import sys
from pathlib import Path

import torch
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, ConfusionMatrix, Average
from ignite.handlers.tensorboard_logger import TensorboardLogger

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.utils import save_to_json, load_from_json

from src.data.io import load_listed_dfs
from src.data.preprocessing import subsample_df

from src.torch.module.checkpoint import load_latest_checkpoint
from src.torch.engine import train_step, eval_step, test_step
from src.torch.builders import (
    create_dataloader,
    create_dataset,
    create_model,
    create_loss,
    create_optimizer,
    create_scheduler,
    create_sampler,
)
from src.torch.data.batch import ensure_batch
from src.torch.data.sampler.batch import RivalClusterBatchSampler

from src.ignite.builders import EngineBuilder
from src.ignite.metrics import F1, Precision, Recall

from src.plot.array import confusion_matrix_to_plot
from src.plot.dict import dict_to_bar_plot

setup_logger()
logger = logging.getLogger(__name__)


def load_data(base_path, extension, label_col, n_samples, random_state):
    """Load train/val/test splits and optionally subsample the training set."""
    train_df, val_df, test_df = load_listed_dfs(
        Path(base_path),
        [f"train.{extension}", f"val.{extension}", f"test.{extension}"],
    )
    if n_samples is not None:
        train_df = subsample_df(train_df, n_samples, random_state, label_col)
    return train_df, val_df, test_df


def make_loader(df, num_cols, cat_cols, label_cols, dataloader_cfg):
    return create_dataloader(
        create_dataset(df, num_cols, cat_cols, label_cols), dataloader_cfg
    )


def _make_sampler(df, label_col, sampler_name, sampler_params):
    """Create a batch sampler with cluster and label arrays from a dataframe."""
    params = dict(sampler_params)
    params["clusters"] = df["cluster"].values
    params["labels"] = df[label_col].values
    return create_sampler(name=sampler_name, params=params)


def _build_rival_updater(model, device, batch_sampler, loss_fn, invalid_classes=None):
    """Build an engine that collects embeddings and updates cluster rivals."""

    def step(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = ensure_batch(batch).to(device, non_blocking=True)
            out = model(*batch.features)
            z = torch.nn.functional.normalize(out["z"], dim=1, eps=1e-8)
        engine.state.all_z.append(z.cpu())
        engine.state.all_y.append(batch.labels[0].cpu())
        engine.state.all_c.append(batch.labels[1].cpu())

    engine = Engine(step)

    @engine.on(Events.STARTED)
    def reset(engine):
        engine.state.all_z = []
        engine.state.all_y = []
        engine.state.all_c = []

    @engine.on(Events.COMPLETED)
    def _compute_rivals(engine):
        all_z = torch.cat(engine.state.all_z).numpy()
        all_y = torch.cat(engine.state.all_y).numpy()
        all_c = torch.cat(engine.state.all_c).numpy()

        cluster_rivals = RivalClusterBatchSampler.compute_rivals(
            all_z, all_y, all_c, invalid_classes
        )
        if not cluster_rivals:
            logger.warning("No cluster rivals found, skipping rival update.")
            return

        batch_sampler.cluster_rivals = cluster_rivals
        loss_fn.update_cluster_rivals(cluster_rivals)
        logger.info(f"Cluster rivals: {cluster_rivals}")

    return engine


def train(
    processed_data_path,
    extension,
    num_cols,
    cat_cols,
    label_col,
    n_samples,
    random_state,
    train_dataloader_cfg,
    val_dataloader_cfg,
    model_name,
    model_params,
    loss_name,
    loss_params,
    optimizer_name,
    optimizer_params,
    scheduler_name,
    scheduler_params,
    sampler_name,
    sampler_params,
    tb_logs_path,
    models_path,
    max_epochs,
    max_grad_norm,
    early_stopping_patience,
    early_stopping_min_delta,
    device,
):
    """Train the supervised classifier."""
    logger.info("Starting training phase...")

    train_df, val_df, _ = load_data(
        processed_data_path, extension, label_col, n_samples, random_state
    )

    train_sampler = val_sampler = None
    if sampler_name is not None:
        train_sampler = _make_sampler(train_df, label_col, sampler_name, sampler_params)
        val_sampler = _make_sampler(val_df, label_col, sampler_name, sampler_params)
        train_dataloader_cfg["batch_sampler"] = train_sampler
        val_dataloader_cfg["batch_sampler"] = val_sampler

    extra_cols = ["cluster"] if train_sampler is not None else []
    train_loader = make_loader(
        train_df, num_cols, cat_cols, [label_col] + extra_cols, train_dataloader_cfg
    )
    val_loader = make_loader(
        val_df, num_cols, cat_cols, [label_col] + extra_cols, val_dataloader_cfg
    )

    model = create_model(model_name, model_params, device)
    loss_fn = create_loss(loss_name, loss_params, device)
    optimizer = create_optimizer(optimizer_name, optimizer_params, model, loss_fn)
    scheduler = create_scheduler(
        scheduler_name, scheduler_params, optimizer, train_loader
    )

    log_dir = tb_logs_path / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorboardLogger(log_dir=log_dir)

    trainer = (
        EngineBuilder(train_step)
        .with_state(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_grad_norm=max_grad_norm,
        )
        .with_metric("loss", Average(output_transform=lambda x: x["loss"]))
        .with_tensorboard(
            tb_logger=tb_logger,
            tag="train",
            output_transform=lambda x: {"loss": x["loss"], "grad_norm": x["grad_norm"]},
        )
        .with_optimizer_logging(tb_logger=tb_logger, optimizer=optimizer)
        .with_weights_logging(tb_logger=tb_logger, model=model)
        .with_gradients_logging(tb_logger=tb_logger, model=model)
        .build()
    )

    validator = (
        EngineBuilder(eval_step)
        .with_state(model=model, loss_fn=loss_fn, device=device)
        .with_metric("loss", Average(output_transform=lambda x: x["loss"]))
        .with_early_stopping(
            trainer=trainer,
            metric="loss",
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
        )
        .with_checkpointing(
            trainer=trainer,
            checkpoint_dir=models_path,
            objects_to_save={"model": model},
            metric="loss",
        )
        .with_tensorboard(
            tb_logger=tb_logger,
            event=Events.COMPLETED,
            tag="validation",
            metric_names=["loss"],
            trainer=trainer,
        )
        .build()
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        logger.info(
            f"Epoch [{engine.state.epoch}] Train Loss: {engine.state.metrics['loss']:.6f}"
        )
        validator.run(val_loader)
        logger.info(
            f"Epoch [{engine.state.epoch}] Val Loss: {validator.state.metrics['loss']:.6f}"
        )

    rival_updater = (
        _build_rival_updater(model, device, train_loader.batch_sampler, loss_fn)
        if train_sampler is not None
        else None
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_cluster_rivals(engine):
        if train_sampler is None or not hasattr(
            train_loader.batch_sampler, "cluster_rivals"
        ):
            return
        rival_updater.run(train_loader)

    try:
        trainer.run(train_loader, max_epochs=max_epochs)
    finally:
        tb_logger.close()

    logger.info("Training completed.")
    return model


def test(
    processed_data_path,
    extension,
    num_cols,
    cat_cols,
    label_col,
    n_samples,
    random_state,
    test_dataloader_cfg,
    model_name,
    model_params,
    models_path,
    tb_logs_path,
    json_logs_path,
    run_id,
    device,
):
    """Test the trained classifier."""
    logger.info("Starting testing phase...")

    _, _, test_df = load_data(
        processed_data_path, extension, label_col, n_samples, random_state
    )
    test_loader = make_loader(
        test_df, num_cols, cat_cols, [label_col], test_dataloader_cfg
    )
    model = create_model(model_name, model_params, device)
    load_latest_checkpoint(models_path, model, device)

    num_classes = model_params.num_classes
    log_dir = tb_logs_path / "testing"
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorboardLogger(log_dir=log_dir)

    prepare_output = lambda x: (
        torch.softmax(x["output"]["logits"], dim=1),
        x["y_true"],
    )

    tester = EngineBuilder(test_step).with_state(model=model, device=device)
    tester.with_metric("accuracy", Accuracy(output_transform=prepare_output))

    for avg_type in ("macro", "weighted"):
        for name, cls in (("precision", Precision), ("recall", Recall), ("f1", F1)):
            tester.with_metric(
                f"{name}_{avg_type}",
                cls(
                    average=avg_type,
                    output_transform=prepare_output,
                    num_classes=num_classes,
                ),
            )

    for name, cls in (("precision", Precision), ("recall", Recall), ("f1", F1)):
        tester.with_metric(
            f"{name}_per_class",
            cls(average=None, output_transform=prepare_output, num_classes=num_classes),
        )

    tester.with_metric(
        "confusion_matrix",
        ConfusionMatrix(num_classes=num_classes, output_transform=prepare_output),
    )
    tester = tester.build()

    _scalar_metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    _per_class_metrics = {
        "precision_per_class",
        "recall_per_class",
        "f1_per_class",
        "confusion_matrix",
    }

    @tester.on(Events.COMPLETED)
    def log_results(engine):
        metrics = engine.state.metrics

        for name, value in metrics.items():
            if name not in _per_class_metrics:
                logger.info(f"{name}: {value}")

        def to_python(v):
            if isinstance(v, torch.Tensor):
                return v.cpu().numpy().tolist() if v.numel() > 1 else float(v)
            return v

        save_to_json(
            {k: to_python(v) for k, v in metrics.items()},
            json_logs_path / "test/summary.json",
        )

        for name in _scalar_metrics:
            if name in metrics:
                tb_logger.writer.add_scalar(f"test/{name}", metrics[name], run_id)

        if "confusion_matrix" in metrics:
            tb_logger.writer.add_figure(
                "test/confusion_matrix",
                confusion_matrix_to_plot(
                    metrics["confusion_matrix"].cpu().numpy(),
                    normalize="true",
                ),
                run_id,
            )

        if "f1_per_class" in metrics:
            f1_dict = {
                f"class_{i}": float(v)
                for i, v in enumerate(metrics["f1_per_class"].cpu().numpy())
            }
            tb_logger.writer.add_figure(
                "test/f1_per_class", dict_to_bar_plot(f1_dict), run_id
            )

    try:
        tester.run(test_loader)
    finally:
        tb_logger.close()

    logger.info("Testing completed.")
    return _scalar_metrics, _per_class_metrics


def sup_classify():
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    json_logs_path = Path(cfg.path.json_logs)
    df_meta = load_from_json(json_logs_path / "data/metadata.json")
    cfg.model.params.num_classes = df_meta["num_classes"]

    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    label_col = "encoded_" + cfg.data.label_col

    processed_data_path = Path(cfg.path.processed_data)
    models_path = Path(cfg.path.models)
    tb_logs_path = Path(cfg.path.tb_logs)

    common = dict(
        processed_data_path=processed_data_path,
        extension=cfg.data.extension,
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=label_col,
        n_samples=cfg.n_samples,
        random_state=cfg.seed,
    )

    train_kwargs = dict(
        **common,
        train_dataloader_cfg=cfg.loops.training.dataloader,
        val_dataloader_cfg=cfg.loops.validation.dataloader,
        model_name=cfg.model.name,
        model_params=cfg.model.params,
        loss_name=cfg.loss.name,
        loss_params=cfg.loss.params,
        optimizer_name=cfg.optimizer.name,
        optimizer_params=cfg.optimizer.params,
        scheduler_name=cfg.scheduler.name,
        scheduler_params=cfg.scheduler.params,
        sampler_name=cfg.sampler.name if "sampler" in cfg else None,
        sampler_params=cfg.sampler.params if "sampler" in cfg else None,
        tb_logs_path=tb_logs_path,
        models_path=models_path,
        max_epochs=cfg.loops.training.epochs,
        max_grad_norm=cfg.loops.training.get("max_grad_norm", 1.0),
        early_stopping_patience=cfg.loops.training.early_stopping.patience,
        early_stopping_min_delta=cfg.loops.training.early_stopping.min_delta,
        device=device,
    )

    test_kwargs = dict(
        **common,
        test_dataloader_cfg=cfg.loops.test.dataloader,
        model_name=cfg.model.name,
        model_params=cfg.model.params,
        models_path=models_path,
        tb_logs_path=tb_logs_path,
        json_logs_path=json_logs_path,
        run_id=cfg.run_id,
        device=device,
    )

    stage = cfg.get("stage", "all")
    if stage in ("all", "training"):
        model = train(**train_kwargs)
    if stage in ("all", "testing"):
        scalar_metrics, per_class_metrics = test(**test_kwargs)
    if stage not in ("all", "training", "testing"):
        logger.error(f"Unknown stage: {stage!r}. Valid: 'all', 'training', 'testing'.")
        sys.exit(1)

    logger.info("All stages completed.")
    return model, scalar_metrics or {}, per_class_metrics or {}


if __name__ == "__main__":
    sup_classify()
