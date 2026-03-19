import logging
import sys
from pathlib import Path

import torch
from ignite.engine import Events
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
)

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


def _build_trainer(
    model,
    loss_fn,
    optimizer,
    scheduler,
    device,
    max_grad_norm,
    tb_logger,
):
    """Configure the training engine with TensorBoard logging."""
    return (
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


def _build_validator(
    model,
    loss_fn,
    device,
    tb_logger,
    trainer,
    early_stopping_patience,
    early_stopping_min_delta,
    models_path,
):
    """Configure the validation engine with early stopping and checkpointing."""
    return (
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


def _build_tester(model, device, num_classes):
    """Configure the test engine with classification metrics (no TensorBoard)."""
    prepare_output = lambda x: (
        torch.softmax(x["output"]["logits"], dim=1),
        x["y_true"],
    )

    builder = EngineBuilder(test_step).with_state(model=model, device=device)
    builder.with_metric("accuracy", Accuracy(output_transform=prepare_output))

    for avg_type in ("macro", "weighted"):
        for name, cls in (("precision", Precision), ("recall", Recall), ("f1", F1)):
            builder.with_metric(
                f"{name}_{avg_type}",
                cls(
                    average=avg_type,
                    output_transform=prepare_output,
                    num_classes=num_classes,
                ),
            )

    for name, cls in (("precision", Precision), ("recall", Recall), ("f1", F1)):
        builder.with_metric(
            f"{name}_per_class",
            cls(
                average=None,
                output_transform=prepare_output,
                num_classes=num_classes,
            ),
        )

    builder.with_metric(
        "confusion_matrix",
        ConfusionMatrix(num_classes=num_classes, output_transform=prepare_output),
    )
    return builder.build()


def _to_python(v):
    """Convert a metric value to a JSON-serialisable Python type."""
    if isinstance(v, torch.Tensor):
        return v.cpu().numpy().tolist() if v.numel() > 1 else float(v)
    return v


def collect_test_results(metrics: dict) -> dict:
    """Convert raw engine metrics into a JSON-serialisable dict."""
    return {k: _to_python(v) for k, v in metrics.items()}


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
    train_loader = make_loader(
        train_df, num_cols, cat_cols, [label_col], train_dataloader_cfg
    )
    val_loader = make_loader(
        val_df, num_cols, cat_cols, [label_col], val_dataloader_cfg
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

    trainer = _build_trainer(
        model,
        loss_fn,
        optimizer,
        scheduler,
        device,
        max_grad_norm,
        tb_logger,
    )
    validator = _build_validator(
        model,
        loss_fn,
        device,
        tb_logger,
        trainer,
        early_stopping_patience,
        early_stopping_min_delta,
        models_path,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        logger.info(
            "Epoch [%d] Train Loss: %.6f",
            engine.state.epoch,
            engine.state.metrics["loss"],
        )
        validator.run(val_loader)
        logger.info(
            "Epoch [%d] Val Loss: %.6f",
            engine.state.epoch,
            validator.state.metrics["loss"],
        )

    try:
        trainer.run(train_loader, max_epochs=max_epochs)
    finally:
        tb_logger.close()

    logger.info("Training completed.")
    return model


SCALAR_METRICS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
]

PER_CLASS_METRICS = ["precision_per_class", "recall_per_class", "f1_per_class"]


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

    tester = _build_tester(model, device, model_params.num_classes)

    log_dir = tb_logs_path / "testing"
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorboardLogger(log_dir=log_dir)

    try:
        tester.run(test_loader)
    finally:
        tb_logger.close()

    metrics = tester.state.metrics

    # --- Logging ---
    for name, value in metrics.items():
        if name not in PER_CLASS_METRICS and name != "confusion_matrix":
            logger.info("%s: %s", name, value)

    # --- Persistence ---
    results = collect_test_results(metrics)
    save_to_json(results, json_logs_path / "test/summary.json")

    # --- TensorBoard ---
    writer = tb_logger.writer
    for name in SCALAR_METRICS:
        if name in metrics:
            writer.add_scalar(f"test/{name}", metrics[name], run_id)

    if "confusion_matrix" in metrics:
        writer.add_figure(
            "test/confusion_matrix",
            confusion_matrix_to_plot(metrics["confusion_matrix"].cpu().numpy()),
            run_id,
        )
    if "f1_per_class" in metrics:
        f1_dict = {
            f"class_{i}": float(v)
            for i, v in enumerate(metrics["f1_per_class"].cpu().numpy())
        }
        writer.add_figure("test/f1_per_class", dict_to_bar_plot(f1_dict), run_id)

    logger.info("Testing completed.")
    return model


def sup_classify(cfg):
    """Run supervised classification pipeline (train and/or test)."""
    json_logs_path = Path(cfg.path.json_logs)
    df_meta = load_from_json(json_logs_path / "data/df_meta.json")
    cfg.model.params.num_classes = df_meta["num_classes"]

    device = torch.device(cfg.device)
    logger.info("Using device: %s", device)

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

    stage = cfg.get("stage", "all")
    model = None

    if stage in ("all", "training"):
        model = train(
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
            tb_logs_path=tb_logs_path,
            models_path=models_path,
            max_epochs=cfg.loops.training.epochs,
            max_grad_norm=cfg.loops.training.get("max_grad_norm", 1.0),
            early_stopping_patience=cfg.loops.training.early_stopping.patience,
            early_stopping_min_delta=cfg.loops.training.early_stopping.min_delta,
            device=device,
        )

    if stage in ("all", "testing"):
        model = test(
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

    if stage not in ("all", "training", "testing"):
        logger.error("Unknown stage: %r. Valid: 'all', 'training', 'testing'.", stage)
        sys.exit(1)

    logger.info("All stages completed.")
    return model


def main():
    """Main entry point for supervised classification."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    sup_classify(cfg)


if __name__ == "__main__":
    main()
