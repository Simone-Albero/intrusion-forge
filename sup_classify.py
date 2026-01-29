import json
import shutil
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
from src.data.io import load_data_splits
from src.data.preprocessing import subsample_df, random_undersample_df
from src.torch.module.checkpoint import load_latest_checkpoint
from src.torch.engine import exclude_ignored_classes, train_step, eval_step, test_step
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


def load_data(
    processed_data_path,
    file_name,
    extension,
    num_cols,
    cat_cols,
    label_col,
    n_samples,
    random_state,
    train_dataloader_cfg,
    val_dataloader_cfg,
    test_dataloader_cfg,
):
    """Load and prepare data loaders."""
    train_df, val_df, test_df = load_data_splits(
        processed_data_path, file_name, extension
    )

    train_df = random_undersample_df(train_df, label_col, random_state)

    if n_samples is not None:
        train_df = subsample_df(train_df, n_samples, random_state, label_col)

    def make_loader(df, dataloader_cfg):
        dataset = create_dataset(df, num_cols, cat_cols, label_col)
        return create_dataloader(dataset, dataloader_cfg)

    return (
        make_loader(train_df, train_dataloader_cfg),
        make_loader(val_df, val_dataloader_cfg),
        make_loader(test_df, test_dataloader_cfg),
    )


def train(
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    device,
    tb_logs_path,
    models_path,
    max_epochs,
    max_grad_norm,
    early_stopping_patience,
    early_stopping_min_delta,
):
    """Train the model with validation and checkpointing."""
    log_dir = tb_logs_path / "training"
    # if log_dir.exists():
    #     shutil.rmtree(log_dir)
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

    try:
        trainer.run(train_loader, max_epochs=max_epochs)
    finally:
        tb_logger.close()


def test(
    test_loader,
    model,
    device,
    num_classes,
    tb_logs_path,
    json_logs_path,
    run_id,
    ignore_classes=None,
):
    """Test the model and log results."""
    log_dir = tb_logs_path / "testing"
    # if log_dir.exists():
    #     shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    tb_logger = TensorboardLogger(log_dir=log_dir)

    if ignore_classes:
        prepare_output = lambda x: exclude_ignored_classes(
            x["output"]["logits"], x["y_true"], ignore_classes
        )
    else:
        prepare_output = lambda x: (x["output"]["logits"], x["y_true"])

    tester = EngineBuilder(test_step).with_state(model=model, device=device)
    tester.with_metric("accuracy", Accuracy(output_transform=prepare_output))

    for avg_type in ["macro", "weighted"]:
        tester.with_metric(
            f"precision_{avg_type}",
            Precision(
                average=avg_type,
                output_transform=prepare_output,
                num_classes=num_classes,
            ),
        )
        tester.with_metric(
            f"recall_{avg_type}",
            Recall(
                average=avg_type,
                output_transform=prepare_output,
                num_classes=num_classes,
            ),
        )
        tester.with_metric(
            f"f1_{avg_type}",
            F1(
                average=avg_type,
                output_transform=prepare_output,
                num_classes=num_classes,
            ),
        )

    for metric_name, metric_cls in [
        ("precision", Precision),
        ("recall", Recall),
        ("f1", F1),
    ]:
        tester.with_metric(
            f"{metric_name}_per_class",
            metric_cls(
                average=None, output_transform=prepare_output, num_classes=num_classes
            ),
        )

    tester.with_metric(
        "confusion_matrix",
        ConfusionMatrix(num_classes=num_classes, output_transform=prepare_output),
    )

    tester = tester.build()

    @tester.on(Events.COMPLETED)
    def log_results(engine):
        """Log metrics to console, JSON, and TensorBoard."""
        metrics = engine.state.metrics

        # Console logging
        logger.info("Test Results:")
        for name, value in metrics.items():
            if name not in [
                "confusion_matrix",
                "precision_per_class",
                "recall_per_class",
                "f1_per_class",
            ]:
                logger.info(f"{name}: {value}")

        # JSON logging
        metrics_to_save = {}
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                metrics_to_save[name] = (
                    value.cpu().numpy().tolist() if value.numel() > 1 else float(value)
                )
            else:
                metrics_to_save[name] = value

        json_path = (
            json_logs_path / f"test_summary{f'_{run_id}' if run_id else ''}.json"
        )
        save_to_json(metrics_to_save, json_path)

        # TensorBoard logging
        global_metrics = [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        ]

        for name in global_metrics:
            if name in metrics:
                tb_logger.writer.add_scalar(
                    f"test/metrics/{name}", metrics[name], run_id
                )

        if "confusion_matrix" in metrics:
            cm_figure = confusion_matrix_to_plot(
                metrics["confusion_matrix"].cpu().numpy(),
                title="Confusion Matrix",
                normalize="true",
            )
            tb_logger.writer.add_figure("test/confusion_matrix", cm_figure, run_id)

        if "f1_per_class" in metrics:
            f1_dict = {
                f"class_{i}": float(v)
                for i, v in enumerate(metrics["f1_per_class"].cpu().numpy())
            }
            tb_logger.writer.add_figure(
                "test/f1_per_class", dict_to_bar_plot(f1_dict), run_id
            )

        tb_logger.writer.add_figure(
            "test/global_metrics",
            dict_to_bar_plot({k: metrics[k] for k in global_metrics if k in metrics}),
            run_id,
        )

    try:
        tester.run(test_loader)
    finally:
        tb_logger.close()


def run_training(
    processed_data_path,
    file_name,
    extension,
    num_cols,
    cat_cols,
    label_col,
    n_samples,
    random_state,
    train_dataloader_cfg,
    val_dataloader_cfg,
    test_dataloader_cfg,
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

    train_loader, val_loader, _ = load_data(
        processed_data_path,
        file_name,
        extension,
        num_cols,
        cat_cols,
        label_col,
        n_samples,
        random_state,
        train_dataloader_cfg,
        val_dataloader_cfg,
        test_dataloader_cfg,
    )
    model = create_model(model_name, model_params, device)
    loss_fn = create_loss(loss_name, loss_params, device)
    optimizer = create_optimizer(optimizer_name, optimizer_params, model, loss_fn)
    scheduler = create_scheduler(
        scheduler_name, scheduler_params, optimizer, train_loader
    )

    train(
        train_loader,
        val_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        device,
        tb_logs_path,
        models_path,
        max_epochs,
        max_grad_norm,
        early_stopping_patience,
        early_stopping_min_delta,
    )
    logger.info("Training completed")


def run_testing(
    processed_data_path,
    file_name,
    extension,
    num_cols,
    cat_cols,
    label_col,
    n_samples,
    random_state,
    train_dataloader_cfg,
    val_dataloader_cfg,
    test_dataloader_cfg,
    model_name,
    model_params,
    models_path,
    tb_logs_path,
    json_logs_path,
    run_id,
    ignore_classes,
    device,
):
    """Test the trained classifier."""
    logger.info("Starting testing phase...")

    _, _, test_loader = load_data(
        processed_data_path,
        file_name,
        extension,
        num_cols,
        cat_cols,
        label_col,
        n_samples,
        random_state,
        train_dataloader_cfg,
        val_dataloader_cfg,
        test_dataloader_cfg,
    )
    model = create_model(model_name, model_params, device)
    load_latest_checkpoint(models_path, model, device)

    test(
        test_loader,
        model,
        device,
        model_params.num_classes,
        tb_logs_path,
        json_logs_path,
        run_id,
        ignore_classes,
    )
    logger.info("Testing completed")


def main():
    """Main training pipeline for supervised learning."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    json_logs_path = Path(cfg.path.json_logs)
    df_meta = load_from_json(
        json_logs_path / "metadata" / f"df_{cfg.run_id}.json",
    )
    cfg.model.params.num_classes = df_meta["num_classes"]
    cfg.loss.params.class_weight = df_meta["class_weights"]
    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = "multi_" + cfg.data.label_col

    processed_data_path = Path(cfg.path.processed_data) / cfg.data.name
    models_path = Path(cfg.path.models)
    tb_logs_path = Path(cfg.path.tb_logs)
    ignore_classes = list(cfg.ignore_classes) if cfg.get("ignore_classes") else None
    run_id = cfg.get("run_id", None)

    # Run pipeline based on stage
    stage = cfg.get("stage", "all")
    if stage == "all":
        run_training(
            processed_data_path,
            cfg.data.file_name,
            cfg.data.extension,
            num_cols,
            cat_cols,
            label_col,
            cfg.n_samples,
            cfg.seed,
            cfg.loops.training.dataloader,
            cfg.loops.validation.dataloader,
            cfg.loops.test.dataloader,
            cfg.model.name,
            cfg.model.params,
            cfg.loss.name,
            cfg.loss.params,
            cfg.optimizer.name,
            cfg.optimizer.params,
            cfg.scheduler.name,
            cfg.scheduler.params,
            tb_logs_path,
            models_path,
            cfg.loops.training.epochs,
            cfg.loops.training.get("max_grad_norm", 1.0),
            cfg.loops.training.early_stopping.patience,
            cfg.loops.training.early_stopping.min_delta,
            device,
        )
        run_testing(
            processed_data_path,
            cfg.data.file_name,
            cfg.data.extension,
            num_cols,
            cat_cols,
            label_col,
            cfg.n_samples,
            cfg.seed,
            cfg.loops.training.dataloader,
            cfg.loops.validation.dataloader,
            cfg.loops.test.dataloader,
            cfg.model.name,
            cfg.model.params,
            models_path,
            tb_logs_path,
            json_logs_path,
            run_id,
            ignore_classes,
            device,
        )
    elif stage == "training":
        run_training(
            processed_data_path,
            cfg.data.file_name,
            cfg.data.extension,
            num_cols,
            cat_cols,
            label_col,
            cfg.n_samples,
            cfg.seed,
            cfg.loops.training.dataloader,
            cfg.loops.validation.dataloader,
            cfg.loops.test.dataloader,
            cfg.model.name,
            cfg.model.params,
            cfg.loss.name,
            cfg.loss.params,
            cfg.optimizer.name,
            cfg.optimizer.params,
            cfg.scheduler.name,
            cfg.scheduler.params,
            tb_logs_path,
            models_path,
            cfg.loops.training.epochs,
            cfg.loops.training.get("max_grad_norm", 1.0),
            cfg.loops.training.early_stopping.patience,
            cfg.loops.training.early_stopping.min_delta,
            device,
        )
    elif stage == "testing":
        run_testing(
            processed_data_path,
            cfg.data.file_name,
            cfg.data.extension,
            num_cols,
            cat_cols,
            label_col,
            cfg.n_samples,
            cfg.seed,
            cfg.loops.training.dataloader,
            cfg.loops.validation.dataloader,
            cfg.loops.test.dataloader,
            cfg.model.name,
            cfg.model.params,
            models_path,
            tb_logs_path,
            json_logs_path,
            run_id,
            ignore_classes,
            device,
        )
    else:
        logger.error(f"Unknown stage: {stage}. Valid: 'all', 'training', 'testing'")
        sys.exit(1)


if __name__ == "__main__":
    main()
