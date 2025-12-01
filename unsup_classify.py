from typing import Optional, Tuple, List
from pathlib import Path
import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Average
from ignite.handlers.tensorboard_logger import *
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

from src.data.io import load_df

from src.common.config import load_config
from src.common.logging import setup_logger

from src.torch.dataset import MixedTabularDataset
from src.torch.batch import default_collate
from src.torch.model import ModelFactory
from src.torch.loss import LossFactory
from src.torch.engine import train_step, eval_step, test_step, filter_output

from src.ml.projection import tsne_projection, subsample_data_and_labels

from src.plot.array import confusion_matrix_to_plot, vectors_plot
from src.plot.dict import dict_to_bar_plot, dict_to_table

setup_logger()
logger = logging.getLogger(__name__)


def prepare_loader(
    cfg,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    logger.info("Preparing data for PyTorch...")

    base_path = Path(cfg.path.processed_data)
    file_base = f"{cfg.data.file_name}"
    ext = cfg.data.extension

    train_df = load_df(base_path / f"{file_base}_train.{ext}")
    benign_train_df = train_df[train_df[cfg.data.label_col] == cfg.data.benign_tag]

    if cfg.noise_fraction > 0:
        malicious_train_df = train_df[
            train_df[cfg.data.label_col] != cfg.data.benign_tag
        ]
        n_noise = int(len(benign_train_df) * cfg.noise_fraction)
        noise_train = malicious_train_df.sample(
            n=min(n_noise, len(malicious_train_df)), random_state=cfg.seed
        )
        train_df = (
            pd.concat([benign_train_df, noise_train])
            .sample(frac=1, random_state=cfg.seed)
            .reset_index(drop=True)
        )
    else:
        train_df = benign_train_df

    val_df = load_df(base_path / f"{file_base}_val.{ext}")
    val_df = val_df[val_df[cfg.data.label_col] == cfg.data.benign_tag]
    test_df = load_df(base_path / f"{file_base}_test.{ext}")

    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)

    train_dataset = MixedTabularDataset(
        train_df,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )
    val_dataset = MixedTabularDataset(
        val_df,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )
    test_dataset = MixedTabularDataset(
        test_df,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.loops.training.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.loops.training.dataloader.num_workers,
        pin_memory=cfg.loops.training.dataloader.pin_memory,
        collate_fn=default_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.loops.validation.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.loops.validation.dataloader.num_workers,
        pin_memory=cfg.loops.validation.dataloader.pin_memory,
        collate_fn=default_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.loops.test.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.loops.test.dataloader.num_workers,
        pin_memory=cfg.loops.test.dataloader.pin_memory,
        collate_fn=default_collate,
    )

    logger.info("Data preparation for PyTorch completed.")
    return (
        train_loader,
        val_loader,
        test_loader,
        test_df["bin_" + cfg.data.label_col].values,
        test_df["multi_" + cfg.data.label_col].values,
    )


def create_model_and_loss(cfg) -> Tuple[nn.Module, nn.Module]:
    logger.info("Creating model and loss function...")

    model = ModelFactory.create(cfg.model.name, cfg.model.params).to(cfg.device)
    loss_fn = LossFactory.create(cfg.loss.name, cfg.loss.params).to(cfg.device)

    return model, loss_fn


def create_optimizer(
    cfg, model: nn.Module, loss_fn: Optional[nn.Module] = None
) -> torch.optim.Optimizer:
    logger.info("Creating optimizer...")

    params = model.parameters()
    if loss_fn is not None and len(list(loss_fn.parameters())) > 0:
        params = list(model.parameters()) + list(loss_fn.parameters())

    optimizer = torch.optim.__dict__[cfg.optimizer.name](params, **cfg.optimizer.params)

    return optimizer


def create_scheduler(
    cfg, optimizer: torch.optim.Optimizer, dataloader: DataLoader
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    logger.info("Creating learning rate scheduler...")

    if cfg.scheduler is None:
        return None

    if cfg.scheduler.params.steps_per_epoch == "auto":
        cfg.scheduler.params.steps_per_epoch = len(dataloader)

    scheduler = torch.optim.lr_scheduler.__dict__[cfg.scheduler.name](
        optimizer, **cfg.scheduler.params
    )

    return scheduler


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    max_grad_norm: float = 1.0,
    max_epochs: int = 50,
    checkpoint_dir: Path = Path("./checkpoints"),
) -> None:
    tb_logger = TensorboardLogger(log_dir=cfg.path.logs)

    trainer = Engine(train_step)
    trainer.state.model = model
    trainer.state.loss_fn = loss_fn
    trainer.state.optimizer = optimizer
    trainer.state.scheduler = scheduler
    trainer.state.device = device
    trainer.state.max_grad_norm = max_grad_norm

    # Attach metrics to the trainer
    Average(output_transform=lambda x: x["loss"]).attach(trainer, "loss")

    validator = Engine(eval_step)
    validator.state.model = model
    validator.state.loss_fn = loss_fn
    validator.state.device = device

    # Attach metrics to the validator
    Average(output_transform=lambda x: x["loss"]).attach(validator, "loss")

    early_stopping_handler = EarlyStopping(
        patience=cfg.loops.training.early_stopping.patience,
        min_delta=cfg.loops.training.early_stopping.min_delta,
        score_function=lambda engine: -engine.state.metrics["loss"],
        trainer=trainer,
    )
    validator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_handler = ModelCheckpoint(
        dirname=checkpoint_dir,
        filename_prefix="best",
        score_function=lambda engine: -engine.state.metrics["loss"],
        score_name="val_loss",
        n_saved=1,
        global_step_transform=lambda engine, event_name: trainer.state.epoch,
        require_empty=False,
    )
    validator.add_event_handler(Events.COMPLETED, best_model_handler, {"model": model})

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_eval(engine):
        train_loss = engine.state.metrics["loss"]
        logger.info(f"Epoch [{engine.state.epoch}] Train Loss: {train_loss:.6f}")
        validator.run(val_loader)
        val_loss = validator.state.metrics["loss"]
        logger.info(f"Epoch [{engine.state.epoch}] Val Loss: {val_loss:.6f}")

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda x: {"loss": x["loss"], "grad_norm": x["grad_norm"]},
    )

    tb_logger.attach_output_handler(
        validator,
        event_name=Events.COMPLETED,
        tag="validation",
        metric_names=["loss"],
        global_step_transform=global_step_from_engine(trainer),
    )

    # tb_logger.attach(
    #     trainer,
    #     log_handler=WeightsHistHandler(model),
    #     event_name=Events.EPOCH_COMPLETED,
    # )

    # tb_logger.attach(
    #     trainer, log_handler=GradsHistHandler(model), event_name=Events.EPOCH_COMPLETED
    # )

    tb_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer, param_name="lr"),
        event_name=Events.ITERATION_COMPLETED,
    )

    trainer.run(train_loader, max_epochs=max_epochs)


def test(
    test_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    log_dir: Path,
    loss_fn: nn.Module,
    bin_labels: np.ndarray,
    multi_labels: np.ndarray,
    global_step: int = 0,
) -> None:
    tb_logger = TensorboardLogger(log_dir=log_dir)

    tester = Engine(test_step)
    tester.state.model = model
    tester.state.device = device
    loss_fn.reduction = "none"
    tester.state.loss_fn = loss_fn

    all_z = []
    all_preds = []

    @tester.on(Events.ITERATION_COMPLETED)
    def store_outputs(engine):
        output = engine.state.output
        all_z.append(output["output"]["z"].cpu().numpy())
        all_preds.append(output["loss"].cpu().numpy())

    @tester.on(Events.COMPLETED)
    def log_results(engine):
        fpr, tpr, thresholds = roc_curve(bin_labels, np.concatenate(all_preds))
        roc_auc = auc(fpr, tpr)
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
        predictions_binary = (np.concatenate(all_preds) >= optimal_threshold).astype(
            int
        )

        precision, recall, f1, _ = precision_recall_fscore_support(
            bin_labels,
            predictions_binary,
            average="macro",
            zero_division=0,
        )

        logger.info("Test Results:")
        logger.info(f"Test Precision (Macro): {precision:.4f}")
        logger.info(f"Test Recall (Macro): {recall:.4f}")
        logger.info(f"Test F1 Score (Macro): {f1:.4f}")
        logger.info(f"Test AUC-ROC: {roc_auc:.4f}")

        # Subsample latent representations and labels for visualization
        subsampled_z, subsampled_labels = subsample_data_and_labels(
            np.vstack(all_z),
            multi_labels,
            n_samples=3000,
            stratify=True,
        )
        projected_z = tsne_projection(subsampled_z)

        global_metrics = {
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "auc_roc": roc_auc,
        }

        tb_logger.writer.add_figure(
            "test/global_metrics", dict_to_bar_plot(global_metrics), global_step
        )
        tb_logger.writer.add_scalar("test/auc_roc", roc_auc, global_step)
        tb_logger.writer.add_scalar("test/f1_macro", f1, global_step)

        latent_figure = vectors_plot(projected_z, subsampled_labels)
        tb_logger.writer.add_figure("test/latent_space_2d", latent_figure, global_step)
        logger.info("Test results logged to TensorBoard.")

    logger.info("Running test evaluation...")
    tester.run(test_loader)
    tb_logger.close()


if __name__ == "__main__":
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    train_loader, val_loader, test_loader, bin_test_labels, multi_test_labels = (
        prepare_loader(cfg)
    )
    model, loss_fn = create_model_and_loss(cfg)
    optimizer = create_optimizer(cfg, model, loss_fn)
    scheduler = create_scheduler(cfg, optimizer, train_loader)

    # Training
    train(
        train_loader,
        val_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        device=torch.device(cfg.device),
        checkpoint_dir=Path(cfg.path.models),
        max_epochs=cfg.loops.training.epochs,
    )

    # Load best model for testing
    best_model_path = Path(cfg.path.models)
    checkpoint_files = list(best_model_path.glob("best_model_*.pt"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading best model from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=cfg.device)
        model.load_state_dict(checkpoint)

    # Testing
    test(
        test_loader=test_loader,
        model=model,
        device=torch.device(cfg.device),
        log_dir=Path(cfg.path.logs),
        loss_fn=loss_fn,
        bin_labels=bin_test_labels,
        multi_labels=multi_test_labels,
        global_step=cfg.global_step,
    )
