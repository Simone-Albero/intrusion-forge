import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Average
from torch.utils.data import DataLoader

from src.domain.training.base import ComponentSpec
from src.engine.dl.builders import (
    create_dataloader,
    create_dataset,
    create_loss,
    create_optimizer,
    create_scheduler,
)
from src.engine.dl.engine import eval_step, train_step
from src.engine.dl.ignite_builder import build_engine
from src.engine.dl.infer import df_to_tensors, forward_eval
from src.engine.dl.model import DLClassifierFactory

logger = logging.getLogger(__name__)


def _create_model(name: str, params: dict, device: torch.device) -> nn.Module:
    return DLClassifierFactory.create(name, params).to(device)


def _build_train_engine(
    model: nn.Module,
    *,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    max_grad_norm: float,
) -> tuple[Engine, dict[str, list[float]]]:
    """Engine that trains for one epoch and collects per-step loss into history."""
    history: dict[str, list[float]] = {"loss": []}

    def _collect(engine) -> None:
        history["loss"].append(float(engine.state.output["loss"]))

    engine = build_engine(
        train_step,
        state={
            "model": model,
            "loss_fn": loss_fn,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "device": device,
            "max_grad_norm": max_grad_norm,
        },
        metric=("loss", Average(output_transform=lambda x: x["loss"])),
        handlers=[(Events.ITERATION_COMPLETED, _collect)],
    )
    return engine, history


def _build_validation_engine(
    model: nn.Module,
    *,
    loss_fn: nn.Module,
    device: torch.device,
    trainer: Engine,
    patience: int,
    min_delta: float,
    checkpoint_dir: Path,
) -> tuple[Engine, ModelCheckpoint]:
    """Validator engine with early stopping, and the handler keeping its best epoch."""
    # Both score functions minimize loss: Ignite's handlers maximize by convention.
    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        score_function=lambda engine: -engine.state.metrics["loss"],
        trainer=trainer,
    )
    checkpoint = ModelCheckpoint(
        dirname=checkpoint_dir,
        score_function=lambda engine: -engine.state.metrics["loss"],
        score_name="loss",
        n_saved=1,
        global_step_transform=lambda engine, _: trainer.state.epoch,
        require_empty=False,
    )

    validator = build_engine(
        eval_step,
        state={"model": model, "loss_fn": loss_fn, "device": device},
        metric=("loss", Average(output_transform=lambda x: x["loss"])),
        handlers=[
            (Events.COMPLETED, early_stopping),
            (Events.COMPLETED, lambda engine: checkpoint(engine, {"model": model})),
        ],
    )
    return validator, checkpoint


@dataclass
class DLTrainer:
    """Fits PyTorch classifiers through Ignite engines over a tabular dataset."""

    device: torch.device
    num_cols: list[str]
    cat_cols: list[str]
    label_col: str
    class_weights: list[float] | None
    loss: ComponentSpec
    optimizer: ComponentSpec
    scheduler: ComponentSpec
    epochs: int
    max_grad_norm: float
    patience: int
    min_delta: float
    train_loader_params: dict
    val_loader_params: dict

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        """The whole frame: the dataset selects its own feature columns."""
        return df

    def prepare(self, df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, None]:
        """The whole frame: the dataset selects its own feature and label columns."""
        return df, None

    def _loader(self, df: pd.DataFrame, params: dict) -> DataLoader:
        return create_dataloader(
            create_dataset(
                df, self.num_cols, self.cat_cols, label_col=[self.label_col]
            ),
            params,
        )

    def fit(
        self,
        name: str,
        params: dict,
        X: pd.DataFrame,
        y: object,
        *,
        X_val: pd.DataFrame,
        save_dir: Path,
    ) -> tuple[nn.Module, dict]:
        """Train with early stopping and return the best checkpoint plus its loss history."""
        # Emptied first: the handler deletes only the files it saved itself, so each fit
        # would otherwise leave one more checkpoint behind.
        checkpoint_dir = Path(save_dir) / "checkpoints"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True)

        loss_params = dict(self.loss.params)
        # The trainer's weights are None whenever the split was rebalanced.
        if loss_params.get("class_weight") == "auto":
            loss_params["class_weight"] = self.class_weights

        model = _create_model(name, params, self.device)
        loss_fn = create_loss(self.loss.name, loss_params, self.device)

        train_loader = self._loader(X, self.train_loader_params)
        val_loader = self._loader(X_val, self.val_loader_params)

        optimizer = create_optimizer(
            self.optimizer.name, self.optimizer.params, model, loss_fn=loss_fn
        )
        scheduler = create_scheduler(
            self.scheduler.name, self.scheduler.params, optimizer, train_loader
        )

        trainer, history = _build_train_engine(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            max_grad_norm=self.max_grad_norm,
        )
        validator, checkpoint = _build_validation_engine(
            model,
            loss_fn=loss_fn,
            device=self.device,
            trainer=trainer,
            patience=self.patience,
            min_delta=self.min_delta,
            checkpoint_dir=checkpoint_dir,
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def _validate(engine) -> None:
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

        trainer.run(train_loader, max_epochs=self.epochs)

        best = checkpoint.last_checkpoint
        if best is None:
            raise RuntimeError(
                f"No checkpoint was saved in {checkpoint_dir}; "
                "refusing to return untrained weights."
            )
        model.load_state_dict(
            torch.load(best, map_location=self.device, weights_only=True)
        )
        logger.info("Best checkpoint %s reloaded after training.", best.name)

        return model, {"history": history}

    def grid_search(
        self,
        name: str,
        params: dict,
        grid: dict,
        X: pd.DataFrame,
        y: object,
        *,
        scoring: str,
        cv: int,
        max_samples: int | None,
        random_state: int,
    ) -> tuple[nn.Module, dict]:
        """Not available: grid search is implemented for ML classifiers only."""
        raise NotImplementedError(
            "Grid search is implemented for ML classifiers only; "
            f"{name!r} is a DL classifier."
        )

    def predict(
        self, model: nn.Module, X: pd.DataFrame, *, return_embedding: bool = False
    ) -> tuple:
        """Predict a DataFrame → (y_pred, y_proba), plus the latent embedding on request."""
        inputs = df_to_tensors(
            X,
            [self.num_cols, self.cat_cols],
            dtypes=[torch.float32, torch.long],
        )
        output = forward_eval(model, inputs, self.device)
        probs = F.softmax(output["logits"].cpu(), dim=1)
        y_pred = probs.argmax(dim=1).numpy()
        y_proba = probs.numpy()
        if return_embedding:
            z = output["z"].cpu().numpy() if "z" in output else None
            return y_pred, y_proba, z
        return y_pred, y_proba

    def save(
        self,
        model: nn.Module,
        path: Path,
        *,
        name: str,
        params: dict,
    ) -> None:
        """Save the state dict and its metadata to `path / model.pt`."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "name": name, "params": params},
            path / "model.pt",
        )

    def load(self, path: Path) -> nn.Module:
        """Load the model from `path / model.pt` onto this trainer's device."""
        ckpt = torch.load(
            Path(path) / "model.pt", map_location="cpu", weights_only=True
        )
        model = _create_model(ckpt["name"], ckpt["params"], self.device)
        model.load_state_dict(ckpt["state_dict"])
        return model

    def has_model(self, path: Path) -> bool:
        """True when `path` holds a saved state dict."""
        return (Path(path) / "model.pt").exists()
