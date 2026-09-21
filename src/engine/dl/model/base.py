from torch import Tensor, nn


class ModelOutput(dict):
    """Dict subclass that enforces Tensor values."""

    def __init__(self, *, data: dict[str, Tensor] | None = None, **kwargs):
        if data is None:
            data = kwargs
        elif kwargs:
            raise ValueError("Cannot use both 'data' argument and keyword arguments.")
        for key, value in data.items():
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"ModelOutput['{key}'] must be a Tensor, got {type(value)}."
                )
        super().__init__(data)

    def __setitem__(self, key: str, value: Tensor) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(
                f"ModelOutput['{key}'] must be a Tensor, got {type(value)}."
            )
        super().__setitem__(key, value)


class BaseModel(nn.Module):
    """Base class for models."""

    def forward(self, x_numerical: Tensor, x_categorical: Tensor) -> ModelOutput:
        """Run the model; implemented by subclasses."""
        raise NotImplementedError

    def for_loss(self, output: ModelOutput, target: Tensor) -> tuple[Tensor, Tensor]:
        """Prepare (prediction, target) for the loss function."""
        return output["logits"], target


class ComposableClassifier(BaseModel):
    """Classifier composed of an encoder + linear head, over numerical and categorical inputs."""

    def __init__(self, encoder_module: nn.Module, head_module: nn.Module) -> None:
        super().__init__()
        self.encoder_module = encoder_module
        self.head_module = head_module

    def forward(self, x_numerical: Tensor, x_categorical: Tensor) -> ModelOutput:
        """Encode both feature blocks and return logits with the latent embedding."""
        z = self.encoder_module(x_numerical, x_categorical)
        return ModelOutput(logits=self.head_module(z), z=z)

    def for_loss(
        self, output: ModelOutput, target: Tensor, *args
    ) -> tuple[Tensor, ...]:
        """Prepare (logits, target, *extras) for the loss function."""
        return (output["logits"], target, *args)
