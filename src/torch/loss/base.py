from torch import nn, Tensor


class BaseLoss(nn.Module):
    """Base class for loss functions with configurable reduction ('mean', 'sum', 'none')."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'.")
        self.reduction = reduction

    def _reduce(self, loss: Tensor) -> Tensor:
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def forward(self, out: Tensor, **extras) -> Tensor:
        raise NotImplementedError
