from __future__ import annotations

import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, value: torch.Tensor, coefficient: float
    ) -> torch.Tensor:
        ctx.coefficient = coefficient
        return value.view_as(value)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return -ctx.coefficient * gradient, None


def gradient_reverse(value: torch.Tensor, coefficient: float) -> torch.Tensor:
    """Identity in the forward pass and `-coefficient` gradient in backward."""
    if coefficient < 0:
        raise ValueError("gradient reversal coefficient cannot be negative")
    return _GradientReversal.apply(value, coefficient)


def grl_coefficient(
    *, epoch: float, warmup_epochs: int = 5, ramp_end_epoch: int = 20, maximum: float = 1.0
) -> float:
    """Piecewise-linear CIFP gradient-reversal schedule."""
    if ramp_end_epoch <= warmup_epochs:
        raise ValueError("ramp_end_epoch must exceed warmup_epochs")
    if epoch <= warmup_epochs:
        return 0.0
    if epoch >= ramp_end_epoch:
        return maximum
    return maximum * (epoch - warmup_epochs) / (ramp_end_epoch - warmup_epochs)


class EnvironmentClassifier(nn.Module):
    """Training-only classifier for content pseudo-environments."""

    def __init__(self, input_dim: int, environment_count: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, environment_count),
        )

    def forward(self, z_for: torch.Tensor, coefficient: float) -> torch.Tensor:
        return self.network(gradient_reverse(z_for, coefficient))
