from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional

from cifp.losses.composition import composition_regularizers
from cifp.models.cifp import CIFPOutput


@dataclass(frozen=True)
class LossWeights:
    lambda_comp: float = 0.1
    lambda_nui: float = 0.1
    sparse: float = 0.1
    balance: float = 1.0
    diversity: float = 0.1
    pos_weight: float | None = None


class CIFPLoss(nn.Module):
    """CIFP detection, composition, and content-environment objective."""

    def __init__(self, weights: LossWeights) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        output: CIFPOutput,
        labels: torch.Tensor,
        content_env: torch.Tensor | None,
        normalized_dictionary: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pos_weight = (
            output.fake_logits.new_tensor(self.weights.pos_weight)
            if self.weights.pos_weight is not None
            else None
        )
        detection = functional.binary_cross_entropy_with_logits(
            output.fake_logits, labels.to(output.fake_logits.dtype), pos_weight=pos_weight
        )
        regularizers = composition_regularizers(output.assignments, normalized_dictionary)
        composition = (
            self.weights.sparse * regularizers["sparse"]
            + self.weights.balance * regularizers["balance"]
            + self.weights.diversity * regularizers["diversity"]
        )
        if output.environment_logits is None:
            nuisance = detection.new_zeros(())
        else:
            if content_env is None:
                raise ValueError("content_env is required when environment logits are present")
            nuisance = functional.cross_entropy(output.environment_logits, content_env.long())
        total = (
            detection + self.weights.lambda_comp * composition + self.weights.lambda_nui * nuisance
        )
        return {
            "total": total,
            "detection": detection,
            "composition": composition,
            **regularizers,
            "nuisance": nuisance,
        }
