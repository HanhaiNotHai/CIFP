from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from cifp.models.backbone import ForensicBackbone


class _BackboneMLPBaseline(nn.Module, ABC):
    def __init__(self, backbone: ForensicBackbone, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(backbone.hidden_size),
            nn.Linear(backbone.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    @abstractmethod
    def _representation(self, images: torch.Tensor) -> torch.Tensor:
        """Return the concrete baseline's single image representation."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._representation(images)).squeeze(-1)

    def inference(self, images: torch.Tensor) -> torch.Tensor:
        return self(images)


class PatchMeanBaseline(_BackboneMLPBaseline):
    """DINOv3 patch-token mean pooling plus MLP baseline."""

    def _representation(self, images: torch.Tensor) -> torch.Tensor:
        patches, _grid, _cls = self.backbone(images)
        return patches.mean(dim=1)


class CLSBaseline(_BackboneMLPBaseline):
    """DINOv3 CLS-token semantic-shortcut diagnostic baseline."""

    def _representation(self, images: torch.Tensor) -> torch.Tensor:
        _patches, _grid, cls_token = self.backbone(images)
        return cls_token
