from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class FakeDINOv3(nn.Module):
    """Small differentiable DINOv3-shaped test double."""

    def __init__(
        self,
        *,
        hidden_size: int = 24,
        patch_size: int = 16,
        num_register_tokens: int = 4,
        num_hidden_layers: int = 4,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            num_hidden_layers=num_hidden_layers,
        )
        self.patch_embed = nn.Conv2d(3, hidden_size, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, hidden_size))
        encoder = nn.Module()
        encoder.layer = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU())
                for _ in range(num_hidden_layers)
            ]
        )
        self.model = encoder
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        patches = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        batch_size = pixel_values.shape[0]
        tokens = torch.cat(
            [
                self.cls_token.expand(batch_size, -1, -1),
                self.register_tokens.expand(batch_size, -1, -1),
                patches,
            ],
            dim=1,
        )
        for layer in self.model.layer:
            tokens = tokens + layer(tokens)
        return SimpleNamespace(last_hidden_state=self.norm(tokens))
