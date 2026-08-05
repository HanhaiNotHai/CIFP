from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class LocalForensicProjector(nn.Module):
    """Project DINOv3 patch tokens to normalized local forensic features."""

    def __init__(self, input_dim: int, output_dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear_in = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        self.linear_out = nn.Linear(output_dim, output_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        projected = self.linear_out(self.activation(self.linear_in(self.norm(patch_tokens))))
        return functional.normalize(projected, dim=-1)


class SparsePrimitiveDictionary(nn.Module):
    """Learnable normalized forensic dictionary with masked top-k soft assignments."""

    def __init__(
        self,
        *,
        primitive_count: int = 32,
        dim: int = 256,
        temperature: float = 0.1,
        top_k: int | None = 4,
        fixed: bool = False,
    ) -> None:
        super().__init__()
        if primitive_count < 1:
            raise ValueError("primitive_count must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and not 1 <= top_k <= primitive_count:
            raise ValueError("top_k must be between 1 and primitive_count")
        self.primitive_count = primitive_count
        self.temperature = temperature
        self.top_k = top_k
        self.dictionary = nn.Parameter(torch.empty(primitive_count, dim), requires_grad=not fixed)
        nn.init.normal_(self.dictionary, std=dim**-0.5)

    def normalized(self) -> torch.Tensor:
        return functional.normalize(self.dictionary, dim=-1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.dictionary.shape[-1]:
            raise ValueError(
                "features must have shape "
                f"[B, N, {self.dictionary.shape[-1]}], got {features.shape}"
            )
        normalized_dictionary = self.normalized()
        logits = features @ normalized_dictionary.transpose(0, 1) / self.temperature
        if self.top_k is not None and self.top_k < self.primitive_count:
            top_values, top_indices = logits.topk(self.top_k, dim=-1)
            masked = torch.full_like(logits, -torch.inf)
            logits = masked.scatter(-1, top_indices, top_values)
        assignments = torch.softmax(logits, dim=-1)
        return assignments, normalized_dictionary
