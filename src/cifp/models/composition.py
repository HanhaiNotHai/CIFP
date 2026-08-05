from __future__ import annotations

import torch
from torch import nn


def symmetric_grid_cooccurrence(
    assignments: torch.Tensor, grid_size: tuple[int, int]
) -> torch.Tensor:
    """Average symmetric primitive cooccurrence over horizontal and vertical grid edges."""
    if assignments.ndim != 3:
        raise ValueError(f"assignments must have shape [B, N, K], got {assignments.shape}")
    grid_height, grid_width = grid_size
    if grid_height * grid_width != assignments.shape[1]:
        raise ValueError(
            f"grid {grid_size} contains {grid_height * grid_width} patches, "
            f"but assignments contains {assignments.shape[1]}"
        )
    grid = assignments.reshape(assignments.shape[0], grid_height, grid_width, assignments.shape[-1])
    starts: list[torch.Tensor] = []
    ends: list[torch.Tensor] = []
    if grid_width > 1:
        starts.append(grid[:, :, :-1].reshape(assignments.shape[0], -1, assignments.shape[-1]))
        ends.append(grid[:, :, 1:].reshape(assignments.shape[0], -1, assignments.shape[-1]))
    if grid_height > 1:
        starts.append(grid[:, :-1, :].reshape(assignments.shape[0], -1, assignments.shape[-1]))
        ends.append(grid[:, 1:, :].reshape(assignments.shape[0], -1, assignments.shape[-1]))
    if not starts:
        raise ValueError("cooccurrence requires a patch grid with at least one adjacent edge")
    edge_start = torch.cat(starts, dim=1)
    edge_end = torch.cat(ends, dim=1)
    forward = torch.einsum("bnk,bnl->bnkl", edge_start, edge_end)
    relation = 0.5 * (forward + forward.transpose(-1, -2))
    return relation.mean(dim=1)


class CompositionPooler(nn.Module):
    """Build the image-level composition representation from primitive activations."""

    def __init__(
        self,
        *,
        primitive_count: int,
        output_dim: int = 128,
        dropout: float = 0.1,
        cooccurrence_enabled: bool = False,
        cooccurrence_dim: int = 128,
    ) -> None:
        super().__init__()
        self.primitive_count = primitive_count
        self.cooccurrence_enabled = cooccurrence_enabled
        if cooccurrence_enabled:
            self.cooccurrence_projector: nn.Module | None = nn.Sequential(
                nn.Linear(primitive_count * primitive_count, cooccurrence_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.cooccurrence_projector = None
        input_dim = 2 * primitive_count + (cooccurrence_dim if cooccurrence_enabled else 0)
        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self, assignments: torch.Tensor, *, grid_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if assignments.shape[-1] != self.primitive_count:
            raise ValueError(
                f"assignment dimension {assignments.shape[-1]} != {self.primitive_count}"
            )
        usage = torch.cat([assignments.mean(dim=1), assignments.max(dim=1).values], dim=-1)
        relation: torch.Tensor | None = None
        representation = usage
        if self.cooccurrence_projector is not None:
            relation = symmetric_grid_cooccurrence(assignments, grid_size)
            cooccurrence = self.cooccurrence_projector(relation.flatten(1))
            representation = torch.cat([usage, cooccurrence], dim=-1)
        return self.projector(representation), usage, relation


class FakeClassifier(nn.Module):
    """Binary classifier whose sole input is the composition representation."""

    def __init__(self, input_dim: int = 128) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, z_for: torch.Tensor) -> torch.Tensor:
        if z_for.ndim != 2 or z_for.shape[-1] != self.linear.in_features:
            raise ValueError(
                f"z_for must have shape [B, {self.linear.in_features}], got {z_for.shape}"
            )
        return self.linear(z_for).squeeze(-1)
