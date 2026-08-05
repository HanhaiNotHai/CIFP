from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cifp.models.adversarial import EnvironmentClassifier
from cifp.models.backbone import ForensicBackbone
from cifp.models.composition import CompositionPooler, FakeClassifier
from cifp.models.primitives import LocalForensicProjector, SparsePrimitiveDictionary


@dataclass(frozen=True)
class CIFPOutput:
    """Training outputs; no reconstruction, residual, or anomaly representation exists."""

    fake_logits: torch.Tensor
    z_for: torch.Tensor
    assignments: torch.Tensor
    primitive_usage: torch.Tensor
    grid_size: tuple[int, int]
    cooccurrence: torch.Tensor | None
    environment_logits: torch.Tensor | None


class CIFP(nn.Module):
    """Content-invariant compositional forensic primitive detector."""

    def __init__(
        self,
        backbone: ForensicBackbone,
        *,
        forensic_dim: int = 256,
        primitive_count: int = 32,
        temperature: float = 0.1,
        top_k: int | None = 4,
        composition_dim: int = 128,
        environment_count: int | None = 100,
        random_fixed_dictionary: bool = False,
        cooccurrence_enabled: bool = False,
        cooccurrence_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = LocalForensicProjector(backbone.hidden_size, forensic_dim)
        self.primitive_dictionary = SparsePrimitiveDictionary(
            primitive_count=primitive_count,
            dim=forensic_dim,
            temperature=temperature,
            top_k=top_k,
            fixed=random_fixed_dictionary,
        )
        self.composition_pooler = CompositionPooler(
            primitive_count=primitive_count,
            output_dim=composition_dim,
            cooccurrence_enabled=cooccurrence_enabled,
            cooccurrence_dim=cooccurrence_dim,
        )
        self.classifier = FakeClassifier(composition_dim)
        self.environment_classifier = (
            EnvironmentClassifier(composition_dim, environment_count)
            if environment_count is not None
            else None
        )

    def forward(self, images: torch.Tensor, *, grl_lambda: float | None = None) -> CIFPOutput:
        patch_tokens, grid_size, _cls_token = self.backbone(images)
        local_features = self.projector(patch_tokens)
        assignments, _dictionary = self.primitive_dictionary(local_features)
        z_for, usage, relation = self.composition_pooler(assignments, grid_size=grid_size)
        fake_logits = self.classifier(z_for)
        environment_logits = None
        if grl_lambda is not None:
            if self.environment_classifier is None:
                raise RuntimeError("content environment classifier is disabled")
            environment_logits = self.environment_classifier(z_for, grl_lambda)
        return CIFPOutput(
            fake_logits=fake_logits,
            z_for=z_for,
            assignments=assignments,
            primitive_usage=usage,
            grid_size=grid_size,
            cooccurrence=relation,
            environment_logits=environment_logits,
        )

    def inference(self, images: torch.Tensor) -> torch.Tensor:
        """Return fake logits without evaluating the training-only environment head."""
        return self(images, grl_lambda=None).fake_logits

    def classify_assignments(
        self, assignments: torch.Tensor, *, grid_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reclassify controlled primitive activations for masking analysis."""
        z_for, _usage, _relation = self.composition_pooler(assignments, grid_size=grid_size)
        return self.classifier(z_for), z_for
