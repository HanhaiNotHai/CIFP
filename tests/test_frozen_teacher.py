from __future__ import annotations

import torch
from conftest import FakeDINOv3

from cifp.environments.teacher import FrozenSemanticTeacher
from cifp.models.backbone import ForensicBackbone
from cifp.models.cifp import CIFP


def test_frozen_teacher_has_no_grad_and_excludes_register_tokens() -> None:
    teacher = FrozenSemanticTeacher(FakeDINOv3(hidden_size=10))
    embeddings = teacher.extract(torch.randn(2, 3, 128, 128))

    assert embeddings.shape == (2, 20)
    assert not embeddings.requires_grad
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    assert torch.allclose(embeddings[:, :10].norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(embeddings[:, 10:].norm(dim=-1), torch.ones(2), atol=1e-5)


def test_cifp_contains_no_semantic_teacher_parameter() -> None:
    model = CIFP(
        ForensicBackbone(FakeDINOv3(), train_last_n_blocks=0),
        forensic_dim=16,
        primitive_count=8,
        top_k=3,
        composition_dim=12,
        environment_count=4,
    )
    assert all("semantic_teacher" not in name for name, _ in model.named_parameters())
