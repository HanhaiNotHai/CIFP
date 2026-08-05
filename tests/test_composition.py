from __future__ import annotations

import inspect

import pytest
import torch

from cifp.models.composition import (
    CompositionPooler,
    FakeClassifier,
    symmetric_grid_cooccurrence,
)


def test_compositional_pooler_mean_max_shape() -> None:
    pooler = CompositionPooler(primitive_count=5, output_dim=13)
    assignments = torch.softmax(torch.randn(3, 12, 5), dim=-1)

    z_for, usage, relation = pooler(assignments, grid_size=(3, 4))

    expected = torch.cat([assignments.mean(dim=1), assignments.max(dim=1).values], dim=-1)
    assert usage.shape == (3, 10)
    assert torch.allclose(usage, expected)
    assert z_for.shape == (3, 13)
    assert relation is None


def test_classifier_forward_accepts_only_z_for() -> None:
    classifier = FakeClassifier(input_dim=8)
    assert list(inspect.signature(classifier.forward).parameters) == ["z_for"]
    assert classifier(torch.randn(4, 8)).shape == (4,)
    with pytest.raises(ValueError, match="z_for"):
        classifier(torch.randn(4, 7))


def test_cooccurrence_dynamic_grid_is_symmetric_and_normalized() -> None:
    assignments = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]])

    relation = symmetric_grid_cooccurrence(assignments, grid_size=(2, 2))

    assert relation.shape == (1, 2, 2)
    assert torch.allclose(relation, relation.transpose(1, 2))
    assert torch.allclose(relation.sum(dim=(1, 2)), torch.ones(1))
    assert torch.allclose(relation[0], torch.tensor([[0.0, 0.5], [0.5, 0.0]]))


def test_cooccurrence_pooler_supports_non_square_grid() -> None:
    pooler = CompositionPooler(
        primitive_count=3,
        output_dim=7,
        cooccurrence_enabled=True,
        cooccurrence_dim=5,
    )
    assignments = torch.softmax(torch.randn(2, 6, 3), dim=-1)
    z_for, usage, relation = pooler(assignments, grid_size=(2, 3))
    assert z_for.shape == (2, 7)
    assert usage.shape == (2, 6)
    assert relation is not None and relation.shape == (2, 3, 3)
