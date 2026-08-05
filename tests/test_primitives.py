from __future__ import annotations

import torch

from cifp.losses.composition import composition_regularizers
from cifp.models.primitives import LocalForensicProjector, SparsePrimitiveDictionary


def test_sparse_assignment_constraints_and_gradients() -> None:
    projector = LocalForensicProjector(12, 8)
    dictionary = SparsePrimitiveDictionary(primitive_count=7, dim=8, temperature=0.1, top_k=3)
    features = torch.randn(2, 5, 12, requires_grad=True)

    assignments, normalized_dictionary = dictionary(projector(features))

    assert assignments.shape == (2, 5, 7)
    assert torch.all(assignments >= 0)
    assert torch.allclose(assignments.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    assert torch.all((assignments > 0).sum(dim=-1) <= 3)
    weighted = assignments * torch.arange(7, dtype=assignments.dtype)
    weighted.sum().backward()
    assert projector.linear_in.weight.grad is not None
    assert torch.count_nonzero(projector.linear_in.weight.grad) > 0
    assert dictionary.dictionary.grad is not None
    assert torch.count_nonzero(dictionary.dictionary.grad) > 0
    assert torch.allclose(normalized_dictionary.norm(dim=-1), torch.ones(7), atol=1e-6)


def test_dense_assignment_keeps_all_primitives() -> None:
    module = SparsePrimitiveDictionary(primitive_count=4, dim=6, temperature=0.2, top_k=None)
    assignments, _ = module(torch.randn(1, 3, 6))
    assert torch.all((assignments > 0).sum(dim=-1) == 4)


def test_dictionary_regularizers_are_finite_and_detect_collapse() -> None:
    assignments = torch.softmax(torch.randn(3, 6, 4), dim=-1)
    orthogonal = torch.eye(4)
    collapsed = torch.ones(4, 4)

    orth_losses = composition_regularizers(assignments, orthogonal)
    collapsed_losses = composition_regularizers(assignments, collapsed)

    assert all(torch.isfinite(value) for value in orth_losses.values())
    assert all(torch.isfinite(value) for value in collapsed_losses.values())
    assert collapsed_losses["diversity"] > orth_losses["diversity"]
