from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from conftest import FakeDINOv3

from cifp.models.backbone import ForensicBackbone, split_dinov3_tokens


def test_patch_token_extraction_removes_cls_and_register_tokens() -> None:
    config = SimpleNamespace(hidden_size=3, patch_size=16, num_register_tokens=4)
    sequence = torch.arange(2 * 69 * 3, dtype=torch.float32).reshape(2, 69, 3)

    cls_token, patches, grid = split_dinov3_tokens(sequence, (128, 128), config)

    assert torch.equal(cls_token, sequence[:, 0])
    assert torch.equal(patches, sequence[:, 5:])
    assert patches.shape == (2, 64, 3)
    assert grid == (8, 8)


def test_patch_count_is_derived_from_runtime_shape_and_config() -> None:
    config = SimpleNamespace(hidden_size=5, patch_size=8, num_register_tokens=2)
    sequence = torch.randn(1, 1 + 2 + 60, 5)

    _, patches, grid = split_dinov3_tokens(sequence, (48, 80), config)

    assert patches.shape[1] == 60
    assert grid == (6, 10)


def test_patch_extraction_rejects_incompatible_shapes() -> None:
    config = SimpleNamespace(hidden_size=5, patch_size=16, num_register_tokens=4)
    with pytest.raises(ValueError, match="divisible"):
        split_dinov3_tokens(torch.randn(1, 10, 5), (127, 128), config)


def test_forensic_backbone_trains_only_requested_tail_blocks() -> None:
    raw = FakeDINOv3(num_hidden_layers=4)
    backbone = ForensicBackbone(raw, train_last_n_blocks=2, train_norm=False)

    assert not any(p.requires_grad for p in raw.model.layer[0].parameters())
    assert not any(p.requires_grad for p in raw.model.layer[1].parameters())
    assert all(p.requires_grad for p in raw.model.layer[2].parameters())
    assert all(p.requires_grad for p in raw.model.layer[3].parameters())
    assert not any(p.requires_grad for p in raw.norm.parameters())

    patches, grid, cls = backbone(torch.randn(2, 3, 128, 128))
    assert patches.shape == (2, 64, raw.config.hidden_size)
    assert grid == (8, 8)
    assert cls.shape == (2, raw.config.hidden_size)
