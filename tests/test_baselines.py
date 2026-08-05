from __future__ import annotations

import torch
from conftest import FakeDINOv3

from cifp.models.backbone import ForensicBackbone
from cifp.models.baselines import CLSBaseline, PatchMeanBaseline
from cifp.models.cifp import CIFP
from cifp.models.factory import build_model


def test_patch_mean_and_cls_baselines_output_fake_logits() -> None:
    images = torch.randn(2, 3, 128, 128)
    patch_model = PatchMeanBaseline(
        ForensicBackbone(FakeDINOv3(), train_last_n_blocks=0), hidden_dim=16
    )
    cls_model = CLSBaseline(ForensicBackbone(FakeDINOv3(), train_last_n_blocks=0), hidden_dim=16)
    assert patch_model(images).shape == (2,)
    assert cls_model(images).shape == (2,)
    assert patch_model.inference(images).shape == (2,)
    assert cls_model.inference(images).shape == (2,)


def test_model_factory_applies_random_dictionary_dense_and_k1_options() -> None:
    base = {
        "kind": "cifp",
        "forensic_dim": 16,
        "primitive_count": 1,
        "temperature": 0.1,
        "top_k": None,
        "composition_dim": 12,
        "assignment": "dense",
        "random_fixed_dictionary": True,
        "cooccurrence": {"enabled": False, "output_dim": 8},
    }
    model = build_model(
        base,
        environment_count=4,
        backbone=ForensicBackbone(FakeDINOv3(), train_last_n_blocks=0),
    )
    assert isinstance(model, CIFP)
    assert model.primitive_dictionary.primitive_count == 1
    assert model.primitive_dictionary.top_k is None
    assert model.primitive_dictionary.dictionary.requires_grad is False
