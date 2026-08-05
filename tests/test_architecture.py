from __future__ import annotations

import torch
from conftest import FakeDINOv3

from cifp.models.backbone import ForensicBackbone
from cifp.models.cifp import CIFP


def _model() -> CIFP:
    return CIFP(
        ForensicBackbone(FakeDINOv3(), train_last_n_blocks=1),
        forensic_dim=16,
        primitive_count=8,
        top_k=3,
        composition_dim=12,
        environment_count=4,
    )


def test_classifier_access_and_no_reconstruction_path() -> None:
    output = _model()(torch.randn(2, 3, 128, 128), grl_lambda=0.2)
    forbidden = {
        "reconstructed_image",
        "predicted_normal_feature",
        "residual_feature",
        "anomaly_score",
        "reconstruction_loss",
    }
    assert forbidden.isdisjoint(output.__dataclass_fields__)
    assert output.fake_logits.shape == (2,)
    assert output.environment_logits is not None


def test_inference_returns_only_fake_logits() -> None:
    model = _model().eval()
    with torch.inference_mode():
        logits = model.inference(torch.randn(2, 3, 128, 128))
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2,)
