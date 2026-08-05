from __future__ import annotations

import torch
from conftest import FakeDINOv3

from cifp.losses.total import CIFPLoss, LossWeights
from cifp.models.backbone import ForensicBackbone
from cifp.models.cifp import CIFP


def test_synthetic_end_to_end_forward_losses_backward_step_and_inference() -> None:
    model = CIFP(
        ForensicBackbone(FakeDINOv3(), train_last_n_blocks=1),
        forensic_dim=16,
        primitive_count=8,
        top_k=3,
        composition_dim=12,
        environment_count=4,
    )
    criterion = CIFPLoss(LossWeights())
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    images = torch.randn(2, 3, 128, 128)
    labels = torch.tensor([0.0, 1.0])
    content_env = torch.tensor([1, 3])

    output = model(images, grl_lambda=0.5)
    losses = criterion(output, labels, content_env, model.primitive_dictionary.normalized())
    losses["total"].backward()
    optimizer.step()

    assert set(losses) == {
        "total",
        "detection",
        "composition",
        "sparse",
        "balance",
        "diversity",
        "nuisance",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    model.eval()
    with torch.inference_mode():
        logits = model.inference(images)
    assert logits.shape == (2,)
