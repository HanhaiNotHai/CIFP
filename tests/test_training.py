from __future__ import annotations

import torch
from conftest import FakeDINOv3
from torch.utils.data import DataLoader, Dataset

from cifp.engine.trainer import train_one_epoch
from cifp.losses.total import CIFPLoss, LossWeights
from cifp.models.backbone import ForensicBackbone
from cifp.models.cifp import CIFP
from cifp.utils.logging import RunLogger


class _TrainingDataset(Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, object]:
        generator = torch.Generator().manual_seed(index)
        return {
            "image": torch.randn(3, 128, 128, generator=generator),
            "label": torch.tensor(float(index % 2)),
            "content_env": torch.tensor(index % 3),
            "path": f"/{index}.png",
            "source": "synthetic",
        }


def _model() -> CIFP:
    return CIFP(
        ForensicBackbone(FakeDINOv3(), train_last_n_blocks=1),
        forensic_dim=16,
        primitive_count=8,
        top_k=3,
        composition_dim=12,
        environment_count=3,
    )


def test_train_one_epoch_runs_one_optimizer_step_and_logs_all_components(tmp_path) -> None:
    model = _model()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    before = model.classifier.linear.weight.detach().clone()
    logger = RunLogger(tmp_path, enabled=False)
    result = train_one_epoch(
        model,
        DataLoader(_TrainingDataset(), batch_size=2),
        CIFPLoss(LossWeights()),
        optimizer,
        device=torch.device("cpu"),
        epoch=0,
        start_global_step=0,
        precision="fp32",
        gradient_accumulation_steps=1,
        max_optimizer_steps=1,
        logger=logger,
    )
    assert result.global_step == 1
    assert not torch.equal(before, model.classifier.linear.weight)
    assert {
        "detection",
        "sparse",
        "balance",
        "diversity",
        "nuisance",
        "grl",
        "environment_accuracy",
        "effective_primitives",
        "activation_entropy",
        "sampler_fallbacks",
    } <= set(result.last_metrics)


def test_bf16_fails_explicitly_on_cpu(tmp_path) -> None:
    model = _model()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    try:
        train_one_epoch(
            model,
            DataLoader(_TrainingDataset(), batch_size=2),
            CIFPLoss(LossWeights()),
            optimizer,
            device=torch.device("cpu"),
            epoch=0,
            start_global_step=0,
            precision="bf16",
            gradient_accumulation_steps=1,
            max_optimizer_steps=1,
            logger=RunLogger(tmp_path, enabled=False),
        )
    except RuntimeError as error:
        assert "bf16" in str(error)
    else:
        raise AssertionError("bf16 CPU training must not silently change precision")
