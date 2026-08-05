from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cifp.engine.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_resume_restores_model_optimizer_step_and_rng(tmp_path: Path) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(torch.randn(2, 3)).sum()
    loss.backward()
    optimizer.step()

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=3, global_step=17, config={"seed": 42})
    expected_random = (random.random(), float(np.random.rand()), torch.rand(2))

    with torch.no_grad():
        model.weight.zero_()
    random.random()
    np.random.rand()
    torch.rand(4)

    state = load_checkpoint(path, model, optimizer)
    actual_random = (random.random(), float(np.random.rand()), torch.rand(2))

    assert state.epoch == 3
    assert state.global_step == 17
    assert torch.count_nonzero(model.weight) > 0
    assert actual_random[0] == expected_random[0]
    assert actual_random[1] == expected_random[1]
    assert torch.equal(actual_random[2], expected_random[2])


def test_checkpoint_loads_torch_version_metadata_safely(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    path = save_checkpoint(
        tmp_path / "version.pt",
        model,
        optimizer,
        epoch=0,
        global_step=1,
        config={},
        metadata={"torch_version": torch.__version__},
    )

    resumed = load_checkpoint(path, model)

    assert str(resumed.metadata["torch_version"]) == str(torch.__version__)
