from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.torch_version import TorchVersion


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    global_step: int
    config: dict[str, Any]
    metadata: dict[str, Any]


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    module = getattr(model, "module", model)
    return module.state_dict()


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=False),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically save train state and deterministic RNG state."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": _model_state(model),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "metadata": metadata or {},
        "rng": _rng_state(),
    }
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> ResumeState:
    """Load trusted CIFP state on CPU, then restore model, optimizer, and RNG."""
    checkpoint_path = Path(path)
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    module = getattr(model, "module", model)
    module.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    _restore_rng_state(payload["rng"])
    return ResumeState(
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        config=dict(payload["config"]),
        metadata=dict(payload["metadata"]),
    )
