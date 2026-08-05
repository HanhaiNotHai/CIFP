from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch process-local RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_metadata() -> dict[str, Any]:
    """Return checkpoint/run provenance without modifying Git or the environment."""
    gpu_names = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    status = _git(["status", "--porcelain"])
    try:
        cudnn_version: int | str | None = torch.backends.cudnn.version()
    except RuntimeError as error:
        cudnn_version = f"unavailable: {error}"
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(status and status != "unavailable"),
        "git_status": status.splitlines(),
        "uv_lock_sha256": _file_hash(Path("uv.lock")),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "gpu_names": gpu_names,
    }


def write_runtime_metadata(path: str | Path) -> dict[str, Any]:
    metadata = runtime_metadata()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def parameter_report(model: torch.nn.Module) -> dict[str, dict[str, int | float]]:
    """Count total/trainable parameters globally and by top-level module."""
    report: dict[str, dict[str, int | float]] = {}
    modules = {"total": model, **dict(model.named_children())}
    for name, module in modules.items():
        total = sum(parameter.numel() for parameter in module.parameters())
        trainable = sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        )
        report[name] = {
            "total": total,
            "trainable": trainable,
            "trainable_ratio": trainable / total if total else 0.0,
        }
    return report
