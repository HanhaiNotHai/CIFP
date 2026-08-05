from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _ddp_requested() -> bool:
    return os.environ.get("CIFP_RUN_DDP_SMOKE") == "1"


@pytest.mark.gpu
@pytest.mark.distributed
@pytest.mark.smoke
@pytest.mark.skipif(
    not _ddp_requested() or torch.cuda.device_count() < 2,
    reason="set CIFP_RUN_DDP_SMOKE=1 with at least two visible GPUs",
)
def test_two_gpu_ddp_synthetic_training(tmp_path: Path) -> None:
    """Opt-in NCCL smoke; normal CPU test runs skip this without failing."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "cifp.cli.train",
        "--config",
        "configs/protocol/forensynths_selfsynthesis.yaml",
        "--synthetic",
        "--max-steps",
        "1",
        "--workers",
        "0",
        "--device",
        "cuda",
        "--output",
        str(tmp_path),
    ]
    subprocess.run(command, check=True, timeout=120)
    assert (tmp_path / "checkpoints" / "last.pt").is_file()
