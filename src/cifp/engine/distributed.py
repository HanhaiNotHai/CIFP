from __future__ import annotations

import os
from collections.abc import Iterator, Sized
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as distributed
from torch.utils.data import Sampler


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(preferred_device: str = "auto") -> DistributedContext:
    """Initialize torchrun/NCCL when WORLD_SIZE is greater than one."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if preferred_device == "cpu":
            raise RuntimeError("native CIFP multi-process training requires NCCL/CUDA")
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL DDP requires CUDA, but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        if preferred_device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"unsupported device selection: {preferred_device}")
        if preferred_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested explicitly but is unavailable")
        use_cuda = preferred_device == "cuda" or (
            preferred_device == "auto" and torch.cuda.is_available()
        )
        device = torch.device("cuda", 0) if use_cuda else torch.device("cpu")
    return DistributedContext(rank, world_size, local_rank, device)


def cleanup_distributed() -> None:
    if distributed.is_available() and distributed.is_initialized():
        distributed.destroy_process_group()


def global_batch_info(
    *,
    per_gpu_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
    expected_global_batch_size: int,
) -> dict[str, int | bool]:
    actual = per_gpu_batch_size * world_size * gradient_accumulation_steps
    return {
        "actual_global_batch_size": actual,
        "non_protocol_batch": actual != expected_global_batch_size,
    }


class DistributedEvalSampler(Sampler[int]):
    """Partition evaluation indices without DistributedSampler padding duplicates."""

    def __init__(self, dataset: Sized, num_replicas: int, rank: int) -> None:
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed evaluation rank/world size")
        self.size = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.size, self.num_replicas))

    def __len__(self) -> int:
        if self.rank >= self.size:
            return 0
        return (self.size - 1 - self.rank) // self.num_replicas + 1


def gather_objects(value: Any) -> list[Any]:
    if not (distributed.is_available() and distributed.is_initialized()):
        return [value]
    gathered: list[Any] = [None] * distributed.get_world_size()
    distributed.all_gather_object(gathered, value)
    return gathered
