from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class EnvironmentBalancedBatchSampler(Sampler[list[int]]):
    """Deterministic DDP-aware batches balancing labels and cycling environments."""

    def __init__(
        self,
        labels: Sequence[int],
        environments: Sequence[int],
        *,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ) -> None:
        if len(labels) != len(environments):
            raise ValueError("labels and environments must have equal length")
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be within num_replicas")
        if set(labels) - {0, 1}:
            raise ValueError("labels must contain only real=0 and fake=1")
        self.labels = list(labels)
        self.environments = list(environments)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.fallback_count = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _ordered_label_indices(self, label: int, rng: random.Random) -> list[int]:
        groups: dict[int, list[int]] = defaultdict(list)
        for index, (sample_label, environment) in enumerate(
            zip(self.labels, self.environments, strict=True)
        ):
            if sample_label == label:
                groups[environment].append(index)
        for indices in groups.values():
            rng.shuffle(indices)
        environments = list(groups)
        rng.shuffle(environments)
        ordered: list[int] = []
        while environments:
            next_round: list[int] = []
            for environment in environments:
                indices = groups[environment]
                if indices:
                    ordered.append(indices.pop())
                if indices:
                    next_round.append(environment)
            environments = next_round
        return ordered

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        per_label = {
            label: self._ordered_label_indices(label, rng)[self.rank :: self.num_replicas]
            for label in (0, 1)
        }
        desired = {0: self.batch_size // 2, 1: self.batch_size - self.batch_size // 2}
        self.fallback_count = 0
        target_batches = len(self)
        for _ in range(target_batches):
            batch: list[int] = []
            for label in (0, 1):
                take = min(desired[label], len(per_label[label]))
                if take < desired[label]:
                    self.fallback_count += 1
                batch.extend(per_label[label][-take:] if take else [])
                if take:
                    del per_label[label][-take:]
            if len(batch) < self.batch_size:
                remainder = per_label[0] + per_label[1]
                rng.shuffle(remainder)
                needed = self.batch_size - len(batch)
                batch.extend(remainder[:needed])
                used = set(remainder[:needed])
                per_label = {
                    label: [index for index in indices if index not in used]
                    for label, indices in per_label.items()
                }
                self.fallback_count += 1
            if len(batch) != self.batch_size:
                break
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        samples_per_rank = len(self.labels) // self.num_replicas
        return samples_per_rank // self.batch_size
