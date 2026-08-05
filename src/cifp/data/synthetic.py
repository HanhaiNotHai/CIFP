from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticImageDataset(Dataset[dict[str, object]]):
    """Deterministic random images for explicit end-to-end smoke tests."""

    def __init__(self, *, sample_count: int, environment_count: int, seed: int = 42) -> None:
        self.sample_count = sample_count
        self.environment_count = environment_count
        self.seed = seed
        self.labels = [index % 2 for index in range(sample_count)]
        self.environments = [index % environment_count for index in range(sample_count)]

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, object]:
        generator = torch.Generator().manual_seed(self.seed + index)
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "image": torch.randn(3, 128, 128, generator=generator),
            "label": torch.tensor(float(self.labels[index]), dtype=torch.float32),
            "content_env": torch.tensor(self.environments[index], dtype=torch.long),
            "path": f"synthetic://{index}",
            "source": "synthetic",
            "semantic_class": "synthetic",
            "generator": "synthetic" if self.labels[index] else "",
            "real_source": "synthetic" if not self.labels[index] else "",
        }
