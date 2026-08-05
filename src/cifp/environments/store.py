from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FeatureMemmap:
    """Float16 semantic feature store with row-aligned completion flags."""

    def __init__(self, directory: str | Path, *, row_count: int, feature_dim: int) -> None:
        if row_count <= 0 or feature_dim <= 0:
            raise ValueError("row_count and feature_dim must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.directory / "metadata.json"
        self.features_path = self.directory / "features.float16.mmap"
        self.completed_path = self.directory / "completed.uint8.mmap"
        expected = {"row_count": row_count, "feature_dim": feature_dim, "dtype": "float16"}
        if self.metadata_path.exists():
            actual = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError(
                    f"feature-store metadata mismatch: expected {expected}, got {actual}"
                )
            mode = "r+"
        else:
            self.metadata_path.write_text(
                json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            mode = "w+"
        self.features = np.memmap(
            self.features_path, dtype=np.float16, mode=mode, shape=(row_count, feature_dim)
        )
        self.completed = np.memmap(
            self.completed_path, dtype=np.uint8, mode=mode, shape=(row_count,)
        )
        if mode == "w+":
            self.completed[:] = 0
            self.completed.flush()

    def pending_indices(self) -> np.ndarray:
        return np.flatnonzero(self.completed == 0)

    def write(self, indices: np.ndarray, features: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        features = np.asarray(features)
        if features.shape != (len(indices), self.features.shape[1]):
            raise ValueError(
                f"features shape {features.shape} does not match "
                f"({len(indices)}, {self.features.shape[1]})"
            )
        if np.any(indices < 0) or np.any(indices >= self.features.shape[0]):
            raise IndexError("feature-store row index is out of range")
        self.features[indices] = features.astype(np.float16, copy=False)
        self.features.flush()
        self.completed[indices] = 1
        self.completed.flush()
