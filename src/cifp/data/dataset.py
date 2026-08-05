from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from cifp.data.manifest import read_manifest
from cifp.data.transforms import ProtocolTransform


class CorruptImageError(RuntimeError):
    """Raised when a manifest image cannot be decoded."""


class ManifestImageDataset(Dataset[dict[str, Any]]):
    """Strict manifest-only dataset with path-rich worker errors."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        transform: ProtocolTransform,
        split: str | None = None,
        corrupt_image_policy: Literal["error"] = "error",
    ) -> None:
        if corrupt_image_policy != "error":
            raise ValueError("CIFP currently permits only corrupt_image_policy='error'")
        frame = read_manifest(manifest_path)
        if split is not None:
            frame = frame[frame["split"] == split].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"manifest contains no samples for split={split!r}: {manifest_path}")
        self.records = frame.to_dict(orient="records")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = Path(str(record["path"])).expanduser()
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        except (FileNotFoundError, OSError, UnidentifiedImageError) as error:
            raise CorruptImageError(f"unable to decode image: {path.resolve()}: {error}") from error
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "image": self.transform(image, path),
            "label": torch.tensor(float(record["label"]), dtype=torch.float32),
            "content_env": torch.tensor(int(record["content_env"]), dtype=torch.long),
            "path": str(path.resolve()),
            "source": str(record["source"]),
            "semantic_class": str(record["semantic_class"]),
            "generator": str(record["generator"]),
            "real_source": str(record["real_source"]),
        }
