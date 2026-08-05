from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from cifp.data.dataset import CorruptImageError, ManifestImageDataset
from cifp.data.manifest import MANIFEST_COLUMNS, read_manifest, validate_manifest, write_manifest
from cifp.data.transforms import ProtocolTransform, SmallImageError


def _row(path: Path, *, label: int, split: str) -> dict[str, object]:
    return {
        "path": str(path),
        "label": label,
        "split": split,
        "source": "source-a",
        "generator": "ProGAN" if label else "",
        "semantic_class": "cat",
        "real_source": "LSUN" if not label else "",
        "protocol": "test",
        "content_env": 2,
    }


def test_dataset_manifest_preserves_label_source_and_split(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (128, 128)).save(image_path)
    manifest_path = tmp_path / "manifest.parquet"
    write_manifest(pd.DataFrame([_row(image_path, label=1, split="train")]), manifest_path)

    frame = read_manifest(manifest_path)
    assert list(frame.columns) == MANIFEST_COLUMNS
    assert frame.loc[0, "label"] == 1
    assert frame.loc[0, "source"] == "source-a"
    assert frame.loc[0, "split"] == "train"


def test_manifest_rejects_path_crossing_train_and_test(tmp_path: Path) -> None:
    path = tmp_path / "same.png"
    frame = pd.DataFrame([_row(path, label=0, split="train"), _row(path, label=0, split="test")])
    with pytest.raises(ValueError, match="multiple splits"):
        validate_manifest(frame)


def test_protocol_transform_errors_on_small_image_with_full_path(tmp_path: Path) -> None:
    path = tmp_path / "tiny.png"
    image = Image.new("RGB", (127, 128))
    transform = ProtocolTransform(crop_size=128, training=False, small_image_policy="error")
    with pytest.raises(SmallImageError, match=str(path)):
        transform(image, path)


def test_protocol_transform_can_reflect_pad(tmp_path: Path) -> None:
    path = tmp_path / "tiny.png"
    tensor = ProtocolTransform(crop_size=128, training=False, small_image_policy="reflect_pad")(
        Image.new("RGB", (100, 120)), path
    )
    assert tensor.shape == (3, 128, 128)


def test_corrupt_image_error_contains_full_path(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    manifest = tmp_path / "manifest.parquet"
    write_manifest(pd.DataFrame([_row(path, label=0, split="train")]), manifest)
    dataset = ManifestImageDataset(
        manifest,
        transform=ProtocolTransform(crop_size=128, training=False),
    )
    with pytest.raises(CorruptImageError, match=str(path)):
        dataset[0]
