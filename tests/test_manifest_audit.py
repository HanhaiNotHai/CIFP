from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from cifp.data import manifest_audit
from cifp.data.dataset import ManifestImageDataset
from cifp.data.manifest import read_manifest, write_manifest
from cifp.data.manifest_audit import audit_manifest_file
from cifp.data.transforms import ProtocolTransform


def _row(path: Path, index: int) -> dict[str, object]:
    return {
        "path": str(path),
        "label": index % 2,
        "split": "train",
        "source": "SDv1.4",
        "generator": "SDv1.4" if index % 2 else "",
        "semantic_class": f"class-{index}",
        "real_source": "ImageNet" if index % 2 == 0 else "",
        "protocol": "genimage_sd14",
        "content_env": -1,
    }


def test_manifest_audit_filters_unreadable_files_and_retains_small_images(
    tmp_path: Path,
) -> None:
    normal = tmp_path / "normal.png"
    small = tmp_path / "small.png"
    empty = tmp_path / "empty.png"
    corrupt = tmp_path / "corrupt.png"
    missing = tmp_path / "missing.png"
    Image.new("RGB", (128, 128)).save(normal)
    Image.new("RGB", (64, 127)).save(small)
    empty.touch()
    corrupt.write_bytes(b"not an image")

    source = tmp_path / "source.parquet"
    paths = (normal, small, empty, corrupt, missing)
    rows = [_row(path, index) for index, path in enumerate(paths)]
    write_manifest(pd.DataFrame(rows), source)
    output = tmp_path / "filtered" / "train.parquet"
    issues = output.parent / "image_issues.csv"
    summary = output.parent / "audit_summary.json"

    result = audit_manifest_file(
        source,
        output,
        issues_path=issues,
        summary_path=summary,
        crop_size=128,
        workers=2,
    )

    assert read_manifest(source)["path"].tolist() == [row["path"] for row in rows]
    filtered = read_manifest(output)
    assert filtered["path"].tolist() == [str(normal), str(small)]
    assert filtered["semantic_class"].tolist() == ["class-0", "class-1"]

    issue_frame = pd.read_csv(issues, keep_default_na=False)
    assert issue_frame["row_index"].tolist() == [1, 2, 3, 4]
    assert issue_frame["issue_type"].tolist() == [
        "small_image",
        "empty",
        "corrupt",
        "missing",
    ]
    assert issue_frame["excluded"].tolist() == [False, True, True, True]
    assert issue_frame.loc[0, ["width", "height"]].astype(float).tolist() == [64, 127]
    assert issue_frame["path"].tolist() == [
        str(small.resolve()),
        str(empty.resolve()),
        str(corrupt.resolve()),
        str(missing.resolve()),
    ]
    assert all(issue_frame["error"].astype(str).str.len() > 0)

    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    assert summary_payload == result
    assert summary_payload["input_count"] == 5
    assert summary_payload["output_count"] == 2
    assert summary_payload["excluded_count"] == 3
    assert summary_payload["small_image_count"] == 1
    assert summary_payload["issue_counts"] == {
        "corrupt": 1,
        "empty": 1,
        "missing": 1,
        "small_image": 1,
    }

    dataset = ManifestImageDataset(
        output,
        transform=ProtocolTransform(
            crop_size=128,
            training=False,
            small_image_policy="reflect_pad",
        ),
    )
    assert dataset[0]["image"].shape == (3, 128, 128)
    assert dataset[1]["image"].shape == (3, 128, 128)


def test_manifest_audit_refuses_to_overwrite_source(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (128, 128)).save(image)
    source = tmp_path / "source.parquet"
    write_manifest(pd.DataFrame([_row(image, 0)]), source)

    with pytest.raises(ValueError, match="must differ"):
        audit_manifest_file(
            source,
            source,
            issues_path=tmp_path / "issues.csv",
            summary_path=tmp_path / "summary.json",
        )


@pytest.mark.parametrize("workers", [1, 2])
def test_manifest_audit_reports_image_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    updates: list[int] = []
    settings: dict[str, object] = {}

    class RecordingProgress:
        def __init__(self, *, total: int, desc: str, unit: str) -> None:
            settings.update(total=total, desc=desc, unit=unit)

        def __enter__(self) -> RecordingProgress:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def update(self, count: int = 1) -> None:
            updates.append(count)

    monkeypatch.setattr(manifest_audit, "tqdm", RecordingProgress)
    rows = []
    for index in range(3):
        image = tmp_path / f"image-{index}.png"
        Image.new("RGB", (128, 128)).save(image)
        rows.append(_row(image, index))
    source = tmp_path / "source.parquet"
    write_manifest(pd.DataFrame(rows), source)

    audit_manifest_file(
        source,
        tmp_path / "filtered.parquet",
        issues_path=tmp_path / "issues.csv",
        summary_path=tmp_path / "summary.json",
        workers=workers,
    )

    assert settings == {"total": 3, "desc": "audit images", "unit": "image"}
    assert sum(updates) == 3
