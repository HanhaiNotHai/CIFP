from __future__ import annotations

from pathlib import Path

from PIL import Image

from cifp.data.audit import audit_dataset_root, render_dataset_audit


def test_dataset_audit_counts_extensions_small_and_corrupt_images(tmp_path: Path) -> None:
    Image.new("RGB", (128, 128)).save(tmp_path / "good.png")
    Image.new("RGB", (64, 128)).save(tmp_path / "small.jpg")
    (tmp_path / "broken.webp").write_bytes(b"broken")
    (tmp_path / "train" / "0_real").mkdir(parents=True)
    (tmp_path / "train" / "chair").mkdir()
    (tmp_path / "train" / "airplane").mkdir()

    report = audit_dataset_root(tmp_path, crop_size=128, max_depth=2)
    markdown = render_dataset_audit({"synthetic": report})

    assert report["image_count"] == 3
    assert report["extension_counts"] == {".jpg": 1, ".png": 1, ".webp": 1}
    assert report["small_image_count"] == 1
    assert report["corrupt_image_count"] == 1
    assert str((tmp_path / "broken.webp").resolve()) in report["corrupt_images"]
    assert "train" in report["split_candidates"]
    assert report["label_candidates"] == ["0_real"]
    assert report["semantic_class_candidates"] == ["airplane", "chair"]
    assert "# CIFP dataset audit" in markdown
