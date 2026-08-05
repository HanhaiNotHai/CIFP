from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from cifp.data.builders import IMAGE_EXTENSIONS


def _limited_tree(root: Path, max_depth: int) -> list[str]:
    entries: list[str] = []
    for directory, names, files in os.walk(root, followlinks=False):
        current = Path(directory)
        depth = len(current.relative_to(root).parts)
        if depth > max_depth:
            names[:] = []
            continue
        relative = "." if current == root else str(current.relative_to(root))
        entries.append(f"{relative}/")
        for name in sorted(files)[:5]:
            entries.append(f"{relative}/{name}")
        names[:] = sorted(names)
    return entries


def audit_dataset_root(
    root: str | Path,
    *,
    crop_size: int = 128,
    max_depth: int = 2,
    max_image_checks: int | None = None,
) -> dict[str, Any]:
    """Read-only streaming audit of one dataset root."""
    data_root = Path(root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {data_root}")
    extension_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    directory_names: set[str] = set()
    top_level_directories: set[str] = set()
    split_child_directories: set[str] = set()
    symlinks: list[str] = []
    corrupt_images: list[str] = []
    corrupt_image_errors: dict[str, str] = {}
    small_images: list[str] = []
    checked = 0
    image_count = 0
    linked_inodes: dict[tuple[int, int], str] = {}
    duplicate_paths: list[str] = []
    for directory, names, files in os.walk(data_root, followlinks=False):
        current = Path(directory)
        directory_names.update(names)
        if current == data_root:
            top_level_directories.update(names)
        if current.parent == data_root and current.name.lower() in {
            "train",
            "val",
            "validation",
            "test",
        }:
            split_child_directories.update(names)
        for name in names:
            path = current / name
            if path.is_symlink():
                symlinks.append(str(path))
        for name in files:
            path = current / name
            if path.is_symlink():
                symlinks.append(str(path))
            suffix = path.suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            image_count += 1
            extension_counts[suffix] += 1
            group_counts[str(current.relative_to(data_root))] += 1
            try:
                stat = path.stat()
                if stat.st_nlink > 1:
                    key = stat.st_dev, stat.st_ino
                    if key in linked_inodes:
                        duplicate_paths.append(str(path.resolve()))
                    else:
                        linked_inodes[key] = str(path.resolve())
            except OSError as error:
                resolved = str(path.resolve())
                corrupt_images.append(resolved)
                corrupt_image_errors[resolved] = f"stat failed: {error}"
                continue
            if max_image_checks is not None and checked >= max_image_checks:
                continue
            checked += 1
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    image.verify()
                if width < crop_size or height < crop_size:
                    small_images.append(str(path.resolve()))
            except (OSError, UnidentifiedImageError) as error:
                resolved = str(path.resolve())
                corrupt_images.append(resolved)
                corrupt_image_errors[resolved] = str(error)
    lower_names = {name.lower(): name for name in directory_names}
    split_candidates = sorted(
        original
        for lower, original in lower_names.items()
        if lower in {"train", "val", "validation", "test"}
    )
    label_candidates = sorted(
        original
        for lower, original in lower_names.items()
        if lower in {"real", "fake", "0_real", "1_fake", "nature", "ai"}
    )
    source_candidates = sorted(
        name
        for name in top_level_directories
        if name.lower()
        not in {candidate.lower() for candidate in split_candidates + label_candidates}
    )
    semantic_class_candidates = sorted(
        name
        for name in split_child_directories
        if name.lower() not in {candidate.lower() for candidate in label_candidates}
    )
    return {
        "root": str(data_root),
        "tree": _limited_tree(data_root, max_depth),
        "image_count": image_count,
        "checked_image_count": checked,
        "image_check_complete": checked == image_count,
        "extension_counts": dict(sorted(extension_counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
        "small_image_count": len(small_images),
        "small_images": small_images,
        "corrupt_image_count": len(corrupt_images),
        "corrupt_images": corrupt_images,
        "corrupt_image_errors": corrupt_image_errors,
        "duplicate_paths": sorted(set(duplicate_paths)),
        "symlinks": sorted(symlinks),
        "split_candidates": split_candidates,
        "label_candidates": label_candidates,
        "source_candidates": source_candidates,
        "semantic_class_candidates": semantic_class_candidates,
    }


def render_dataset_audit(reports: dict[str, dict[str, Any]]) -> str:
    """Render complete machine-derived audit results as Markdown."""
    lines = [
        "# CIFP dataset audit",
        "",
        "This report is generated read-only. No sample was deleted, moved, or filtered.",
        "",
    ]
    for name, report in reports.items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Root: `{report['root']}`",
                f"- Images: {report['image_count']:,}",
                f"- Image headers checked: {report['checked_image_count']:,} "
                f"(complete={report['image_check_complete']})",
                f"- Smaller than crop: {report['small_image_count']:,}",
                f"- Corrupt/unreadable: {report['corrupt_image_count']:,}",
                f"- Symlinks: {len(report['symlinks']):,}",
                f"- Duplicate hard-linked/resolved paths: {len(report['duplicate_paths']):,}",
                "",
                "### Bounded tree",
                "",
                "```text",
                *report["tree"],
                "```",
                "",
                "### Extension counts",
                "",
                "```json",
                json.dumps(report["extension_counts"], indent=2, ensure_ascii=False),
                "```",
                "",
                "### Per-group counts",
                "",
                "```json",
                json.dumps(report["group_counts"], indent=2, ensure_ascii=False),
                "```",
                "",
                "### Candidates",
                "",
                f"- Splits: {report['split_candidates']}",
                f"- Real/fake labels: {report['label_candidates']}",
                f"- Generator/source candidates: {report['source_candidates']}",
                f"- Semantic class candidates: {report['semantic_class_candidates']}",
                "",
                "### Problem paths",
                "",
                "```json",
                json.dumps(
                    {
                        "small_images": report["small_images"],
                        "corrupt_images": report["corrupt_images"],
                        "corrupt_image_errors": report["corrupt_image_errors"],
                        "duplicate_paths": report["duplicate_paths"],
                        "symlinks": report["symlinks"],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "```",
                "",
            ]
        )
    return "\n".join(lines)
