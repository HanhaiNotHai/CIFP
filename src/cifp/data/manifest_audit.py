from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice, repeat
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from cifp.data.manifest import read_manifest, write_manifest

ISSUE_COLUMNS = ["row_index", "path", "issue_type", "error", "width", "height", "excluded"]


def _inspect_image(item: tuple[int, str], crop_size: int) -> dict[str, Any] | None:
    row_index, raw_path = item
    path = Path(raw_path).expanduser().resolve()
    try:
        size_bytes = path.stat().st_size
    except FileNotFoundError as error:
        return _issue(row_index, path, "missing", str(error), excluded=True)
    except OSError as error:
        return _issue(row_index, path, "corrupt", f"stat failed: {error}", excluded=True)
    if size_bytes == 0:
        return _issue(row_index, path, "empty", "file is empty (0 bytes)", excluded=True)

    width: int | None = None
    height: int | None = None
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except (OSError, SyntaxError, ValueError, EOFError, UnidentifiedImageError) as error:
        return _issue(
            row_index,
            path,
            "corrupt",
            str(error),
            width=width,
            height=height,
            excluded=True,
        )
    if width < crop_size or height < crop_size:
        return _issue(
            row_index,
            path,
            "small_image",
            f"image size {width}x{height} is smaller than crop size {crop_size}",
            width=width,
            height=height,
            excluded=False,
        )
    return None


def _issue(
    row_index: int,
    path: Path,
    issue_type: str,
    error: str,
    *,
    width: int | None = None,
    height: int | None = None,
    excluded: bool,
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "path": str(path),
        "issue_type": issue_type,
        "error": error,
        "width": width,
        "height": height,
        "excluded": excluded,
    }


def audit_manifest_file(
    source_path: str | Path,
    output_path: str | Path,
    *,
    issues_path: str | Path,
    summary_path: str | Path,
    crop_size: int = 128,
    workers: int = 8,
) -> dict[str, Any]:
    """Verify manifest images and write an audited, filtered manifest plus reports."""
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")

    source = Path(source_path).expanduser().resolve()
    destinations = [
        Path(output_path).expanduser().resolve(),
        Path(issues_path).expanduser().resolve(),
        Path(summary_path).expanduser().resolve(),
    ]
    if source in destinations:
        raise ValueError("source manifest and audit output paths must differ")
    if len(set(destinations)) != len(destinations):
        raise ValueError("audit output paths must be distinct")

    frame = read_manifest(source)
    items = enumerate(frame["path"].astype(str))
    issues = []
    with tqdm(total=len(frame), desc="audit images", unit="image") as progress:
        if workers == 1:
            inspected = map(_inspect_image, items, repeat(crop_size))
            for issue in inspected:
                progress.update()
                if issue is not None:
                    issues.append(issue)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                while batch := list(islice(items, workers * 32)):
                    inspected = executor.map(_inspect_image, batch, repeat(crop_size))
                    for issue in inspected:
                        progress.update()
                        if issue is not None:
                            issues.append(issue)

    excluded_indices = {int(issue["row_index"]) for issue in issues if issue["excluded"]}
    keep = [index not in excluded_indices for index in range(len(frame))]
    filtered = frame.loc[keep].reset_index(drop=True)
    write_manifest(filtered, destinations[0])

    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    destinations[1].parent.mkdir(parents=True, exist_ok=True)
    issue_frame.to_csv(destinations[1], index=False)

    counts = Counter(str(issue["issue_type"]) for issue in issues)
    summary = {
        "source_manifest": str(source),
        "output_manifest": str(destinations[0]),
        "issues_path": str(destinations[1]),
        "crop_size": crop_size,
        "input_count": len(frame),
        "output_count": len(filtered),
        "excluded_count": len(excluded_indices),
        "small_image_count": counts["small_image"],
        "issue_counts": dict(sorted(counts.items())),
    }
    destinations[2].parent.mkdir(parents=True, exist_ok=True)
    destinations[2].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
