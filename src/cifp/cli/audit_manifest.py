from __future__ import annotations

import argparse
import json
from pathlib import Path

from cifp.data.manifest_audit import audit_manifest_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit image paths and build a filtered manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("artifacts/manifests/genimage_sd14_filtered/train.parquet"),
    )
    parser.add_argument("--issues", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.output_manifest
    summary = audit_manifest_file(
        args.manifest,
        destination,
        issues_path=args.issues or destination.parent / "image_issues.csv",
        summary_path=args.summary or destination.parent / "audit_summary.json",
        crop_size=args.crop_size,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
