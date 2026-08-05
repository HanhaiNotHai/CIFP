from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from cifp.data.audit import audit_dataset_root, render_dataset_audit

DEFAULT_ROOTS = {
    "ForenSynths": Path("/data/zhy/CNNDetection/dataset"),
    "GenImage": Path("/data/zhy/GenImage"),
    "Self-Synthesis": Path("/data/zhy/GANGen-Detection"),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only audit of CIFP external datasets")
    parser.add_argument("--output", type=Path, default=Path("docs/dataset_audit.md"))
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument(
        "--max-image-checks",
        type=int,
        default=None,
        help="limit decoded headers per root; default checks every image",
    )
    parser.add_argument("--forensynths-root", type=Path, default=DEFAULT_ROOTS["ForenSynths"])
    parser.add_argument("--genimage-root", type=Path, default=DEFAULT_ROOTS["GenImage"])
    parser.add_argument("--self-synthesis-root", type=Path, default=DEFAULT_ROOTS["Self-Synthesis"])
    arguments = parser.parse_args(argv)
    roots = {
        "ForenSynths": arguments.forensynths_root,
        "GenImage": arguments.genimage_root,
        "Self-Synthesis": arguments.self_synthesis_root,
    }
    reports = {
        name: audit_dataset_root(
            root,
            max_depth=arguments.max_depth,
            max_image_checks=arguments.max_image_checks,
        )
        for name, root in roots.items()
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(render_dataset_audit(reports), encoding="utf-8")
    print(f"dataset audit written to {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
