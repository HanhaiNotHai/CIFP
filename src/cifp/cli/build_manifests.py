from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from cifp.config.loader import load_config
from cifp.data.builders import (
    build_forensynths_selfsynthesis,
    build_genimage_sd14,
    build_optional_ufd,
    save_protocol_manifests,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build CIFP protocol manifests")
    parser.add_argument(
        "--protocol",
        required=True,
        choices=["forensynths_selfsynthesis", "genimage_sd14", "optional_ufd"],
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--forensynths-root", type=Path, default=None)
    parser.add_argument("--self-synthesis-root", type=Path, default=None)
    parser.add_argument("--genimage-root", type=Path, default=None)
    parser.add_argument("--ufd-root", type=Path, default=None)
    arguments = parser.parse_args(argv)
    config_path = arguments.config or Path("configs/protocol") / f"{arguments.protocol}.yaml"
    config = load_config(config_path)
    data_config = config["data"]
    if arguments.protocol == "forensynths_selfsynthesis":
        manifests = build_forensynths_selfsynthesis(
            arguments.forensynths_root or Path(data_config["forensynths_root"]),
            arguments.self_synthesis_root or Path(data_config["self_synthesis_root"]),
            class_aliases=dict(data_config["semantic_class_aliases"]),
            source_aliases=dict(data_config["self_synthesis_source_aliases"]),
        )
    elif arguments.protocol == "genimage_sd14":
        manifests = build_genimage_sd14(
            arguments.genimage_root or Path(data_config["genimage_root"]),
            source_aliases=dict(data_config["source_aliases"]),
        )
    else:
        manifests = build_optional_ufd(
            arguments.ufd_root or Path(data_config["ufd_root"]),
            source_aliases=dict(data_config["source_aliases"]),
        )
    paths = save_protocol_manifests(manifests, arguments.output_root / arguments.protocol)
    for split, path in paths.items():
        print(f"{split}: {len(manifests[split])} rows -> {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
