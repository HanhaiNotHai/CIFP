from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cifp.analysis.io import load_analysis_features, require_arrays
from cifp.analysis.statistics import primitive_coverage_by_group


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seen/unknown primitive coverage and novelty")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--unknown-features", type=Path, required=True)
    parser.add_argument("--top-r", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    train = load_analysis_features(arguments.train_features)
    unknown = load_analysis_features(arguments.unknown_features)
    require_arrays(train, "usage")
    require_arrays(unknown, "usage", "sources")
    report = primitive_coverage_by_group(
        train["usage"],
        unknown["usage"],
        unknown["sources"],
        top_r=arguments.top_r,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
