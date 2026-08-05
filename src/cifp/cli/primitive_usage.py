from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cifp.analysis.io import load_analysis_features, require_arrays
from cifp.analysis.statistics import primitive_usage_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report CIFP primitive usage")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    features = load_analysis_features(arguments.features)
    require_arrays(features, "usage", "labels", "generators", "content_env")
    report = primitive_usage_report(
        features["usage"],
        labels=features["labels"],
        generators=features["generators"],
        content_env=features["content_env"],
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
