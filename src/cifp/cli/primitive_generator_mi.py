from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cifp.analysis.io import load_analysis_features, require_arrays
from cifp.analysis.statistics import primitive_mutual_information


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Primitive activation/generator mutual information"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    features = load_analysis_features(arguments.features)
    require_arrays(features, "usage", "generators")
    generators = features["generators"].astype(str)
    generators[generators == ""] = "real"
    scores = primitive_mutual_information(features["usage"], generators)
    report = {"per_primitive_mi": scores.tolist(), "mean_mi": float(scores.mean())}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
