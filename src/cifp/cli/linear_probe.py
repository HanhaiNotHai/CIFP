from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from cifp.analysis.io import load_analysis_features, require_arrays


def _probe(z_for: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    values = np.asarray(target).astype(str)
    valid = values != ""
    features = z_for[valid]
    values = values[valid]
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(values)
    counts = np.bincount(encoded)
    if len(encoder.classes_) < 2 or len(encoded) < 10 or counts.min() < 2:
        return {
            "status": "insufficient_data",
            "samples": int(len(encoded)),
            "classes": encoder.classes_.tolist(),
        }
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        encoded,
        test_size=0.2,
        random_state=42,
        stratify=encoded,
    )
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    classifier.fit(train_x, train_y)
    predictions = classifier.predict(test_x)
    return {
        "status": "ok",
        "samples": int(len(encoded)),
        "classes": encoder.classes_.tolist(),
        "test_accuracy": float(accuracy_score(test_y, predictions)),
        "random_state": 42,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Linear probes on CIFP z_for")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    features = load_analysis_features(arguments.features)
    require_arrays(
        features,
        "z_for",
        "labels",
        "semantic_classes",
        "content_env",
        "generators",
        "real_sources",
    )
    generators = features["generators"].astype(str)
    generators[generators == ""] = "real"
    real_sources = features["real_sources"].astype(str)
    real_sources[real_sources == ""] = "generated"
    targets = {
        "true_fake": features["labels"],
        "semantic_class": features["semantic_classes"],
        "content_env": features["content_env"],
        "generator": generators,
        "real_source": real_sources,
    }
    report = {name: _probe(features["z_for"], target) for name, target in targets.items()}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
