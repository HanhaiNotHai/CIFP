from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cifp.config.loader import load_config
from cifp.data.manifest import read_manifest, write_manifest
from cifp.environments.clustering import (
    assign_fixed_random_environments,
    environment_audit,
    fit_content_environments,
)
from cifp.environments.store import FeatureMemmap


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit and assign CIFP content environments")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--feature-store", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--random", action="store_true", help="fixed random environment ablation")
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    protocol_name = str(config["protocol"]["name"])
    manifest_path = arguments.manifest or Path(config["protocol"]["train_manifest"])
    output_manifest = arguments.output_manifest or manifest_path
    artifact_directory = Path("artifacts/manifests") / protocol_name
    feature_directory = arguments.feature_store or artifact_directory / "semantic_features"
    environment_count = int(config["environment"]["count"])
    random_state = int(config["environment"]["random_state"])
    frame = read_manifest(manifest_path)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    if arguments.random:
        assigned_frame = assign_fixed_random_environments(
            frame, environment_count=environment_count, random_state=random_state
        )
        clusterer = None
        fit_indices = np.array([], dtype=np.int64)
    else:
        metadata = json.loads((feature_directory / "metadata.json").read_text(encoding="utf-8"))
        store = FeatureMemmap(
            feature_directory,
            row_count=len(frame),
            feature_dim=int(metadata["feature_dim"]),
        )
        pending = store.pending_indices()
        if len(pending):
            raise RuntimeError(
                f"semantic feature extraction is incomplete: {len(pending)} rows pending"
            )
        clusterer, assignments, fit_indices = fit_content_environments(
            store.features,
            frame,
            environment_count=environment_count,
            max_fit_samples=int(config["environment"]["max_fit_samples"]),
            random_state=random_state,
        )
        assigned_frame = frame.copy()
        assigned_frame["content_env"] = assignments
        with (artifact_directory / "clusterer.pkl").open("wb") as handle:
            pickle.dump(clusterer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_manifest(assigned_frame, output_manifest)
    np.save(artifact_directory / "cluster_fit_indices.npy", fit_indices)
    report = environment_audit(assigned_frame, environment_count=environment_count)
    (artifact_directory / "environment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (artifact_directory / "environment_config.json").write_text(
        json.dumps(
            {
                "protocol": protocol_name,
                "environment_count": environment_count,
                "max_fit_samples": int(config["environment"]["max_fit_samples"]),
                "random_state": random_state,
                "assignment": "fixed_random" if arguments.random else "MiniBatchKMeans",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"content environments written to {output_manifest.resolve()}")
    print(f"environment report: {(artifact_directory / 'environment_report.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
