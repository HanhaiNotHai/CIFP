from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from cifp.analysis.io import load_analysis_features, require_arrays
from cifp.config.loader import load_config
from cifp.engine.checkpoint import load_checkpoint
from cifp.metrics.binary import evaluate_by_source
from cifp.models.cifp import CIFP
from cifp.models.factory import build_model


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure performance after primitive masking")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    arguments = parser.parse_args(argv)
    features = load_analysis_features(arguments.features)
    require_arrays(
        features,
        "assignments",
        "usage",
        "labels",
        "sources",
        "fake_logits",
        "grid_size",
    )
    config = load_config(arguments.config)
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA primitive masking requested but unavailable")
    model = build_model(config["model"], environment_count=int(config["environment"]["count"])).to(
        device
    )
    if not isinstance(model, CIFP):
        raise TypeError("primitive masking requires a CIFP checkpoint")
    load_checkpoint(arguments.checkpoint, model)
    model.eval()
    labels = features["labels"].astype(int).tolist()
    sources = features["sources"].astype(str).tolist()
    threshold = float(config["evaluation"]["threshold"])
    baseline_scores = torch.sigmoid(torch.from_numpy(features["fake_logits"])).tolist()
    baseline = evaluate_by_source(labels, baseline_scores, sources, threshold=threshold)
    assignments = features["assignments"]
    primitive_count = assignments.shape[-1]
    if not 1 <= arguments.mask_count <= primitive_count:
        raise ValueError("mask-count must be within the primitive count")
    grid_size = tuple(int(value) for value in features["grid_size"].tolist())

    def evaluate_mask(masked_primitives: list[int]) -> dict[str, object]:
        scores: list[float] = []
        for start in range(0, len(assignments), arguments.batch_size):
            values = torch.from_numpy(
                assignments[start : start + arguments.batch_size].astype(np.float32)
            ).to(device)
            values[:, :, masked_primitives] = 0
            totals = values.sum(dim=-1, keepdim=True)
            values = torch.where(totals > 0, values / totals.clamp_min(1e-12), values)
            logits, _z_for = model.classify_assignments(values, grid_size=grid_size)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
        metrics = evaluate_by_source(labels, scores, sources, threshold=threshold)
        return {
            "masked_primitives": masked_primitives,
            "metrics": metrics,
            "accuracy_drop": baseline["overall"]["accuracy"] - metrics["overall"]["accuracy"],
            "average_precision_drop": baseline["overall"]["average_precision"]
            - metrics["overall"]["average_precision"],
        }

    rng = np.random.default_rng(42)
    random_mask = sorted(
        rng.choice(primitive_count, size=arguments.mask_count, replace=False).tolist()
    )
    top_mask = np.argsort(features["usage"].mean(axis=0))[-arguments.mask_count :].tolist()
    report = {
        "baseline": baseline,
        "individual": [evaluate_mask([primitive]) for primitive in range(primitive_count)],
        "random": evaluate_mask(random_mask),
        "top_used": evaluate_mask(sorted(top_mask)),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
