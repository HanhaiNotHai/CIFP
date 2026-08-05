from __future__ import annotations

import numpy as np
import torch
from conftest import FakeDINOv3

from cifp.analysis.statistics import (
    primitive_coverage,
    primitive_coverage_by_group,
    primitive_mutual_information,
    primitive_usage_report,
)
from cifp.models.backbone import ForensicBackbone
from cifp.models.cifp import CIFP


def test_primitive_usage_report_groups_labels_generators_and_environments() -> None:
    usage = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    report = primitive_usage_report(
        usage,
        labels=np.array([0, 0, 1, 1]),
        generators=np.array(["real", "real", "g1", "g2"]),
        content_env=np.array([0, 1, 0, 1]),
    )
    assert report["overall_usage"] == [0.5, 0.5]
    assert set(report["by_label"]) == {"real", "fake"}
    assert set(report["by_generator"]) == {"real", "g1", "g2"}
    assert set(report["by_content_env"]) == {"0", "1"}
    assert report["effective_primitive_count"] == 2


def test_primitive_mutual_information_detects_correlated_activation() -> None:
    usage = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    scores = primitive_mutual_information(usage, np.array(["a", "a", "b", "b"]))
    assert scores.shape == (2,)
    assert np.all(scores > 0.5)


def test_primitive_coverage_reports_seen_primitives_and_novel_combinations() -> None:
    train = np.array([[0.9, 0.8, 0.1], [0.8, 0.1, 0.9]], dtype=np.float32)
    unknown = np.array([[0.8, 0.7, 0.1], [0.1, 0.8, 0.9]], dtype=np.float32)
    report = primitive_coverage(train, unknown, top_r=2)
    assert report["primitive_coverage"] == 1.0
    assert report["combination_novelty"] == 0.5

    grouped = primitive_coverage_by_group(
        train,
        unknown,
        np.array(["generator_a", "generator_b"]),
        top_r=2,
    )
    assert set(grouped["per_unknown_source"]) == {"generator_a", "generator_b"}
    assert grouped["aggregate"] == report


def test_cifp_can_reclassify_masked_assignments_without_backbone() -> None:
    model = CIFP(
        ForensicBackbone(FakeDINOv3(), train_last_n_blocks=0),
        forensic_dim=16,
        primitive_count=4,
        top_k=2,
        composition_dim=12,
        environment_count=3,
    )
    assignments = torch.softmax(torch.randn(2, 4, 4), dim=-1)
    logits, z_for = model.classify_assignments(assignments, grid_size=(2, 2))
    assert logits.shape == (2,)
    assert z_for.shape == (2, 12)
