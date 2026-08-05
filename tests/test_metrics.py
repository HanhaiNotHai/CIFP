from __future__ import annotations

import pytest
import torch

from cifp.metrics.binary import binary_metrics, evaluate_by_source


def test_metrics_use_fake_probability_and_fixed_threshold() -> None:
    labels = [0, 1, 1, 0]
    probabilities = [0.1, 0.8, 0.4, 0.6]
    metrics = binary_metrics(labels, probabilities, threshold=0.5)

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["fpr"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]

    logits = torch.tensor([0.0])
    probability = torch.sigmoid(logits).item()
    assert probability == 0.5
    assert int(probability >= 0.5) == 1


def test_per_source_macro_and_overall_metrics() -> None:
    report = evaluate_by_source(
        labels=[0, 1, 0, 1],
        probabilities=[0.1, 0.9, 0.8, 0.2],
        sources=["a", "a", "b", "b"],
        threshold=0.5,
    )
    assert set(report["per_source"]) == {"a", "b"}
    assert report["macro"]["mAcc"] == pytest.approx(0.5)
    assert report["macro"]["mAP"] == pytest.approx(0.75)
    assert report["worst_source"]["accuracy"] == pytest.approx(0.0)
