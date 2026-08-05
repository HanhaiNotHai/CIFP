from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    labels: Sequence[int], probabilities: Sequence[float], *, threshold: float = 0.5
) -> dict[str, Any]:
    """Compute binary metrics with fake as the positive class."""
    targets = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if targets.shape != scores.shape or targets.ndim != 1 or targets.size == 0:
        raise ValueError("labels and probabilities must be non-empty equal-length vectors")
    if set(np.unique(targets)) - {0, 1}:
        raise ValueError("labels must be real=0 or fake=1")
    predictions = (scores >= threshold).astype(np.int64)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    negative_count = true_negative + false_positive
    auroc = float(roc_auc_score(targets, scores)) if np.unique(targets).size == 2 else float("nan")
    average_precision = float(average_precision_score(targets, scores))
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "average_precision": average_precision,
        "auroc": auroc,
        "fpr": float(false_positive / negative_count) if negative_count else float("nan"),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "count": int(targets.size),
        "real_count": int((targets == 0).sum()),
        "fake_count": int((targets == 1).sum()),
        "true_positive": int(true_positive),
        "false_negative": int(false_negative),
        "threshold": threshold,
    }


def evaluate_by_source(
    labels: Sequence[int],
    probabilities: Sequence[float],
    sources: Sequence[str],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute per-source, macro, worst-source, and sample-weighted overall metrics."""
    if not (len(labels) == len(probabilities) == len(sources)):
        raise ValueError("labels, probabilities, and sources must have equal length")
    per_source: dict[str, dict[str, Any]] = {}
    for source in sorted(set(sources)):
        indices = [index for index, value in enumerate(sources) if value == source]
        per_source[source] = binary_metrics(
            [labels[index] for index in indices],
            [probabilities[index] for index in indices],
            threshold=threshold,
        )
    accuracies = [metrics["accuracy"] for metrics in per_source.values()]
    average_precisions = [metrics["average_precision"] for metrics in per_source.values()]
    return {
        "per_source": per_source,
        "macro": {
            "mAcc": float(np.mean(accuracies)),
            "mAP": float(np.mean(average_precisions)),
        },
        "overall": binary_metrics(labels, probabilities, threshold=threshold),
        "worst_source": {
            "accuracy": float(min(accuracies)),
            "average_precision": float(min(average_precisions)),
        },
    }
