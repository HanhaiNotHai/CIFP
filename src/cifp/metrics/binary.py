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


def select_accuracy_optimal_threshold(
    labels: Sequence[int], probabilities: Sequence[float]
) -> float:
    """Select the global accuracy-optimal threshold with a deterministic tie-break."""
    targets = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if targets.shape != scores.shape or targets.ndim != 1 or targets.size == 0:
        raise ValueError("labels and probabilities must be non-empty equal-length vectors")
    if set(np.unique(targets)) - {0, 1}:
        raise ValueError("labels must be real=0 or fake=1")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")

    best_threshold = 0.5
    best_correct = int((targets == (scores >= best_threshold)).sum())

    def consider(threshold: float, correct: int) -> None:
        nonlocal best_correct, best_threshold
        candidate_key = (abs(threshold - 0.5), threshold)
        best_key = (abs(best_threshold - 0.5), best_threshold)
        if correct > best_correct or (correct == best_correct and candidate_key < best_key):
            best_correct = correct
            best_threshold = threshold

    order = np.argsort(-scores, kind="stable")
    correct = int((targets == 0).sum())
    position = 0
    while position < len(order):
        score = float(scores[order[position]])
        consider(float(np.nextafter(score, np.inf)), correct)
        end = position
        while end < len(order) and scores[order[end]] == score:
            correct += 1 if targets[order[end]] == 1 else -1
            end += 1
        consider(score, correct)
        position = end
    return best_threshold


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
