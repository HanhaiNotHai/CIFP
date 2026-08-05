from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import mutual_info_score


def _normalized_usage(usage: np.ndarray) -> np.ndarray:
    values = np.asarray(usage, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("usage must be a non-empty [samples, primitives] matrix")
    if np.any(values < 0):
        raise ValueError("primitive usage cannot be negative")
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)


def _group_usage(usage: np.ndarray, groups: Sequence[object]) -> dict[str, list[float]]:
    labels = np.asarray(groups).astype(str)
    if len(labels) != len(usage):
        raise ValueError("group labels must match usage rows")
    return {group: usage[labels == group].mean(axis=0).tolist() for group in sorted(set(labels))}


def primitive_usage_report(
    usage: np.ndarray,
    *,
    labels: np.ndarray,
    generators: np.ndarray,
    content_env: np.ndarray,
) -> dict[str, Any]:
    """Summarize overall and grouped primitive use and activation entropy."""
    normalized = _normalized_usage(usage)
    overall = normalized.mean(axis=0)
    label_names = np.where(np.asarray(labels, dtype=np.int64) == 1, "fake", "real")
    entropy = -np.sum(normalized * np.log(normalized + 1e-12), axis=1)
    return {
        "overall_usage": overall.tolist(),
        "by_label": _group_usage(normalized, label_names),
        "by_generator": _group_usage(normalized, generators),
        "by_content_env": _group_usage(normalized, content_env),
        "effective_primitive_count": int(np.count_nonzero(overall > 1e-4)),
        "mean_activation_entropy": float(entropy.mean()),
    }


def primitive_mutual_information(usage: np.ndarray, categories: Sequence[object]) -> np.ndarray:
    """Compute per-primitive MI after deterministic median activation binarization."""
    normalized = _normalized_usage(usage)
    targets = np.asarray(categories).astype(str)
    if len(targets) != len(normalized):
        raise ValueError("categories must match usage rows")
    scores = []
    for primitive in range(normalized.shape[1]):
        values = normalized[:, primitive]
        active = (values > np.median(values)).astype(np.int8)
        scores.append(mutual_info_score(targets, active))
    return np.asarray(scores, dtype=np.float64)


def primitive_coverage(
    train_usage: np.ndarray, unknown_usage: np.ndarray, *, top_r: int
) -> dict[str, Any]:
    """Measure primitive coverage and unseen top-r combinations."""
    train = _normalized_usage(train_usage)
    unknown = _normalized_usage(unknown_usage)
    if train.shape[1] != unknown.shape[1]:
        raise ValueError("train and unknown usage dimensions differ")
    if not 1 <= top_r <= train.shape[1]:
        raise ValueError("top_r must be within the primitive count")

    def combinations(values: np.ndarray) -> list[tuple[int, ...]]:
        indices = np.argpartition(values, -top_r, axis=1)[:, -top_r:]
        return [tuple(sorted(row.tolist())) for row in indices]

    train_combinations = combinations(train)
    unknown_combinations = combinations(unknown)
    train_primitives = set().union(*(set(row) for row in train_combinations))
    coverage = np.mean([len(set(row) & train_primitives) / top_r for row in unknown_combinations])
    seen_combinations = set(train_combinations)
    novelty = np.mean([row not in seen_combinations for row in unknown_combinations])
    return {
        "top_r": top_r,
        "train_primitive_set": sorted(train_primitives),
        "primitive_coverage": float(coverage),
        "combination_novelty": float(novelty),
        "train_unique_combinations": len(seen_combinations),
        "unknown_unique_combinations": len(set(unknown_combinations)),
    }


def primitive_coverage_by_group(
    train_usage: np.ndarray,
    unknown_usage: np.ndarray,
    unknown_groups: Sequence[object],
    *,
    top_r: int,
) -> dict[str, Any]:
    """Compare the seen training set with each unknown generator/source group."""
    groups = np.asarray(unknown_groups).astype(str)
    if len(groups) != len(unknown_usage):
        raise ValueError("unknown groups must match unknown usage rows")
    return {
        "aggregate": primitive_coverage(train_usage, unknown_usage, top_r=top_r),
        "per_unknown_source": {
            group: primitive_coverage(train_usage, unknown_usage[groups == group], top_r=top_r)
            for group in sorted(set(groups))
        },
    }
