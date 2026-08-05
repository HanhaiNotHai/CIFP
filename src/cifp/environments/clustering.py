from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def select_balanced_fit_indices(
    frame: pd.DataFrame, *, max_samples: int, random_state: int = 42
) -> np.ndarray:
    """Round-robin samples across label, semantic class, and source strata."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    required = {"label", "semantic_class", "source"}
    if missing := required - set(frame.columns):
        raise ValueError(f"frame is missing fit-stratification columns: {sorted(missing)}")
    rng = np.random.default_rng(random_state)
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in frame.reset_index(drop=True).iterrows():
        groups[(int(row["label"]), str(row["semantic_class"]), str(row["source"]))].append(index)
    for indices in groups.values():
        rng.shuffle(indices)
    keys = list(groups)
    rng.shuffle(keys)
    selected: list[int] = []
    while keys and len(selected) < min(max_samples, len(frame)):
        remaining: list[tuple[Any, ...]] = []
        for key in keys:
            indices = groups[key]
            if indices and len(selected) < max_samples:
                selected.append(indices.pop())
            if indices:
                remaining.append(key)
        keys = remaining
    return np.asarray(selected, dtype=np.int64)


def fit_content_environments(
    features: np.ndarray,
    frame: pd.DataFrame,
    *,
    environment_count: int = 100,
    max_fit_samples: int = 200000,
    random_state: int = 42,
    batch_size: int = 4096,
) -> tuple[MiniBatchKMeans, np.ndarray, np.ndarray]:
    """Fit balanced MiniBatchKMeans and assign every manifest row."""
    if features.ndim != 2 or features.shape[0] != len(frame):
        raise ValueError("features must be [manifest rows, feature dimension]")
    if len(frame) < environment_count:
        raise ValueError(f"environment_count={environment_count} exceeds sample count={len(frame)}")
    fit_indices = select_balanced_fit_indices(
        frame, max_samples=max_fit_samples, random_state=random_state
    )
    clusterer = MiniBatchKMeans(
        n_clusters=environment_count,
        batch_size=batch_size,
        random_state=random_state,
        n_init="auto",
    )
    clusterer.fit(np.asarray(features[fit_indices], dtype=np.float32))
    assignments = np.empty(len(frame), dtype=np.int32)
    for start in range(0, len(frame), batch_size):
        end = min(start + batch_size, len(frame))
        assignments[start:end] = clusterer.predict(
            np.asarray(features[start:end], dtype=np.float32)
        )
    return clusterer, assignments, fit_indices


def assign_fixed_random_environments(
    frame: pd.DataFrame, *, environment_count: int, random_state: int = 42
) -> pd.DataFrame:
    """Assign path-stable random environments for the approved ablation."""
    if environment_count <= 0:
        raise ValueError("environment_count must be positive")
    assigned = frame.copy()

    def environment(path: object) -> int:
        digest = hashlib.blake2b(f"{random_state}\0{path}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "little") % environment_count

    assigned["content_env"] = assigned["path"].map(environment).astype(np.int32)
    return assigned


def environment_audit(
    frame: pd.DataFrame,
    *,
    environment_count: int,
    unreadable_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Describe content-environment balance without filtering any sample."""
    counts = frame["content_env"].value_counts().sort_index()
    empty = [index for index in range(environment_count) if index not in counts]
    single_label: list[int] = []
    extreme: list[int] = []
    environments: dict[str, Any] = {}
    for environment in range(environment_count):
        group = frame[frame["content_env"] == environment]
        label_counts = group["label"].value_counts().sort_index()
        if not group.empty and group["label"].nunique() == 1:
            single_label.append(environment)
        fake_ratio = float(group["label"].mean()) if not group.empty else None
        if fake_ratio is not None and (fake_ratio < 0.05 or fake_ratio > 0.95):
            extreme.append(environment)
        environments[str(environment)] = {
            "count": int(len(group)),
            "real_count": int(label_counts.get(0, 0)),
            "fake_count": int(label_counts.get(1, 0)),
            "fake_ratio": fake_ratio,
            "source_distribution": {
                str(key): int(value) for key, value in group["source"].value_counts().items()
            },
            "semantic_class_distribution": {
                str(key): int(value)
                for key, value in group["semantic_class"].value_counts().items()
            },
        }
    duplicate_paths = sorted(
        frame.loc[frame.duplicated("path", keep=False), "path"].astype(str).unique()
    )
    return {
        "environment_count": environment_count,
        "environments": environments,
        "empty_environments": empty,
        "single_label_environments": single_label,
        "extremely_imbalanced_environments": extreme,
        "duplicate_paths": duplicate_paths,
        "unreadable_images": unreadable_paths or [],
    }
