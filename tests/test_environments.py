from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cifp.environments.clustering import (
    assign_fixed_random_environments,
    environment_audit,
    fit_content_environments,
    select_balanced_fit_indices,
)
from cifp.environments.store import FeatureMemmap


def _frame(count: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path": [f"/images/{index}.png" for index in range(count)],
            "label": [index % 2 for index in range(count)],
            "split": ["train"] * count,
            "source": [f"source-{index % 3}" for index in range(count)],
            "generator": ["g" if index % 2 else "" for index in range(count)],
            "semantic_class": [f"class-{index % 4}" for index in range(count)],
            "real_source": ["real" if not index % 2 else "" for index in range(count)],
            "protocol": ["test"] * count,
            "content_env": [-1] * count,
        }
    )


def test_feature_memmap_resumes_and_preserves_row_alignment(tmp_path: Path) -> None:
    store = FeatureMemmap(tmp_path, row_count=5, feature_dim=6)
    store.write(np.array([1, 3]), np.ones((2, 6), dtype=np.float32))

    resumed = FeatureMemmap(tmp_path, row_count=5, feature_dim=6)
    assert resumed.pending_indices().tolist() == [0, 2, 4]
    assert np.allclose(resumed.features[1], 1.0)
    assert np.allclose(resumed.features[3], 1.0)


def test_balanced_fit_selection_is_deterministic_and_covers_strata() -> None:
    frame = _frame()
    first = select_balanced_fit_indices(frame, max_samples=12, random_state=42)
    second = select_balanced_fit_indices(frame, max_samples=12, random_state=42)
    selected = frame.iloc[first]
    assert np.array_equal(first, second)
    assert len(first) == 12
    assert set(selected["label"]) == {0, 1}
    assert len(set(selected["source"])) == 3
    assert len(set(selected["semantic_class"])) == 4


def test_content_environment_fit_assigns_every_row() -> None:
    frame = _frame()
    rng = np.random.default_rng(42)
    features = rng.normal(size=(len(frame), 8)).astype(np.float32)
    clusterer, assignments, fit_indices = fit_content_environments(
        features,
        frame,
        environment_count=4,
        max_fit_samples=20,
        random_state=42,
        batch_size=8,
    )
    assert clusterer.n_clusters == 4
    assert assignments.shape == (len(frame),)
    assert set(assignments) <= {0, 1, 2, 3}
    assert len(fit_indices) == 20


def test_fixed_random_environments_are_path_stable() -> None:
    frame = _frame()
    assigned = assign_fixed_random_environments(frame, environment_count=5, random_state=42)
    shuffled = frame.sample(frac=1, random_state=9).reset_index(drop=True)
    shuffled_assigned = assign_fixed_random_environments(
        shuffled, environment_count=5, random_state=42
    )
    mapping = dict(zip(assigned["path"], assigned["content_env"], strict=True))
    shuffled_mapping = dict(
        zip(shuffled_assigned["path"], shuffled_assigned["content_env"], strict=True)
    )
    assert mapping == shuffled_mapping


def test_environment_audit_reports_empty_single_label_and_duplicates() -> None:
    frame = _frame(8)
    frame.loc[:, "content_env"] = [0, 0, 1, 1, 1, 1, 1, 1]
    frame.loc[2:, "label"] = 1
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    report = environment_audit(frame, environment_count=4)
    assert report["empty_environments"] == [2, 3]
    assert 1 in report["single_label_environments"]
    assert report["duplicate_paths"] == ["/images/0.png"]
