from __future__ import annotations

from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = [
    "path",
    "label",
    "split",
    "source",
    "generator",
    "semantic_class",
    "real_source",
    "protocol",
    "content_env",
]


def validate_manifest(frame: pd.DataFrame) -> None:
    """Validate the stable CIFP manifest schema and split isolation."""
    missing = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")
    invalid_labels = sorted(set(frame["label"].astype(int)) - {0, 1})
    if invalid_labels:
        raise ValueError(f"manifest labels must be real=0 or fake=1, got {invalid_labels}")
    split_counts = frame.groupby("path", dropna=False)["split"].nunique()
    crossing = split_counts[split_counts > 1]
    if not crossing.empty:
        sample = crossing.index[0]
        raise ValueError(f"path appears in multiple splits: {sample}")
    duplicates = frame[frame.duplicated(subset=["path", "split"], keep=False)]
    if not duplicates.empty:
        raise ValueError(f"duplicate manifest path: {duplicates.iloc[0]['path']}")


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a Parquet or CSV manifest and validate it."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path.resolve()}")
    if manifest_path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(manifest_path)
    elif manifest_path.suffix.lower() == ".csv":
        frame = pd.read_csv(manifest_path, keep_default_na=False)
    else:
        raise ValueError(f"manifest must be .parquet or .csv: {manifest_path}")
    for column in ("generator", "semantic_class", "real_source"):
        if column in frame:
            frame[column] = frame[column].fillna("").astype(str)
    validate_manifest(frame)
    return frame.loc[:, MANIFEST_COLUMNS].reset_index(drop=True)


def write_manifest(frame: pd.DataFrame, path: str | Path) -> Path:
    """Validate and write a project-owned manifest."""
    validate_manifest(frame)
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.loc[:, MANIFEST_COLUMNS].copy()
    if manifest_path.suffix.lower() == ".parquet":
        ordered.to_parquet(manifest_path, index=False)
    elif manifest_path.suffix.lower() == ".csv":
        ordered.to_csv(manifest_path, index=False)
    else:
        raise ValueError(f"manifest must be .parquet or .csv: {manifest_path}")
    return manifest_path
