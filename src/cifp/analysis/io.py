from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_analysis_features(path: str | Path, **arrays: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination


def load_analysis_features(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"analysis feature file does not exist: {source.resolve()}")
    with np.load(source, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def require_arrays(features: dict[str, np.ndarray], *names: str) -> None:
    missing = [name for name in names if name not in features]
    if missing:
        raise ValueError(f"analysis feature file is missing arrays: {missing}")
