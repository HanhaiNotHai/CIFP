from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge configuration mappings without mutating either input."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve(reference: str | Path, parent: Path | None = None) -> Path:
    path = Path(reference).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_candidate = (Path.cwd() / path).resolve()
    if project_candidate.is_file() or parent is None:
        return project_candidate
    return (parent / path).resolve()


def _load(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"circular configuration reference: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"configuration does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    next_stack = (*stack, path)
    base: dict[str, Any] = {}
    base_reference = payload.pop("base_config", None)
    if base_reference is not None:
        base = _load(_resolve(str(base_reference), path.parent), next_stack)
    protocol = payload.get("protocol")
    if isinstance(protocol, dict) and "model_config" in protocol:
        model_reference = protocol["model_config"]
        model_config = _load(_resolve(str(model_reference), path.parent), next_stack)
        base = deep_merge(base, model_config)
    return deep_merge(base, payload)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML protocol or ablation with explicit recursive inheritance."""
    return _load(_resolve(path), ())
