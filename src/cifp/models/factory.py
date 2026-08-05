from __future__ import annotations

from typing import Any

from torch import nn

from cifp.models.backbone import ForensicBackbone
from cifp.models.baselines import CLSBaseline, PatchMeanBaseline
from cifp.models.cifp import CIFP


def build_model(
    model_config: dict[str, Any],
    *,
    environment_count: int | None,
    backbone: ForensicBackbone | None = None,
) -> nn.Module:
    """Build an explicitly configured detector without semantic-teacher fallback."""
    if backbone is None:
        backbone = ForensicBackbone.from_pretrained(
            str(model_config["model_id"]),
            model_path_env=str(model_config.get("model_path_env", "CIFP_DINOV3_PATH")),
            local_files_only=bool(model_config.get("local_files_only", True)),
            train_last_n_blocks=int(model_config.get("train_last_n_blocks", 2)),
            train_norm=bool(model_config.get("train_norm", False)),
        )
    kind = str(model_config.get("kind", "cifp"))
    if kind == "patch_mean":
        return PatchMeanBaseline(backbone)
    if kind == "cls":
        return CLSBaseline(backbone)
    if kind != "cifp":
        raise ValueError(f"unsupported model kind: {kind}")
    assignment = str(model_config.get("assignment", "topk"))
    if assignment not in {"topk", "dense"}:
        raise ValueError(f"unsupported assignment backend: {assignment}")
    top_k = None if assignment == "dense" else model_config.get("top_k", 4)
    cooccurrence = model_config.get("cooccurrence", {})
    return CIFP(
        backbone,
        forensic_dim=int(model_config.get("forensic_dim", 256)),
        primitive_count=int(model_config.get("primitive_count", 32)),
        temperature=float(model_config.get("temperature", 0.1)),
        top_k=None if top_k is None else int(top_k),
        composition_dim=int(model_config.get("composition_dim", 128)),
        environment_count=environment_count,
        random_fixed_dictionary=bool(model_config.get("random_fixed_dictionary", False)),
        cooccurrence_enabled=bool(cooccurrence.get("enabled", False)),
        cooccurrence_dim=int(cooccurrence.get("output_dim", 128)),
    )
