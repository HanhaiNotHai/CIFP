from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional

from cifp.models.backbone import split_dinov3_tokens


class FrozenSemanticTeacher(nn.Module):
    """Offline-only frozen DINOv3 teacher producing CLS+mean-patch embeddings."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config  # type: ignore[attr-defined]
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        model_path_env: str = "CIFP_DINOV3_PATH",
        local_files_only: bool = True,
    ) -> FrozenSemanticTeacher:
        from transformers import AutoModel

        configured_path = os.environ.get(model_path_env)
        source = str(Path(configured_path).expanduser().resolve()) if configured_path else model_id
        try:
            model = AutoModel.from_pretrained(source, local_files_only=local_files_only)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"Unable to load frozen semantic teacher '{model_id}' from '{source}'. "
                f"Set {model_path_env} to the authorized local snapshot, or run "
                "`huggingface-cli login` and accept the DINOv3 license. No fallback is allowed."
            ) from error
        if getattr(model.config, "model_type", None) != "dinov3_vit":
            raise ValueError(
                f"required a DINOv3 ViT teacher, loaded model_type={model.config.model_type!r}"
            )
        return cls(model)

    def train(self, mode: bool = True) -> FrozenSemanticTeacher:
        super().train(False)
        return self

    @torch.inference_mode()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=images)
        cls_token, patches, _grid = split_dinov3_tokens(
            outputs.last_hidden_state,
            (int(images.shape[-2]), int(images.shape[-1])),
            self.config,
        )
        return torch.cat(
            [
                functional.normalize(cls_token, dim=-1),
                functional.normalize(patches.mean(dim=1), dim=-1),
            ],
            dim=-1,
        )
