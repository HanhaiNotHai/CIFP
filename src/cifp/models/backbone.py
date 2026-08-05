from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch
from torch import nn


class DINOv3ConfigLike(Protocol):
    hidden_size: int
    num_register_tokens: int
    patch_size: int | Sequence[int]


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"patch_size must contain two values, got {value}")
    return int(value[0]), int(value[1])


def split_dinov3_tokens(
    last_hidden_state: torch.Tensor,
    image_size: tuple[int, int],
    config: DINOv3ConfigLike,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Split DINOv3 output into CLS and true patch tokens using model config."""
    if last_hidden_state.ndim != 3:
        raise ValueError(
            f"last_hidden_state must have shape [B, tokens, hidden], got {last_hidden_state.shape}"
        )
    patch_height, patch_width = _pair(config.patch_size)
    image_height, image_width = image_size
    if image_height % patch_height or image_width % patch_width:
        raise ValueError(
            f"image shape {image_size} must be divisible by patch size "
            f"{(patch_height, patch_width)}"
        )
    grid = image_height // patch_height, image_width // patch_width
    patch_count = grid[0] * grid[1]
    patch_start = 1 + int(config.num_register_tokens)
    expected_tokens = patch_start + patch_count
    if last_hidden_state.shape[1] != expected_tokens:
        raise ValueError(
            f"DINOv3 returned {last_hidden_state.shape[1]} tokens, expected {expected_tokens} "
            f"(1 CLS + {config.num_register_tokens} registers + {patch_count} patches)"
        )
    if last_hidden_state.shape[2] != int(config.hidden_size):
        raise ValueError(
            f"DINOv3 hidden size {last_hidden_state.shape[2]} does not match config "
            f"{config.hidden_size}"
        )
    return last_hidden_state[:, 0], last_hidden_state[:, patch_start:], grid


class ForensicBackbone(nn.Module):
    """DINOv3 student exposing patch tokens while controlling trainable tail blocks."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        train_last_n_blocks: int = 2,
        train_norm: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config  # type: ignore[attr-defined]
        self.hidden_size = int(self.config.hidden_size)
        self.patch_size = _pair(self.config.patch_size)
        self.num_register_tokens = int(self.config.num_register_tokens)
        self._configure_trainability(train_last_n_blocks, train_norm)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        model_path_env: str = "CIFP_DINOV3_PATH",
        local_files_only: bool = True,
        train_last_n_blocks: int = 2,
        train_norm: bool = False,
    ) -> ForensicBackbone:
        """Load exactly the configured DINOv3 model without fallback."""
        from transformers import AutoModel

        configured_path = os.environ.get(model_path_env)
        source = str(Path(configured_path).expanduser().resolve()) if configured_path else model_id
        try:
            backbone = AutoModel.from_pretrained(source, local_files_only=local_files_only)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"Unable to load required DINOv3 model '{model_id}' from '{source}'. "
                f"Set {model_path_env} to the authorized local snapshot, or run "
                "`huggingface-cli login` and accept the DINOv3 model license. "
                "CIFP will not fall back to another model or random weights."
            ) from error
        if getattr(backbone.config, "model_type", None) != "dinov3_vit":
            raise ValueError(
                f"required a DINOv3 ViT, loaded model_type={backbone.config.model_type!r}"
            )
        return cls(
            backbone,
            train_last_n_blocks=train_last_n_blocks,
            train_norm=train_norm,
        )

    def _blocks(self) -> nn.ModuleList:
        encoder = getattr(self.backbone, "model", None)
        blocks = getattr(encoder, "layer", None)
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError("DINOv3 backbone must expose transformer blocks as model.layer")
        return blocks

    def _configure_trainability(self, train_last_n_blocks: int, train_norm: bool) -> None:
        blocks = self._blocks()
        if not 0 <= train_last_n_blocks <= len(blocks):
            raise ValueError(
                f"train_last_n_blocks must be in [0, {len(blocks)}], got {train_last_n_blocks}"
            )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if train_last_n_blocks:
            for block in blocks[-train_last_n_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
        for module in self.backbone.modules():
            if isinstance(module, nn.LayerNorm):
                for parameter in module.parameters():
                    parameter.requires_grad_(train_norm)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int], torch.Tensor]:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError(f"pixel_values must have shape [B, 3, H, W], got {pixel_values.shape}")
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token, patch_tokens, grid = split_dinov3_tokens(
            outputs.last_hidden_state,
            (int(pixel_values.shape[-2]), int(pixel_values.shape[-1])),
            self.config,
        )
        return patch_tokens, grid, cls_token
