from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional

SmallImagePolicy = Literal["error", "reflect_pad"]


class SmallImageError(ValueError):
    """Raised when a protocol crop cannot be taken without forbidden resizing."""


class ProtocolTransform:
    """128-crop CIFP preprocessing without resize or pixel mapping."""

    def __init__(
        self,
        *,
        crop_size: int = 128,
        training: bool,
        small_image_policy: SmallImagePolicy = "error",
        horizontal_flip: bool = False,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if small_image_policy not in {"error", "reflect_pad"}:
            raise ValueError(f"unsupported small_image_policy: {small_image_policy}")
        self.crop_size = crop_size
        self.training = training
        self.small_image_policy = small_image_policy
        self.horizontal_flip = horizontal_flip
        self.mean = mean
        self.std = std

    def _handle_small(self, image: Image.Image, path: Path) -> Image.Image:
        width, height = image.size
        if min(width, height) >= self.crop_size:
            return image
        if self.small_image_policy == "error":
            raise SmallImageError(
                f"image is smaller than {self.crop_size} and resize is forbidden: {path.resolve()} "
                f"(width={width}, height={height})"
            )
        pad_width = max(0, self.crop_size - width)
        pad_height = max(0, self.crop_size - height)
        left, right = pad_width // 2, pad_width - pad_width // 2
        top, bottom = pad_height // 2, pad_height - pad_height // 2
        array = np.asarray(image)
        padded = np.pad(array, ((top, bottom), (left, right), (0, 0)), mode="reflect")
        return Image.fromarray(padded)

    def __call__(self, image: Image.Image, path: str | Path) -> torch.Tensor:
        source_path = Path(path)
        image = self._handle_small(image.convert("RGB"), source_path)
        width, height = image.size
        if self.training:
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
        else:
            top = (height - self.crop_size) // 2
            left = (width - self.crop_size) // 2
        image = functional.crop(image, top, left, self.crop_size, self.crop_size)
        if self.training and self.horizontal_flip and random.random() < 0.5:
            image = functional.hflip(image)
        tensor = functional.pil_to_tensor(image).to(torch.float32).div_(255.0)
        return functional.normalize(tensor, self.mean, self.std)
