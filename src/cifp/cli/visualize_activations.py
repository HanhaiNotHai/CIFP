from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

from cifp.analysis.io import load_analysis_features, require_arrays

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _strict_center_crop(image: Image.Image, crop_size: int, path: Path) -> Image.Image:
    if min(image.size) < crop_size:
        raise ValueError(f"image is smaller than visualization crop {crop_size}: {path.resolve()}")
    left = (image.width - crop_size) // 2
    top = (image.height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render CIFP patch primitive activation heatmaps")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--top-primitives", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=128)
    arguments = parser.parse_args(argv)
    features = load_analysis_features(arguments.features)
    require_arrays(features, "assignments", "usage", "paths", "grid_size")
    grid_height, grid_width = (int(value) for value in features["grid_size"])
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(arguments.limit, len(features["paths"]))
    for sample_index in range(sample_count):
        path = Path(str(features["paths"][sample_index]))
        with Image.open(path) as opened:
            image = _strict_center_crop(opened.convert("RGB"), arguments.crop_size, path)
        top_primitives = np.argsort(features["usage"][sample_index])[-arguments.top_primitives :][
            ::-1
        ]
        for primitive in top_primitives:
            heatmap = features["assignments"][sample_index, :, primitive].reshape(
                grid_height, grid_width
            )
            figure, axis = plt.subplots(figsize=(5, 5))
            axis.imshow(image)
            axis.imshow(
                heatmap,
                cmap="magma",
                alpha=0.55,
                interpolation="nearest",
                extent=(0, image.width, image.height, 0),
            )
            axis.set_title(f"sample={sample_index}, primitive={int(primitive)}")
            axis.axis("off")
            figure.tight_layout()
            figure.savefig(
                arguments.output_dir / f"sample_{sample_index:05d}_primitive_{primitive:03d}.png",
                dpi=160,
            )
            plt.close(figure)
    (arguments.output_dir / "DISCLAIMER.txt").write_text(
        "These activation maps visualize model assignments; they are not automatically "
        "equivalent to human-interpretable forensic traces.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
