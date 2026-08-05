from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cifp.analysis.io import save_analysis_features
from cifp.config.loader import load_config
from cifp.data.dataset import ManifestImageDataset
from cifp.data.transforms import ProtocolTransform
from cifp.engine.checkpoint import load_checkpoint
from cifp.models.cifp import CIFP, CIFPOutput
from cifp.models.factory import build_model


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract CIFP primitive analysis tensors")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA analysis extraction requested but unavailable")
    data = config["data"]
    dataset = ManifestImageDataset(
        arguments.manifest,
        transform=ProtocolTransform(
            crop_size=int(data["crop_size"]),
            training=False,
            small_image_policy=str(data["small_image_policy"]),
        ),
    )
    model = build_model(config["model"], environment_count=int(config["environment"]["count"])).to(
        device
    )
    if not isinstance(model, CIFP):
        raise TypeError("primitive analysis requires a CIFP model, not a pooling baseline")
    load_checkpoint(arguments.checkpoint, model)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=arguments.workers > 0,
    )
    collected: dict[str, list[np.ndarray]] = {
        "assignments": [],
        "usage": [],
        "z_for": [],
        "fake_logits": [],
        "labels": [],
        "content_env": [],
    }
    strings: dict[str, list[str]] = {
        "paths": [],
        "sources": [],
        "generators": [],
        "semantic_classes": [],
        "real_sources": [],
    }
    grid_size: tuple[int, int] | None = None
    for batch in tqdm(loader, desc="analysis features"):
        output = model(batch["image"].to(device), grl_lambda=None)
        assert isinstance(output, CIFPOutput)
        if grid_size is not None and output.grid_size != grid_size:
            raise ValueError("analysis archive requires a consistent patch grid")
        grid_size = output.grid_size
        collected["assignments"].append(output.assignments.cpu().numpy().astype(np.float16))
        collected["usage"].append(output.assignments.mean(dim=1).cpu().numpy())
        collected["z_for"].append(output.z_for.cpu().numpy())
        collected["fake_logits"].append(output.fake_logits.cpu().numpy())
        collected["labels"].append(batch["label"].numpy().astype(np.int8))
        collected["content_env"].append(batch["content_env"].numpy().astype(np.int32))
        strings["paths"].extend(str(value) for value in batch["path"])
        strings["sources"].extend(str(value) for value in batch["source"])
        strings["generators"].extend(str(value) for value in batch["generator"])
        strings["semantic_classes"].extend(str(value) for value in batch["semantic_class"])
        strings["real_sources"].extend(str(value) for value in batch["real_source"])
    assert grid_size is not None
    save_analysis_features(
        arguments.output,
        **{name: np.concatenate(values, axis=0) for name, values in collected.items()},
        **{name: np.asarray(values, dtype=str) for name, values in strings.items()},
        grid_size=np.asarray(grid_size, dtype=np.int32),
    )
    print(f"analysis features: {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
