from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cifp.config.loader import load_config
from cifp.data.dataset import ManifestImageDataset
from cifp.data.manifest import read_manifest
from cifp.data.transforms import ProtocolTransform
from cifp.environments.store import FeatureMemmap
from cifp.environments.teacher import FrozenSemanticTeacher


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline DINOv3 semantic feature extraction")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    manifest_path = arguments.manifest or Path(config["protocol"]["train_manifest"])
    protocol_name = str(config["protocol"]["name"])
    output = arguments.output or Path("artifacts/manifests") / protocol_name / "semantic_features"
    frame = read_manifest(manifest_path)
    transform = ProtocolTransform(
        crop_size=int(config["data"]["crop_size"]),
        training=False,
        small_image_policy=str(config["data"]["small_image_policy"]),
    )
    dataset = ManifestImageDataset(manifest_path, transform=transform)
    model_config = config["model"]
    teacher = FrozenSemanticTeacher.from_pretrained(
        str(model_config["model_id"]),
        model_path_env=str(model_config["model_path_env"]),
        local_files_only=bool(model_config["local_files_only"]),
    )
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA semantic extraction requested but CUDA is unavailable")
    teacher.to(device)
    store = FeatureMemmap(
        output,
        row_count=len(frame),
        feature_dim=2 * int(teacher.config.hidden_size),
    )
    index_path = output / "path_index.parquet"
    expected_index = pd.DataFrame({"row": np.arange(len(frame)), "path": frame["path"]})
    if index_path.exists():
        existing_index = pd.read_parquet(index_path)
        if not expected_index.equals(existing_index):
            raise ValueError("semantic feature path index differs from the current manifest")
    else:
        expected_index.to_parquet(index_path, index=False)
    pending = store.pending_indices()
    loader = DataLoader(
        Subset(dataset, pending.tolist()),
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=arguments.workers > 0,
    )
    for batch in tqdm(loader, desc="semantic features", disable=len(pending) == 0):
        row_indices = batch["index"].numpy()
        embeddings = teacher.extract(batch["image"].to(device, non_blocking=True))
        store.write(row_indices, embeddings.float().cpu().numpy())
    print(f"semantic rows complete: {len(frame) - len(store.pending_indices())}/{len(frame)}")
    print(f"feature store: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
