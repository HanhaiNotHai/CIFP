from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from torch.utils.data import DataLoader

from cifp.config.loader import load_config
from cifp.data.dataset import ManifestImageDataset
from cifp.data.transforms import ProtocolTransform
from cifp.engine.checkpoint import load_checkpoint
from cifp.engine.distributed import (
    DistributedEvalSampler,
    cleanup_distributed,
    initialize_distributed,
)
from cifp.engine.evaluator import evaluate_model, save_evaluation_outputs
from cifp.models.factory import build_model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a CIFP checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--test-oracle-threshold",
        action="store_true",
        help="select the global accuracy-optimal threshold using test labels (leaky analysis)",
    )
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    context = initialize_distributed(arguments.device)
    try:
        manifest = arguments.manifest or Path(config["protocol"]["test_manifest"])
        data_config = config["data"]
        dataset = ManifestImageDataset(
            manifest,
            transform=ProtocolTransform(
                crop_size=int(data_config["crop_size"]),
                training=False,
                small_image_policy=str(data_config["small_image_policy"]),
            ),
        )
        model = build_model(
            config["model"], environment_count=int(config["environment"]["count"])
        ).to(context.device)
        load_checkpoint(arguments.checkpoint, model)
        sampler = DistributedEvalSampler(dataset, context.world_size, context.rank)
        workers = int(
            config["training"]["workers"] if arguments.workers is None else arguments.workers
        )
        loader_options: dict[str, object] = {
            "num_workers": workers,
            "pin_memory": bool(config["training"]["pin_memory"]) and context.device.type == "cuda",
        }
        if workers > 0:
            loader_options["prefetch_factor"] = int(config["training"]["prefetch_factor"])
            loader_options["persistent_workers"] = bool(config["training"]["persistent_workers"])
        loader = DataLoader(
            dataset,
            batch_size=int(config["training"]["per_gpu_batch_size"]),
            sampler=sampler,
            **loader_options,
        )
        expected_paths = {str(Path(str(record["path"])).resolve()) for record in dataset.records}
        predictions, report = evaluate_model(
            model,
            loader,
            device=context.device,
            threshold=float(config["evaluation"]["threshold"]),
            expected_paths=expected_paths,
            show_progress=context.is_main,
            select_test_oracle_threshold=arguments.test_oracle_threshold,
        )
        if context.is_main:
            output = arguments.output or Path("outputs") / f"eval_{arguments.checkpoint.stem}"
            save_evaluation_outputs(predictions, report, output)
            print(f"evaluation outputs: {output.resolve()}")
        return 0
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
