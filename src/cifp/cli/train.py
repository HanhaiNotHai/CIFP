from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as distributed
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from cifp.config.loader import load_config
from cifp.data.dataset import ManifestImageDataset
from cifp.data.sampler import EnvironmentBalancedBatchSampler
from cifp.data.synthetic import SyntheticImageDataset
from cifp.data.transforms import ProtocolTransform
from cifp.engine.checkpoint import load_checkpoint, save_checkpoint
from cifp.engine.distributed import (
    DistributedEvalSampler,
    cleanup_distributed,
    global_batch_info,
    initialize_distributed,
)
from cifp.engine.evaluator import evaluate_model, save_evaluation_outputs
from cifp.engine.trainer import Precision, train_one_epoch
from cifp.environments.clustering import assign_fixed_random_environments
from cifp.losses.total import CIFPLoss, LossWeights
from cifp.models.backbone import ForensicBackbone
from cifp.models.factory import build_model
from cifp.models.synthetic import TinyDINOv3
from cifp.utils.logging import RunLogger
from cifp.utils.reproducibility import (
    parameter_report,
    seed_everything,
    write_runtime_metadata,
)


def _loader_options(training: dict[str, Any], workers: int, device: torch.device) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(training["pin_memory"]) and device.type == "cuda",
    }
    if workers > 0:
        options["prefetch_factor"] = int(training["prefetch_factor"])
        options["persistent_workers"] = bool(training["persistent_workers"])
    return options


def _manifest_dataset(
    config: dict[str, Any], manifest: Path, *, training: bool
) -> ManifestImageDataset:
    data = config["data"]
    return ManifestImageDataset(
        manifest,
        transform=ProtocolTransform(
            crop_size=int(data["crop_size"]),
            training=training,
            small_image_policy=str(data["small_image_policy"]),
            horizontal_flip=bool(data.get("horizontal_flip", False)) and training,
        ),
    )


def _labels_and_environments(dataset: Any) -> tuple[list[int], list[int]]:
    if isinstance(dataset, SyntheticImageDataset):
        return dataset.labels, dataset.environments
    labels = [int(record["label"]) for record in dataset.records]
    environments = [int(record["content_env"]) for record in dataset.records]
    return labels, environments


def _apply_random_environments(dataset: ManifestImageDataset, count: int, seed: int) -> None:
    frame = pd.DataFrame(dataset.records)
    assigned = assign_fixed_random_environments(frame, environment_count=count, random_state=seed)
    dataset.records = assigned.to_dict(orient="records")


def _random_environments_requested(config: dict[str, Any], cli_flag: bool) -> bool:
    return cli_flag or config["environment"].get("assignment") == "fixed_random"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CIFP with native PyTorch DDP")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--random-content-env", action="store_true")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    arguments = parser.parse_args(argv)
    if arguments.max_steps is not None and not 1 <= arguments.max_steps <= 20:
        parser.error("--max-steps must be between 1 and 20")
    config = load_config(arguments.config)
    context = initialize_distributed(arguments.device)
    try:
        seed = int(config["protocol"]["seed"])
        seed_everything(seed + context.rank)
        environment_count = int(config["environment"]["count"])
        model_config = config["model"]
        if arguments.synthetic:
            backbone = ForensicBackbone(
                TinyDINOv3(),
                train_last_n_blocks=int(model_config.get("train_last_n_blocks", 2)),
                train_norm=bool(model_config.get("train_norm", False)),
            )
            train_dataset: Any = SyntheticImageDataset(
                sample_count=64, environment_count=environment_count, seed=seed
            )
            validation_dataset = None
        else:
            backbone = None
            train_dataset = _manifest_dataset(
                config, Path(config["protocol"]["train_manifest"]), training=True
            )
            if _random_environments_requested(config, arguments.random_content_env):
                _apply_random_environments(train_dataset, environment_count, seed)
            validation_path = Path(config["protocol"]["validation_manifest"])
            validation_dataset = _manifest_dataset(config, validation_path, training=False)
        model = build_model(
            model_config, environment_count=environment_count, backbone=backbone
        ).to(context.device)
        training = config["training"]
        workers = int(training["workers"] if arguments.workers is None else arguments.workers)
        labels, environments = _labels_and_environments(train_dataset)
        batch_sampler = EnvironmentBalancedBatchSampler(
            labels,
            environments,
            batch_size=int(training["per_gpu_batch_size"]),
            num_replicas=context.world_size,
            rank=context.rank,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            **_loader_options(training, workers, context.device),
        )
        validation_loader = None
        expected_validation_paths: set[str] | None = None
        if validation_dataset is not None:
            validation_sampler = DistributedEvalSampler(
                validation_dataset, context.world_size, context.rank
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=int(training["per_gpu_batch_size"]),
                sampler=validation_sampler,
                **_loader_options(training, workers, context.device),
            )
            expected_validation_paths = {
                str(Path(str(record["path"])).resolve()) for record in validation_dataset.records
            }
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("model has no trainable parameters")
        optimizer_config = config["optimizer"]
        if optimizer_config["name"] != "Adam":
            raise ValueError("paper-aligned CIFP supports torch.optim.Adam only")
        optimizer = torch.optim.Adam(
            parameters,
            lr=float(optimizer_config["lr"]),
            betas=tuple(float(value) for value in optimizer_config["betas"]),
            weight_decay=float(optimizer_config["weight_decay"]),
        )
        loss_config = config["loss"]
        criterion = CIFPLoss(
            LossWeights(
                lambda_comp=float(loss_config["lambda_comp"]),
                lambda_nui=float(loss_config["lambda_nui"]),
                sparse=float(loss_config["sparse_weight"]),
                balance=float(loss_config["balance_weight"]),
                diversity=float(loss_config["diversity_weight"]),
                pos_weight=loss_config["pos_weight"],
            )
        )
        run_id = arguments.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_directory = arguments.output or Path(config["output"]["root"]) / run_id
        if context.is_main:
            run_directory.mkdir(parents=True, exist_ok=True)
            (run_directory / "resolved_config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            metadata = write_runtime_metadata(run_directory / "run_metadata.json")
            print(json.dumps({"parameters": parameter_report(model)}, indent=2))
            batch_information = global_batch_info(
                per_gpu_batch_size=int(training["per_gpu_batch_size"]),
                world_size=context.world_size,
                gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
                expected_global_batch_size=int(training["expected_global_batch_size"]),
            )
            print(
                json.dumps(
                    {"per_gpu_batch_size": training["per_gpu_batch_size"], **batch_information}
                )
            )
        else:
            metadata = {}
        if context.world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
            )
        start_epoch = 0
        global_step = 0
        if arguments.resume is not None:
            resumed = load_checkpoint(arguments.resume, model, optimizer)
            start_epoch = resumed.epoch + 1
            global_step = resumed.global_step
        precision: Precision = arguments.precision or str(training["precision"])
        best_ap = float("-inf")
        best_accuracy = float("-inf")
        with RunLogger(run_directory, enabled=context.is_main) as logger:
            for epoch in range(start_epoch, int(training["epochs"])):
                batch_sampler.set_epoch(epoch)
                remaining = (
                    None if arguments.max_steps is None else arguments.max_steps - global_step
                )
                if remaining is not None and remaining <= 0:
                    break
                result = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device=context.device,
                    epoch=epoch,
                    start_global_step=global_step,
                    precision=precision,
                    gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
                    max_optimizer_steps=remaining,
                    logger=logger,
                    grl_max=float(loss_config["grl_max"]),
                    grl_warmup_epochs=int(loss_config["grl_warmup_epochs"]),
                    grl_ramp_end_epoch=int(loss_config["grl_ramp_end_epoch"]),
                )
                global_step = result.global_step
                if context.is_main:
                    save_checkpoint(
                        run_directory / "checkpoints" / "last.pt",
                        model,
                        optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        config=config,
                        metadata=metadata,
                    )
                if arguments.max_steps is not None:
                    if global_step >= arguments.max_steps:
                        break
                    continue
                assert validation_loader is not None and expected_validation_paths is not None
                predictions, report = evaluate_model(
                    model,
                    validation_loader,
                    device=context.device,
                    threshold=float(config["evaluation"]["threshold"]),
                    expected_paths=expected_validation_paths,
                )
                if context.is_main:
                    save_evaluation_outputs(
                        predictions, report, run_directory / "validation" / f"epoch_{epoch:04d}"
                    )
                    accuracy = float(report["overall"]["accuracy"])
                    average_precision = float(report["overall"]["average_precision"])
                    if average_precision > best_ap:
                        best_ap = average_precision
                        save_checkpoint(
                            run_directory / "checkpoints" / "best_validation_ap.pt",
                            model,
                            optimizer,
                            epoch=epoch,
                            global_step=global_step,
                            config=config,
                            metadata=metadata,
                        )
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        save_checkpoint(
                            run_directory / "checkpoints" / "best_validation_accuracy.pt",
                            model,
                            optimizer,
                            epoch=epoch,
                            global_step=global_step,
                            config=config,
                            metadata=metadata,
                        )
                if distributed.is_available() and distributed.is_initialized():
                    distributed.barrier()
        return 0
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
