from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from cifp.losses.total import CIFPLoss
from cifp.models.adversarial import grl_coefficient
from cifp.models.cifp import CIFP, CIFPOutput
from cifp.utils.logging import RunLogger

Precision = Literal["bf16", "fp32"]


@dataclass(frozen=True)
class EpochResult:
    global_step: int
    optimizer_steps: int
    last_metrics: dict[str, float | int]


def _validate_precision(precision: Precision, device: torch.device) -> None:
    if precision == "fp32":
        return
    if precision != "bf16":
        raise ValueError(f"precision must be bf16 or fp32, got {precision}")
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "bf16 was requested but is unsupported on the selected device; "
            "set precision=fp32 explicitly (CIFP never silently switches to fp16)"
        )


def _module(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: CIFPLoss,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epoch: int,
    start_global_step: int,
    precision: Precision,
    gradient_accumulation_steps: int,
    max_optimizer_steps: int | None,
    logger: RunLogger,
    show_progress: bool = False,
    grl_max: float = 1.0,
    grl_warmup_epochs: int = 5,
    grl_ramp_end_epoch: int = 20,
) -> EpochResult:
    """Train one epoch and record every CIFP objective/mechanism statistic."""
    _validate_precision(precision, device)
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = start_global_step
    optimizer_steps = 0
    last_metrics: dict[str, float | int] = {}
    unwrapped = _module(model)
    optimizer_step_total = (len(loader) + gradient_accumulation_steps - 1) // (
        gradient_accumulation_steps
    )
    if max_optimizer_steps is not None:
        optimizer_step_total = min(optimizer_step_total, max_optimizer_steps)
    progress = tqdm(
        total=optimizer_step_total,
        desc=f"train epoch {epoch + 1}",
        unit="step",
        disable=not show_progress,
    )
    for batch_index, batch in enumerate(loader):
        fractional_epoch = epoch + batch_index / max(1, len(loader))
        grl = grl_coefficient(
            epoch=fractional_epoch,
            warmup_epochs=grl_warmup_epochs,
            ramp_end_epoch=grl_ramp_end_epoch,
            maximum=grl_max,
        )
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        content_env = batch["content_env"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            if isinstance(unwrapped, CIFP):
                output = model(images, grl_lambda=grl)
                assert isinstance(output, CIFPOutput)
                if torch.any(content_env < 0):
                    raise ValueError(
                        "training manifest contains unassigned content_env; run offline clustering "
                        "or request the fixed-random environment ablation explicitly"
                    )
                losses = criterion(
                    output,
                    labels,
                    content_env,
                    unwrapped.primitive_dictionary.normalized(),
                )
                environment_accuracy = (
                    (output.environment_logits.argmax(dim=-1) == content_env).float().mean()
                )
                mean_usage = output.assignments.mean(dim=(0, 1))
                effective_primitives = int((mean_usage > 1e-4).sum().item())
                activation_entropy = losses["sparse"]
            else:
                logits = model(images)
                detection = functional.binary_cross_entropy_with_logits(
                    logits, labels.to(logits.dtype)
                )
                zero = detection.new_zeros(())
                losses = {
                    "total": detection,
                    "detection": detection,
                    "composition": zero,
                    "sparse": zero,
                    "balance": zero,
                    "diversity": zero,
                    "nuisance": zero,
                }
                environment_accuracy = zero
                effective_primitives = 0
                activation_entropy = zero
            scaled_loss = losses["total"] / gradient_accumulation_steps
        scaled_loss.backward()
        is_accumulation_boundary = (batch_index + 1) % gradient_accumulation_steps == 0
        is_last_batch = batch_index + 1 == len(loader)
        if not (is_accumulation_boundary or is_last_batch):
            continue
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        global_step += 1
        last_metrics = {
            "step": global_step,
            "epoch": epoch,
            "total": float(losses["total"].detach()),
            "detection": float(losses["detection"].detach()),
            "composition": float(losses["composition"].detach()),
            "sparse": float(losses["sparse"].detach()),
            "balance": float(losses["balance"].detach()),
            "diversity": float(losses["diversity"].detach()),
            "nuisance": float(losses["nuisance"].detach()),
            "grl": float(grl),
            "environment_accuracy": float(environment_accuracy.detach()),
            "effective_primitives": effective_primitives,
            "activation_entropy": float(activation_entropy.detach()),
            "sampler_fallbacks": int(
                getattr(getattr(loader, "batch_sampler", None), "fallback_count", 0)
            ),
        }
        logger.log(last_metrics)
        progress.update()
        if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
            break
    progress.close()
    return EpochResult(global_step, optimizer_steps, last_metrics)
