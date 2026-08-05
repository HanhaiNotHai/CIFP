from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cifp.engine.distributed import DistributedEvalSampler, global_batch_info
from cifp.engine.evaluator import evaluate_model, save_evaluation_outputs
from cifp.utils.logging import RunLogger


class _PredictionDataset(Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "image": torch.tensor([float(index)]),
            "label": torch.tensor(float(index % 2)),
            "path": f"/image/{index}.png",
            "source": "a" if index < 2 else "b",
        }


class _ScoreModel(nn.Module):
    def inference(self, images: torch.Tensor) -> torch.Tensor:
        return torch.where(images[:, 0].remainder(2) == 1, 3.0, -3.0)


class _DuplicatePathDataset(_PredictionDataset):
    def __getitem__(self, index: int) -> dict[str, object]:
        item = super().__getitem__(index)
        item["path"] = "/duplicate.png"
        return item


def test_global_batch_marks_six_gpu_non_protocol_run() -> None:
    standard = global_batch_info(
        per_gpu_batch_size=32,
        world_size=4,
        gradient_accumulation_steps=1,
        expected_global_batch_size=128,
    )
    nonstandard = global_batch_info(
        per_gpu_batch_size=32,
        world_size=6,
        gradient_accumulation_steps=1,
        expected_global_batch_size=128,
    )
    assert standard == {"actual_global_batch_size": 128, "non_protocol_batch": False}
    assert nonstandard == {"actual_global_batch_size": 192, "non_protocol_batch": True}


def test_distributed_eval_sampler_has_no_padding_or_overlap() -> None:
    dataset = list(range(5))
    partitions = [list(DistributedEvalSampler(dataset, 3, rank)) for rank in range(3)]
    flattened = [index for partition in partitions for index in partition]
    assert sorted(flattened) == dataset
    assert len(flattened) == len(set(flattened))


def test_evaluator_and_output_files(tmp_path: Path) -> None:
    dataset = _PredictionDataset()
    predictions, report = evaluate_model(
        _ScoreModel(),
        DataLoader(dataset, batch_size=2),
        device=torch.device("cpu"),
        expected_paths={f"/image/{index}.png" for index in range(4)},
    )
    save_evaluation_outputs(predictions, report, tmp_path)

    assert report["overall"]["accuracy"] == 1.0
    assert report["macro"]["mAcc"] == 1.0
    assert list(predictions.columns) == ["path", "label", "score", "prediction", "source"]
    for name in ("metrics.json", "metrics.csv", "predictions.csv", "table.md", "table.tex"):
        assert (tmp_path / name).is_file()
    assert json.loads((tmp_path / "metrics.json").read_text())["overall"]["accuracy"] == 1.0
    assert not pd.read_csv(tmp_path / "predictions.csv").empty


def test_evaluator_rejects_duplicate_sample_paths() -> None:
    with pytest.raises(RuntimeError, match="duplicate"):
        evaluate_model(
            _ScoreModel(),
            DataLoader(_DuplicatePathDataset(), batch_size=2),
            device=torch.device("cpu"),
        )


def test_run_logger_writes_jsonl_csv_and_tensorboard(tmp_path: Path) -> None:
    with RunLogger(tmp_path, enabled=True) as logger:
        logger.log({"step": 1, "detection": 0.5, "grl": 0.0})
        logger.log({"step": 2, "detection": 0.4, "grl": 0.1})
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2
    assert len(pd.read_csv(tmp_path / "metrics.csv")) == 2
    assert list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))
