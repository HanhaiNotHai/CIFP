from __future__ import annotations

import csv
import json
from pathlib import Path
from types import TracebackType
from typing import TextIO

from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    """Rank-zero console, JSONL, CSV, and TensorBoard scalar logger."""

    def __init__(self, output_directory: str | Path, *, enabled: bool) -> None:
        self.enabled = enabled
        self.output = Path(output_directory)
        self.jsonl: TextIO | None = None
        self.csv_file: TextIO | None = None
        self.csv_writer: csv.DictWriter[str] | None = None
        self.tensorboard: SummaryWriter | None = None
        if enabled:
            self.output.mkdir(parents=True, exist_ok=True)
            self.jsonl = (self.output / "metrics.jsonl").open("a", encoding="utf-8")
            self.csv_file = (self.output / "metrics.csv").open("a", newline="", encoding="utf-8")
            self.tensorboard = SummaryWriter(self.output / "tensorboard")

    def log(self, metrics: dict[str, int | float | str | bool]) -> None:
        if not self.enabled:
            return
        print(json.dumps(metrics, sort_keys=True))
        assert self.jsonl is not None and self.csv_file is not None and self.tensorboard is not None
        self.jsonl.write(json.dumps(metrics, sort_keys=True) + "\n")
        self.jsonl.flush()
        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=list(metrics))
            if self.csv_file.tell() == 0:
                self.csv_writer.writeheader()
        self.csv_writer.writerow(metrics)
        self.csv_file.flush()
        step = int(metrics.get("step", 0))
        for key, value in metrics.items():
            if key != "step" and isinstance(value, int | float) and not isinstance(value, bool):
                self.tensorboard.add_scalar(key, value, step)
        self.tensorboard.flush()

    def close(self) -> None:
        if self.jsonl is not None:
            self.jsonl.close()
        if self.csv_file is not None:
            self.csv_file.close()
        if self.tensorboard is not None:
            self.tensorboard.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
