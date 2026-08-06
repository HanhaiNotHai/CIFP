from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cifp.engine.distributed import gather_objects
from cifp.metrics.binary import evaluate_by_source

PREDICTION_COLUMNS = ["path", "label", "score", "prediction", "source"]


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    threshold: float = 0.5,
    expected_paths: set[str] | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate fake probabilities, gather ranks, and reject duplicate/missing samples."""
    model.eval()
    local_rows: list[dict[str, object]] = []
    module = getattr(model, "module", model)
    progress = tqdm(total=len(loader), desc="evaluate", unit="batch", disable=not show_progress)
    for batch in loader:
        logits = module.inference(batch["image"].to(device, non_blocking=True))
        scores = torch.sigmoid(logits).cpu().tolist()
        labels = batch["label"].long().cpu().tolist()
        for path, label, score, source in zip(
            batch["path"], labels, scores, batch["source"], strict=True
        ):
            local_rows.append(
                {
                    "path": str(path),
                    "label": int(label),
                    "score": float(score),
                    "prediction": int(score >= threshold),
                    "source": str(source),
                }
            )
        progress.update()
    progress.close()
    gathered = gather_objects(local_rows)
    rows = [row for rank_rows in gathered for row in rank_rows]
    paths = [str(row["path"]) for row in rows]
    if len(paths) != len(set(paths)):
        duplicates = sorted(path for path, count in Counter(paths).items() if count > 1)
        raise RuntimeError(f"distributed evaluation produced duplicate samples: {duplicates[:10]}")
    if expected_paths is not None and set(paths) != expected_paths:
        missing = sorted(expected_paths - set(paths))
        unexpected = sorted(set(paths) - expected_paths)
        raise RuntimeError(
            f"distributed evaluation sample mismatch; missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    report = evaluate_by_source(
        predictions["label"].tolist(),
        predictions["score"].tolist(),
        predictions["source"].tolist(),
        threshold=threshold,
    )
    return predictions, report


def _metrics_rows(report: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source, metrics in report["per_source"].items():
        rows.append({"scope": source, **metrics})
    rows.append(
        {
            "scope": "macro",
            "accuracy": report["macro"]["mAcc"],
            "average_precision": report["macro"]["mAP"],
        }
    )
    rows.append({"scope": "overall", **report["overall"]})
    return rows


def _display_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_display_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines) + "\n"


def _latex_table(frame: pd.DataFrame) -> str:
    columns = "l" * len(frame.columns)
    header = " & ".join(str(column).replace("_", "\\_") for column in frame.columns)
    rows = [
        " & ".join(_display_value(value).replace("_", "\\_") for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def save_evaluation_outputs(
    predictions: pd.DataFrame, report: dict[str, Any], output_directory: str | Path
) -> None:
    """Save machine-readable predictions/metrics and paper-ready Markdown/LaTeX tables."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    table = pd.DataFrame(_metrics_rows(report))
    table.to_csv(output / "metrics.csv", index=False)
    display_columns = [
        column
        for column in ("scope", "accuracy", "average_precision", "auroc", "count")
        if column in table
    ]
    display = table[display_columns]
    (output / "table.md").write_text(_markdown_table(display), encoding="utf-8")
    (output / "table.tex").write_text(_latex_table(display), encoding="utf-8")
