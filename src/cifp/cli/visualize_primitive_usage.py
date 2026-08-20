from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

PLAIN_FILENAME = "primitive_usage_heatmap.png"
ANNOTATED_FILENAME = "primitive_usage_heatmap_annotated.png"
GENERATOR_ORDER = (
    "Midjourney",
    "SDv1.4",
    "SDv1.5",
    "ADM",
    "GLIDE",
    "Wukong",
    "VQDM",
    "BigGAN",
)


def _usage_vector(value: object, *, name: str, primitive_count: int | None = None) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if primitive_count is not None and vector.size != primitive_count:
        raise ValueError(f"{name} has {vector.size} primitives, expected {primitive_count}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(vector < 0):
        raise ValueError(f"{name} contains negative usage values")
    return vector


def build_usage_matrix(report: Mapping[str, Any]) -> tuple[np.ndarray, list[str], set[int]]:
    """Build ordered usage and signed-difference rows from a usage report."""
    overall = _usage_vector(report.get("overall_usage"), name="overall_usage")
    primitive_count = overall.size
    by_label = report.get("by_label")
    by_generator = report.get("by_generator")
    if not isinstance(by_label, Mapping):
        raise ValueError("by_label must be an object")
    if not isinstance(by_generator, Mapping):
        raise ValueError("by_generator must be an object")
    real = _usage_vector(
        by_label.get("real"), name="by_label.real", primitive_count=primitive_count
    )
    generated = _usage_vector(
        by_label.get("fake"), name="by_label.fake", primitive_count=primitive_count
    )
    missing_generators = [name for name in GENERATOR_ORDER if name not in by_generator]
    if missing_generators:
        raise ValueError(f"by_generator缺少生成器: {', '.join(missing_generators)}")

    rows = [overall, real, generated, generated - real]
    labels = [
        "全体样本",
        "真实图像",
        "生成图像",
        "生成相对真实图像",
    ]
    for name in GENERATOR_ORDER:
        rows.append(
            _usage_vector(
                by_generator[name],
                name=f"by_generator.{name}",
                primitive_count=primitive_count,
            )
        )
        labels.append(name)
    return np.stack(rows), labels, {3}


def _cell_label(value: float, *, difference: bool) -> str:
    return f"{value:+.2f}" if difference else f"{value:.2f}"


def _chinese_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei"):
        if candidate in available:
            return candidate
    raise RuntimeError("未找到中文字体；请安装 Noto Sans CJK SC 或 WenQuanYi Micro Hei")


def _render_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    difference_rows: set[int],
    output: Path,
    *,
    annotate: bool,
) -> None:
    percentages = matrix * 100.0
    limit = float(np.max(np.abs(percentages)))
    if limit == 0:
        limit = 1.0
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    primitive_count = matrix.shape[1]
    figure, axis = plt.subplots(
        figsize=(max(12.0, primitive_count * 0.45), max(6.0, len(row_labels) * 0.5)),
        constrained_layout=True,
    )
    image = axis.imshow(percentages, cmap="RdBu_r", norm=norm, aspect="auto")
    axis.set_xticks(np.arange(primitive_count), [str(index) for index in range(primitive_count)])
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.tick_params(axis="x", labelrotation=0)
    axis.set_xlabel("取证基元")
    axis.set_ylabel("样本分组与使用率差")
    axis.set_title("取证基元使用分布")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("平均使用率 / 使用率差（% / 个百分点）")

    if annotate:
        for row in range(percentages.shape[0]):
            for column in range(percentages.shape[1]):
                value = percentages[row, column]
                label = _cell_label(value, difference=row in difference_rows)
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if abs(value) > limit * 0.5 else "black",
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="将 CIFP 基元使用率绘制为中文热力图")
    parser.add_argument("--input", type=Path, required=True, help="primitive_usage生成的usage.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="热力图输出目录")
    arguments = parser.parse_args(argv)
    try:
        report = json.loads(arguments.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取usage报告 {arguments.input.resolve()}: {error}") from error
    if not isinstance(report, Mapping):
        raise ValueError("usage报告顶层必须是JSON对象")
    matrix, row_labels, difference_rows = build_usage_matrix(report)
    with plt.rc_context({"font.family": _chinese_font(), "axes.unicode_minus": False}):
        _render_heatmap(
            matrix,
            row_labels,
            difference_rows,
            arguments.output_dir / PLAIN_FILENAME,
            annotate=False,
        )
        _render_heatmap(
            matrix,
            row_labels,
            difference_rows,
            arguments.output_dir / ANNOTATED_FILENAME,
            annotate=True,
        )
    print(f"纯热力图：{(arguments.output_dir / PLAIN_FILENAME).resolve()}")
    print(f"数值热力图：{(arguments.output_dir / ANNOTATED_FILENAME).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
