from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt
from PIL import Image

from cifp.cli.visualize_primitive_usage import (
    _cell_label,
    _chinese_font,
    _render_heatmap,
    build_usage_matrix,
    main,
)

GENERATOR_ORDER = [
    "Midjourney",
    "SDv1.4",
    "SDv1.5",
    "ADM",
    "GLIDE",
    "Wukong",
    "VQDM",
    "BigGAN",
]


def _usage_report() -> dict[str, object]:
    return {
        "overall_usage": [0.5, 0.5],
        "by_label": {"real": [0.7, 0.3], "fake": [0.2, 0.8]},
        "by_generator": {
            "": [0.7, 0.3],
            **{name: [0.1, 0.9] for name in GENERATOR_ORDER},
        },
    }


def test_build_usage_matrix_orders_rows_and_computes_signed_differences() -> None:
    matrix, row_labels, difference_rows = build_usage_matrix(_usage_report())

    assert row_labels == [
        "全体样本",
        "真实图像",
        "生成图像",
        "生成相对真实图像",
        *GENERATOR_ORDER,
    ]
    np.testing.assert_allclose(matrix[3], [-0.5, 0.5])
    assert difference_rows == {3}


def test_cell_labels_omit_percent_sign_and_sign_the_difference() -> None:
    assert _cell_label(2.345, difference=False) == "2.35"
    assert _cell_label(2.345, difference=True) == "+2.35"


def test_heatmap_x_tick_labels_are_horizontal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    with plt.rc_context({"font.family": _chinese_font()}):
        _render_heatmap(
            np.array([[0.5, 0.5]]),
            ["全体样本"],
            set(),
            tmp_path / "heatmap.png",
            annotate=False,
        )

    axis = plt.gcf().axes[0]
    assert {label.get_rotation() for label in axis.get_xticklabels()} == {0.0}


def test_annotated_heatmap_uses_approved_typography_and_bottom_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    with plt.rc_context({"font.family": _chinese_font()}):
        _render_heatmap(
            np.array([[0.5, 0.5]]),
            ["全体样本"],
            set(),
            tmp_path / "heatmap.png",
            annotate=True,
        )

    axis, colorbar_axis = plt.gcf().axes
    assert -0.3 < axis.title.get_position()[1] < 0.0
    assert axis.title.get_fontsize() == 18
    assert axis.xaxis.label.get_fontsize() == 16
    assert axis.yaxis.label.get_fontsize() == 16
    assert axis.yaxis.label.get_rotation() == 270
    assert axis.yaxis.labelpad == 30
    assert {label.get_fontsize() for label in axis.get_xticklabels()} == {12}
    assert {label.get_fontsize() for label in axis.get_yticklabels()} == {14}
    assert {label.get_fontsize() for label in colorbar_axis.get_yticklabels()} == {12}
    assert colorbar_axis.yaxis.label.get_fontsize() == 16
    assert colorbar_axis.yaxis.label.get_rotation() == 270
    assert colorbar_axis.yaxis.labelpad == 30
    assert {label.get_fontsize() for label in axis.texts} == {10}


def test_usage_heatmap_cli_writes_pngs_and_annotated_pdf_at_300_dpi(tmp_path: Path) -> None:
    input_path = tmp_path / "usage.json"
    input_path.write_text(json.dumps(_usage_report()), encoding="utf-8")

    assert main(["--input", str(input_path), "--output-dir", str(tmp_path)]) == 0

    for filename in (
        "primitive_usage_heatmap.png",
        "primitive_usage_heatmap_annotated.png",
        "primitive_usage_heatmap_annotated.pdf",
    ):
        output = tmp_path / filename
        assert output.is_file()
        assert output.stat().st_size > 0

    with Image.open(tmp_path / "primitive_usage_heatmap_annotated.png") as image:
        assert image.info["dpi"] == pytest.approx((300, 300), abs=0.1)


def test_build_usage_matrix_rejects_inconsistent_primitive_counts() -> None:
    report = _usage_report()
    report["by_label"] = {"real": [1.0], "fake": [0.2, 0.8]}

    with pytest.raises(ValueError, match="by_label.real has 1 primitives, expected 2"):
        build_usage_matrix(report)
