from __future__ import annotations

from pathlib import Path

from cifp.cli.train import main


def test_synthetic_train_cli_writes_last_checkpoint(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--config",
            "configs/protocol/forensynths_selfsynthesis.yaml",
            "--synthetic",
            "--max-steps",
            "1",
            "--precision",
            "fp32",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--output",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "checkpoints" / "last.pt").is_file()
    assert (tmp_path / "resolved_config.yaml").is_file()
    assert (tmp_path / "run_metadata.json").is_file()
