from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from cifp.data.manifest import MANIFEST_COLUMNS, validate_manifest, write_manifest

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FORENSYNTHS_CLASSES = ("car", "cat", "chair", "horse")
SELF_SYNTHESIS_SOURCES = (
    "AttGAN",
    "BEGAN",
    "CramerGAN",
    "InfoMaxGAN",
    "MMDGAN",
    "RelGAN",
    "S3GAN",
    "SNGAN",
    "STGAN",
)
GENIMAGE_ALIASES = {
    "Midjourney": "Midjourney",
    "SDv1.4": "stable_diffusion_v_1_4",
    "SDv1.5": "stable_diffusion_v_1_5",
    "ADM": "ADM",
    "GLIDE": "glide",
    "Wukong": "wukong",
    "VQDM": "VQDM",
    "BigGAN": "BigGAN",
}


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"required dataset directory does not exist: {directory}")
    paths = sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"required dataset directory contains no images: {directory}")
    return paths


def _rows(
    paths: Iterable[Path],
    *,
    label: int,
    split: str,
    source: str,
    generator: str,
    semantic_class: str,
    real_source: str,
    protocol: str,
) -> list[dict[str, object]]:
    return [
        {
            "path": str(path),
            "label": label,
            "split": split,
            "source": source,
            "generator": generator,
            "semantic_class": semantic_class,
            "real_source": real_source,
            "protocol": protocol,
            "content_env": -1,
        }
        for path in paths
    ]


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    validate_manifest(frame)
    return frame


def build_forensynths_selfsynthesis(
    forensynths_root: str | Path,
    self_synthesis_root: str | Path,
    *,
    class_aliases: dict[str, str] | None = None,
    source_aliases: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Enumerate protocol A without writing to either external data root."""
    foren_root = Path(forensynths_root)
    self_root = Path(self_synthesis_root)
    observed_classes = class_aliases or {name: name for name in FORENSYNTHS_CLASSES}
    observed_sources = source_aliases or {name: name for name in SELF_SYNTHESIS_SOURCES}
    manifests: dict[str, pd.DataFrame] = {}
    for output_split, observed_split in (("train", "train"), ("validation", "val")):
        rows: list[dict[str, object]] = []
        for semantic_class in FORENSYNTHS_CLASSES:
            observed_class = observed_classes.get(semantic_class)
            if observed_class is None:
                raise ValueError(f"missing class alias for {semantic_class}")
            for label, label_directory in ((0, "0_real"), (1, "1_fake")):
                paths = _images(foren_root / observed_split / observed_class / label_directory)
                rows.extend(
                    _rows(
                        paths,
                        label=label,
                        split=output_split,
                        source="ProGAN",
                        generator="ProGAN" if label else "",
                        semantic_class=semantic_class,
                        real_source="LSUN" if label == 0 else "",
                        protocol="forensynths_selfsynthesis",
                    )
                )
        manifests[output_split] = _frame(rows)
    test_rows: list[dict[str, object]] = []
    for source in SELF_SYNTHESIS_SOURCES:
        observed_source = observed_sources.get(source)
        if observed_source is None:
            raise ValueError(f"missing source alias for {source}")
        for label, label_directory in ((0, "0_real"), (1, "1_fake")):
            test_rows.extend(
                _rows(
                    _images(self_root / observed_source / label_directory),
                    label=label,
                    split="test",
                    source=source,
                    generator=source if label else "",
                    semantic_class="",
                    real_source="Self-Synthesis" if label == 0 else "",
                    protocol="forensynths_selfsynthesis",
                )
            )
    manifests["test"] = _frame(test_rows)
    return manifests


def _genimage_rows(
    root: Path,
    sources: Iterable[str],
    aliases: dict[str, str],
    *,
    observed_split: str,
    output_split: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in sources:
        observed = aliases[source]
        for label, label_directory in ((0, "nature"), (1, "ai")):
            rows.extend(
                _rows(
                    _images(root / observed / observed_split / label_directory),
                    label=label,
                    split=output_split,
                    source=source,
                    generator=source if label else "",
                    semantic_class="",
                    real_source="ImageNet" if label == 0 else "",
                    protocol="genimage_sd14",
                )
            )
    return rows


def build_genimage_sd14(
    root: str | Path, *, source_aliases: dict[str, str] | None = None
) -> dict[str, pd.DataFrame]:
    """Enumerate SDv1.4-only training and eight-source GenImage evaluation."""
    data_root = Path(root)
    aliases = source_aliases or GENIMAGE_ALIASES
    if missing := set(GENIMAGE_ALIASES) - set(aliases):
        raise ValueError(f"missing GenImage source aliases: {sorted(missing)}")
    return {
        "train": _frame(
            _genimage_rows(
                data_root,
                ["SDv1.4"],
                aliases,
                observed_split="train",
                output_split="train",
            )
        ),
        "validation": _frame(
            _genimage_rows(
                data_root,
                ["SDv1.4"],
                aliases,
                observed_split="val",
                output_split="validation",
            )
        ),
        "test": _frame(
            _genimage_rows(
                data_root,
                GENIMAGE_ALIASES,
                aliases,
                observed_split="val",
                output_split="test",
            )
        ),
    }


def build_optional_ufd(
    root: str | Path, *, source_aliases: dict[str, str] | None = None
) -> dict[str, pd.DataFrame]:
    """Enumerate extracted optional UFD sources using one explicit binary label layout."""
    data_root = Path(root)
    aliases = source_aliases or {
        "Guided": "Guided",
        "LDM": "LDM",
        "GLIDE": "GLIDE",
        "DALL-E": "DALL-E",
    }
    required_sources = {"Guided", "LDM", "GLIDE", "DALL-E"}
    if missing_aliases := required_sources - set(aliases):
        raise ValueError(f"missing optional UFD source aliases: {sorted(missing_aliases)}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"optional UFD root does not exist: {data_root.resolve()}")
    directories = {path.name.lower(): path for path in data_root.iterdir() if path.is_dir()}
    missing = [
        canonical for canonical, observed in aliases.items() if observed.lower() not in directories
    ]
    if missing:
        raise FileNotFoundError(
            f"optional UFD data are not extracted under {data_root.resolve()}; missing {missing}"
        )
    rows: list[dict[str, object]] = []
    label_layouts = (("0_real", "1_fake"), ("real", "fake"), ("nature", "ai"))
    for source in ("Guided", "LDM", "GLIDE", "DALL-E"):
        source_root = directories[aliases[source].lower()]
        selected_layout = next(
            (
                (real_directory, fake_directory)
                for real_directory, fake_directory in label_layouts
                if (source_root / real_directory).is_dir()
                and (source_root / fake_directory).is_dir()
            ),
            None,
        )
        if selected_layout is None:
            raise ValueError(
                f"optional UFD source has no supported explicit real/fake layout: {source_root}"
            )
        for label, label_directory in enumerate(selected_layout):
            rows.extend(
                _rows(
                    _images(source_root / label_directory),
                    label=label,
                    split="test",
                    source=source,
                    generator=source if label else "",
                    semantic_class="",
                    real_source="UFD" if label == 0 else "",
                    protocol="optional_ufd",
                )
            )
    return {"test": _frame(rows)}


def save_protocol_manifests(
    manifests: dict[str, pd.DataFrame], output_directory: str | Path
) -> dict[str, Path]:
    """Write protocol split manifests as Parquet files."""
    destination = Path(output_directory)
    return {
        split: write_manifest(frame, destination / f"{split}.parquet")
        for split, frame in manifests.items()
    }
