from __future__ import annotations

from pathlib import Path

from PIL import Image

from cifp.cli.train import _random_environments_requested
from cifp.config.loader import load_config
from cifp.data.builders import (
    build_forensynths_selfsynthesis,
    build_genimage_sd14,
    build_optional_ufd,
)


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128)).save(path)


def test_config_loader_merges_protocol_model_and_ablation() -> None:
    config = load_config("configs/ablation/random_dictionary.yaml")
    assert config["protocol"]["name"] == "forensynths_selfsynthesis"
    assert config["model"]["primitive_count"] == 32
    assert config["model"]["random_fixed_dictionary"] is True
    assert config["optimizer"]["name"] == "Adam"


def test_genimage_reflect_pad_variant_preserves_protocol_identity() -> None:
    standard = load_config("configs/protocol/genimage_sd14.yaml")
    variant = load_config("configs/protocol/genimage_sd14_reflect_pad.yaml")

    assert standard["data"]["small_image_policy"] == "error"
    assert variant["data"]["small_image_policy"] == "reflect_pad"
    assert variant["protocol"]["name"] == "genimage_sd14"
    assert variant["protocol"]["preprocessing_variant"] == "reflect_pad_non_protocol"
    assert variant["protocol"]["train_manifest"] == standard["protocol"]["train_manifest"]


def test_genimage_filtered_variant_uses_independent_artifacts() -> None:
    variant = load_config("configs/protocol/genimage_sd14_reflect_pad_filtered.yaml")

    assert variant["data"]["small_image_policy"] == "reflect_pad"
    assert variant["protocol"]["name"] == "genimage_sd14_filtered"
    assert variant["protocol"]["train_manifest"] == (
        "artifacts/manifests/genimage_sd14_filtered/train.parquet"
    )
    assert variant["environment"]["manifest_dir"] == "artifacts/manifests/genimage_sd14_filtered"


def test_random_environment_ablation_is_activated_by_config_or_cli() -> None:
    standard = load_config("configs/protocol/forensynths_selfsynthesis.yaml")
    ablation = load_config("configs/ablation/random_content_env.yaml")

    assert not _random_environments_requested(standard, False)
    assert _random_environments_requested(standard, True)
    assert _random_environments_requested(ablation, False)


def test_forensynths_selfsynthesis_builder_maps_labels_and_splits(tmp_path: Path) -> None:
    for split in ("train", "val"):
        for semantic_class in ("car", "cat", "chair", "horse"):
            _image(tmp_path / "foren" / split / semantic_class / "0_real" / "r.png")
            _image(tmp_path / "foren" / split / semantic_class / "1_fake" / "f.png")
    for source in (
        "AttGAN",
        "BEGAN",
        "CramerGAN",
        "InfoMaxGAN",
        "MMDGAN",
        "RelGAN",
        "S3GAN",
        "SNGAN",
        "STGAN",
    ):
        _image(tmp_path / "self" / source / "0_real" / "r.png")
        _image(tmp_path / "self" / source / "1_fake" / "f.png")

    manifests = build_forensynths_selfsynthesis(tmp_path / "foren", tmp_path / "self")

    assert len(manifests["train"]) == 8
    assert len(manifests["validation"]) == 8
    assert len(manifests["test"]) == 18
    assert set(manifests["train"]["label"]) == {0, 1}
    assert set(manifests["train"]["semantic_class"]) == {"car", "cat", "chair", "horse"}
    assert set(manifests["test"]["source"]) == {
        "AttGAN",
        "BEGAN",
        "CramerGAN",
        "InfoMaxGAN",
        "MMDGAN",
        "RelGAN",
        "S3GAN",
        "SNGAN",
        "STGAN",
    }


def test_genimage_builder_trains_only_on_sd14_and_tests_eight_sources(tmp_path: Path) -> None:
    aliases = {
        "Midjourney": "Midjourney",
        "SDv1.4": "stable_diffusion_v_1_4",
        "SDv1.5": "stable_diffusion_v_1_5",
        "ADM": "ADM",
        "GLIDE": "glide",
        "Wukong": "wukong",
        "VQDM": "VQDM",
        "BigGAN": "BigGAN",
    }
    for observed in aliases.values():
        for split in ("train", "val"):
            _image(tmp_path / observed / split / "nature" / "r.png")
            _image(tmp_path / observed / split / "ai" / "f.png")

    manifests = build_genimage_sd14(tmp_path)

    assert len(manifests["train"]) == 2
    assert set(manifests["train"]["source"]) == {"SDv1.4"}
    assert len(manifests["validation"]) == 2
    assert set(manifests["test"]["source"]) == set(aliases)
    assert set(manifests["test"]["split"]) == {"test"}


def test_optional_ufd_builder_accepts_explicit_binary_layouts(tmp_path: Path) -> None:
    for source in ("Guided", "LDM", "GLIDE", "DALL-E"):
        _image(tmp_path / source / "0_real" / "real.png")
        _image(tmp_path / source / "1_fake" / "fake.png")

    manifest = build_optional_ufd(tmp_path)["test"]

    assert len(manifest) == 8
    assert set(manifest["source"]) == {"Guided", "LDM", "GLIDE", "DALL-E"}
    assert set(manifest["label"]) == {0, 1}
