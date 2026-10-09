"""Tests for MobileFaceNet backbone construction and strict checkpoint loaders."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from research.embeddings.base import (
    CheckpointProvenance,
    ModelSpec,
    PreprocessingSpec,
)
from research.embeddings.pytorch.official_backbones import (
    MobileFaceNetBackbone,
    build_mobilefacenet_backbone,
)
from research.embeddings.pytorch.official_loaders import (
    CheckpointCompatibilityError,
    load_adaface_checkpoint,
    load_arcface_checkpoint,
    load_magface_checkpoint,
    load_mobilefacenet_checkpoint,
)


def _make_spec(
    checkpoint: Path,
    *,
    family: str = "arcface",
    architecture: str = "mobilefacenet",
) -> ModelSpec:
    return ModelSpec(
        family=family,
        architecture=architecture,
        training_dataset="test",
        implementation_repository="test",
        checkpoint=CheckpointProvenance.from_file(
            checkpoint,
            source_url="local://test",
        ),
        preprocessing=PreprocessingSpec(
            input_height=112,
            input_width=112,
            source_color_order="rgb",
            model_color_order="bgr",
            channel_mean=(127.5, 127.5, 127.5),
            channel_std=(128.0, 128.0, 128.0),
        ),
        target_layer="conv_sep.conv",
        embedding_dim=512,
    )


def test_build_mobilefacenet_backbone_topology():
    model = build_mobilefacenet_backbone("mobilefacenet", embedding_dim=512)
    assert isinstance(model, MobileFaceNetBackbone)

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 1_200_512

    # Check forward pass shape
    dummy = torch.randn(2, 3, 112, 112)
    out = model(dummy)
    assert out.shape == (2, 512)
    assert hasattr(model.conv_sep, "conv")


def test_load_mobilefacenet_checkpoint_formats(tmp_path: Path):
    backbone = build_mobilefacenet_backbone("mobilefacenet")
    raw_sd = backbone.state_dict()

    # Format 1: nested "state_dict"
    p1 = tmp_path / "f1.pt"
    torch.save({"state_dict": raw_sd}, p1)

    spec1 = _make_spec(p1, family="arcface", architecture="mobilefacenet")
    loaded1 = load_mobilefacenet_checkpoint(spec1)
    assert isinstance(loaded1, MobileFaceNetBackbone)

    # Format 2: with "module." prefix
    p2 = tmp_path / "f2.pt"
    prefixed_sd = {f"module.{k}": v for k, v in raw_sd.items()}
    torch.save({"state_dict": prefixed_sd}, p2)
    loaded2 = load_mobilefacenet_checkpoint(
        _make_spec(p2, family="arcface", architecture="mobilefacenet")
    )
    assert isinstance(loaded2, MobileFaceNetBackbone)

    # Format 3: direct state_dict
    p3 = tmp_path / "f3.pt"
    torch.save(raw_sd, p3)
    loaded3 = load_mobilefacenet_checkpoint(
        _make_spec(p3, family="arcface", architecture="mobilefacenet")
    )
    assert isinstance(loaded3, MobileFaceNetBackbone)


def test_load_across_all_families_for_mobilefacenet(tmp_path: Path):
    backbone = build_mobilefacenet_backbone("mobilefacenet")
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"state_dict": backbone.state_dict()}, ckpt)

    for family, loader in [
        ("arcface", load_arcface_checkpoint),
        ("adaface", load_adaface_checkpoint),
        ("magface", load_magface_checkpoint),
    ]:
        spec = _make_spec(ckpt, family=family, architecture="mobilefacenet")
        loaded = loader(spec)
        assert isinstance(loaded, MobileFaceNetBackbone)


def test_load_mobilefacenet_architecture_mismatch(tmp_path: Path):
    p = tmp_path / "dummy.pt"
    torch.save({}, p)
    spec = _make_spec(p, family="arcface", architecture="iresnet50")
    with pytest.raises(ValueError, match="architecture='mobilefacenet'"):
        load_mobilefacenet_checkpoint(spec)
