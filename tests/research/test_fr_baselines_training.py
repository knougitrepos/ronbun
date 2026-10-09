"""Tests for face recognition baseline training modules, losses, and protocols."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from research.embeddings.base import ModelSpec
from research.embeddings.pytorch.official_loaders import (
    load_arcface_checkpoint,
    load_mobilefacenet_checkpoint,
)
from research.training.fr_baselines.classifiers import (
    AdaFaceClassifier,
    ArcFaceClassifier,
    MagFaceClassifier,
)
from research.training.fr_baselines.models import (
    FRBaselineTrainingModule,
    build_fr_baseline_model,
    export_backbone_checkpoint,
)
from research.training.fr_baselines.protocol import (
    FRDataset,
    build_survface_finetune_manifest,
)
from research.training.fr_baselines.trainer import (
    FRBaselineTrainer,
)


def test_arcface_classifier_forward_and_backward():
    classifier = ArcFaceClassifier(in_features=512, num_classes=10, scale=64.0, margin=0.6)
    features = torch.randn(4, 512, requires_grad=True)
    labels = torch.tensor([0, 2, 5, 9])

    logits = classifier(features, labels)
    assert logits.shape == (4, 10)

    loss = F.cross_entropy(logits, labels)
    loss.backward()
    assert features.grad is not None
    assert classifier.weight.grad is not None


def _make_spec(
    checkpoint: Path,
    *,
    family: str = "arcface",
    architecture: str = "mobilefacenet",
) -> ModelSpec:
    from research.embeddings.base import CheckpointProvenance, PreprocessingSpec

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


def test_adaface_classifier_ema_and_margin():
    classifier = AdaFaceClassifier(
        in_features=512, num_classes=10, scale=64.0, margin=0.4, h=0.333, t_alpha=0.1
    )
    classifier.train()
    features = (torch.randn(4, 512) * 15.0).detach().requires_grad_(True)
    labels = torch.tensor([1, 3, 4, 7])

    old_mean = classifier.batch_mean.item()
    logits = classifier(features, labels)
    assert logits.shape == (4, 10)
    assert classifier.batch_mean.item() != old_mean

    loss = F.cross_entropy(logits, labels)
    loss.backward()
    assert features.grad is not None

    # In eval mode, batch_mean should not change
    classifier.eval()
    cur_mean = classifier.batch_mean.item()
    _ = classifier(features, labels)
    assert classifier.batch_mean.item() == cur_mean


def test_magface_classifier_aux_loss_and_gradients():
    classifier = MagFaceClassifier(
        in_features=512,
        num_classes=10,
        scale=64.0,
        l_a=10.0,
        u_a=110.0,
        l_m=0.45,
        u_m=0.8,
        lambda_g=20.0,
    )
    features = (torch.randn(4, 512) * 20.0).detach().requires_grad_(True)
    labels = torch.tensor([0, 1, 2, 3])

    logits, aux_loss = classifier(features, labels)
    assert logits.shape == (4, 10)
    assert aux_loss.item() > 0.0

    total_loss = F.cross_entropy(logits, labels) + aux_loss
    total_loss.backward()
    assert features.grad is not None


def test_fr_baseline_training_module():
    model = build_fr_baseline_model(
        head_type="magface",
        num_classes=8,
        architecture="mobilefacenet",
        embedding_dim=512,
        dropout=0.0,
    )
    images = torch.randn(2, 3, 112, 112)
    labels = torch.tensor([0, 5])

    # Forward with labels
    out = model(images, labels)
    assert "loss" in out
    assert "ce_loss" in out
    assert "aux_loss" in out
    assert out["logits"].shape == (2, 8)
    assert out["features"].shape == (2, 512)

    # Forward without labels
    out_inference = model(images)
    assert "logits" in out_inference
    assert out_inference["features"].shape == (2, 512)


def test_export_backbone_checkpoint_roundtrip(tmp_path: Path):
    model = build_fr_baseline_model(
        head_type="arcface",
        num_classes=5,
        architecture="mobilefacenet",
        embedding_dim=512,
    )
    export_path = tmp_path / "test_backbone.pt"
    saved = export_backbone_checkpoint(
        model,
        export_path,
        metadata={"tested": True},
        verify_load=True,
    )
    assert saved.is_file()

    spec = _make_spec(saved, family="arcface")
    loaded = load_arcface_checkpoint(spec)
    features = loaded(torch.randn(1, 3, 112, 112))
    assert features.shape == (1, 512)


def test_survface_finetune_manifest_partition(tmp_path: Path):
    surv_root = tmp_path / "QMUL-SurvFace"
    train_dir = surv_root / "training_set"
    train_dir.mkdir(parents=True)

    # Create 10 identities with 2 images each
    for i in range(10):
        ident_dir = train_dir / f"id_{i:04d}"
        ident_dir.mkdir()
        (ident_dir / "001.jpg").write_bytes(b"dummy")
        (ident_dir / "002.jpg").write_bytes(b"dummy")

    bundle = build_survface_finetune_manifest(
        survface_root=surv_root,
        project_root=tmp_path,
        seed=123,
        train_identities_count=4,
        val_identities_count=2,
        calib_gallery_count=3,
        calib_probe_count=1,
        output_csv_path=tmp_path / "manifest.csv",
    )

    assert bundle.summary["total_identities"] == 10
    assert bundle.summary["train_identities"] == 4
    assert bundle.summary["val_identities"] == 2
    assert bundle.summary["calib_gallery_identities"] == 3
    assert bundle.summary["calib_probe_identities"] == 1

    df = bundle.manifest
    train_ids = set(df.loc[df["finetune_role"] == "train", "identity_id"])
    val_ids = set(df.loc[df["finetune_role"] == "val", "identity_id"])
    calib_gal_ids = set(df.loc[df["finetune_role"] == "calib_gallery", "identity_id"])
    calib_prb_ids = set(df.loc[df["finetune_role"] == "calib_probe", "identity_id"])

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & calib_gal_ids) == 0
    assert len(train_ids & calib_prb_ids) == 0
    assert len(val_ids & calib_gal_ids) == 0
    assert len(calib_gal_ids & calib_prb_ids) == 0


def test_fr_baseline_trainer_dry_run_fit(tmp_path: Path):
    train_ds = FRDataset(synthetic=True, synthetic_samples=8, synthetic_classes=4, is_train=True)
    val_ds = FRDataset(synthetic=True, synthetic_samples=4, synthetic_classes=4, is_train=False)

    model = build_fr_baseline_model(
        head_type="adaface",
        num_classes=4,
        architecture="mobilefacenet",
        embedding_dim=512,
    )

    trainer = FRBaselineTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=4,
        micro_batch_size=2,
        epochs=1,
        device="cpu",
        output_dir=tmp_path / "run_test",
        keep_raw_results=False,
        dry_run=True,
    )

    res = trainer.fit()
    assert "final_backbone" in res
    assert Path(res["final_backbone"]).is_file()
    assert Path(res["metrics_history"]).is_file()


def test_adaface_single_sample_and_zero_std_handling():
    classifier = AdaFaceClassifier(
        in_features=512, num_classes=10, scale=64.0, margin=0.4, h=0.333, t_alpha=0.1
    )
    classifier.train()

    # Single-element batch (size 1): safe_norms.std() produces NaN in PyTorch
    old_mean = classifier.batch_mean.item()
    features_single = (torch.ones(1, 512) * 5.0).requires_grad_(True)
    labels_single = torch.tensor([2])
    logits_single = classifier(features_single, labels_single)
    assert logits_single.shape == (1, 10)
    assert not torch.isnan(logits_single).any()
    # batch_mean must still be updated even when std is NaN
    assert classifier.batch_mean.item() != old_mean
    assert not torch.isnan(classifier.batch_std)

    # Identical norm batch (std = 0)
    features_identical = (torch.ones(4, 512) * 12.0).requires_grad_(True)
    labels_identical = torch.tensor([0, 1, 2, 3])
    logits_identical = classifier(features_identical, labels_identical)
    assert logits_identical.shape == (4, 10)
    assert not torch.isnan(logits_identical).any()


def test_build_fr_baseline_model_pretrained_cloning(tmp_path: Path):
    from research.embeddings.pytorch.official_backbones import build_mobilefacenet_backbone

    base_bb = build_mobilefacenet_backbone("mobilefacenet")
    # Clone into ArcFace
    m_arc = build_fr_baseline_model(
        head_type="arcface",
        num_classes=10,
        pretrained_backbone=base_bb,
    )
    # Clone into AdaFace
    m_ada = build_fr_baseline_model(
        head_type="adaface",
        num_classes=10,
        pretrained_backbone=base_bb,
    )
    # Weights of both backbones must be bitwise identical
    for p1, p2 in zip(m_arc.backbone.parameters(), m_ada.backbone.parameters()):
        assert torch.equal(p1, p2)

    # Test cloning from file path
    ckpt_file = tmp_path / "base_bb.pt"
    export_backbone_checkpoint(base_bb, ckpt_file, verify_load=False)
    m_mag = build_fr_baseline_model(
        head_type="magface",
        num_classes=10,
        pretrained_backbone=ckpt_file,
    )
    for p1, p3 in zip(base_bb.parameters(), m_mag.backbone.parameters()):
        assert torch.equal(p1, p3)


def test_fr_baseline_trainer_resume_and_grad_accum(tmp_path: Path):
    train_ds = FRDataset(synthetic=True, synthetic_samples=8, synthetic_classes=4, is_train=True)
    model = build_fr_baseline_model("arcface", num_classes=4)

    trainer1 = FRBaselineTrainer(
        model=model,
        train_dataset=train_ds,
        micro_batch_size=2,
        grad_accum_steps=3,
        epochs=1,
        device="cpu",
        output_dir=tmp_path / "run_accum",
        dry_run=True,
    )
    assert trainer1.accumulation_steps == 3
    assert trainer1.batch_size == 6
    trainer1.fit()

    ckpt_latest = tmp_path / "run_accum" / "checkpoint_latest.pt"
    assert ckpt_latest.is_file()

    # Resume in new trainer
    model2 = build_fr_baseline_model("arcface", num_classes=4)
    trainer2 = FRBaselineTrainer(
        model=model2,
        train_dataset=train_ds,
        micro_batch_size=2,
        grad_accum_steps=2,
        epochs=2,
        device="cpu",
        output_dir=tmp_path / "run_accum",
        resume_checkpoint=ckpt_latest,
        dry_run=True,
    )
    assert trainer2.current_epoch == 1


def test_fr_dataset_color_order_bgr(tmp_path: Path):
    from PIL import Image

    # Create dummy RGB image (pure red: R=255, G=0, B=0)
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    img_p = img_dir / "red.jpg"
    red_img = Image.new("RGB", (112, 112), color=(255, 0, 0))
    red_img.save(img_p)

    manifest_df = pd.DataFrame([{
        "image_path": "img/red.jpg",
        "identity_id": "id1",
        "finetune_role": "train",
        "split": "development",
    }])

    # BGR dataset
    ds_bgr = FRDataset(
        manifest_df,
        project_root=tmp_path,
        finetune_role="train",
        color_order="bgr",
        is_train=False,
    )
    tensor_bgr, _ = ds_bgr[0]
    # In BGR order, channel 0 is Blue (~ -1.0) and channel 2 is Red (~ 1.0)
    assert tensor_bgr[2].mean() > tensor_bgr[0].mean()

    # RGB dataset
    ds_rgb = FRDataset(
        manifest_df,
        project_root=tmp_path,
        finetune_role="train",
        color_order="rgb",
        is_train=False,
    )
    tensor_rgb, _ = ds_rgb[0]
    # In RGB order, channel 0 is Red (~ 1.0) and channel 2 is Blue (~ -1.0)
    assert tensor_rgb[0].mean() > tensor_rgb[2].mean()

