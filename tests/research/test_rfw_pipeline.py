from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import research.experiments.rfw_pipeline as module
from research.compression import PCACompressor
from research.datasets.rfw_aligned_bin import RFWAlignedPairBatch
from research.experiments.rfw_pipeline import (
    FrozenCodecSpec,
    evaluate_rfw_frozen_codecs,
    extract_rfw_origin_embeddings,
    frozen_codec_specs_from_completed_run,
    load_rfw_origin_embedding_artifact,
    rfw_frozen_codec_evaluation_uid,
    rfw_occurrence_pairs,
)
from research.runtime.hashing import sha256_file


def _pairs() -> pd.DataFrame:
    rows = []
    for official_index, (fold, genuine) in enumerate(
        ((0, True), (0, False), (1, True), (1, False))
    ):
        rows.append(
            {
                "pair_id": f"pair-{official_index}",
                "rfw_group": "African",
                "fold_index": fold,
                "official_index": official_index,
                "is_genuine": genuine,
            }
        )
    return pd.DataFrame(rows)


def _occurrences() -> pd.DataFrame:
    rows = []
    for pair_id, fold, genuine, official_index in (
        ("pair-0", 0, True, 0),
        ("pair-1", 0, False, 1),
        ("pair-2", 1, True, 2),
        ("pair-3", 1, False, 3),
    ):
        for side in ("left", "right"):
            rows.append(
                {
                    "occurrence_id": f"{pair_id}:{side}",
                    "pair_id": pair_id,
                    "side": side,
                    "rfw_group": "African",
                    "fold_index": fold,
                    "official_index": official_index,
                    "is_genuine": genuine,
                }
            )
    return pd.DataFrame(rows)


class _Adapter:
    def embed(self, faces: np.ndarray):
        rows = len(faces)
        values = np.zeros((rows, 512), dtype=np.float32)
        axes = (0, 0, 1, 2, 3, 3, 4, 5)
        for index in range(rows):
            values[index, axes[index]] = 1.0
        return SimpleNamespace(
            normalized_embedding=values,
            raw_norm=np.ones(rows, dtype=np.float32),
        )


def test_rfw_origin_embedding_artifact_is_hashed_and_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "aligned.tar.gz"
    archive.write_bytes(b"archive")
    model_spec = tmp_path / "model.json"
    model_spec.write_text("{}", encoding="utf-8")
    spec = SimpleNamespace(
        model_uid="edgeface-test",
        checkpoint=SimpleNamespace(sha256="a" * 64),
        preprocessing=SimpleNamespace(preprocess_hash="b" * 64),
        family="edgeface",
        architecture="edgeface_xs_gamma_06",
        training_dataset="webface12m",
    )
    monkeypatch.setattr(
        module,
        "inspect_rfw_aligned_bin_archive",
        lambda *args, **kwargs: SimpleNamespace(
            archive_path=archive.resolve(), archive_sha256="c" * 64
        ),
    )
    monkeypatch.setattr(module, "read_model_spec", lambda path: spec)
    monkeypatch.setattr(
        module, "create_pytorch_adapter_from_spec", lambda *args, **kwargs: _Adapter()
    )
    monkeypatch.setattr(
        module,
        "iter_rfw_aligned_pair_batches",
        lambda *args, **kwargs: iter(
            [
                RFWAlignedPairBatch(
                    faces=np.zeros((8, 112, 112, 3), dtype=np.uint8),
                    occurrences=_occurrences(),
                )
            ]
        ),
    )
    output = tmp_path / "artifact"

    artifact = extract_rfw_origin_embeddings(
        aligned_bin_archive_path=archive,
        pairs=_pairs(),
        model_spec_path=model_spec,
        output_dir=output,
        expected_model_uid="edgeface-test",
        device="cpu",
        strict_official=False,
    )
    reused = extract_rfw_origin_embeddings(
        aligned_bin_archive_path=archive,
        pairs=_pairs(),
        model_spec_path=model_spec,
        output_dir=output,
        expected_model_uid="edgeface-test",
        device="cpu",
        strict_official=False,
    )
    evaluation_pairs = rfw_occurrence_pairs(artifact)

    assert artifact.embeddings.shape == (8, 512)
    assert artifact.manifest["status"] == "completed"
    assert artifact.manifest["model_uid"] == "edgeface-test"
    assert reused.manifest == artifact.manifest
    assert evaluation_pairs["left_image_id"].tolist() == [
        "pair-0:left",
        "pair-1:left",
        "pair-2:left",
        "pair-3:left",
    ]

    codec_path = tmp_path / "pca_2.joblib"
    PCACompressor(2, random_state=42).fit(
        np.asarray(artifact.embeddings)
    ).save(codec_path)
    fit_manifest = tmp_path / "fit_manifest.json"
    codec_sha256 = sha256_file(codec_path)
    fit_manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "artifact_type": "frozen_compression_codec_bundle",
                "fit_source_dataset": "lfw",
                "fit_source_run_id": "lfw-test",
                "fit_on_rfw": False,
                "model_uid": "edgeface-test",
                "codecs": [
                    {
                        "profile_name": "pca_2",
                        "family": "pca",
                        "artifact_sha256": codec_sha256,
                        "artifact_byte_count": codec_path.stat().st_size,
                        "fit_seed": 42,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    codec = FrozenCodecSpec(
        profile_name="pca_2",
        family="pca",
        artifact_path=codec_path,
        artifact_sha256=codec_sha256,
        fit_source_dataset="lfw",
        fit_source_run_id="lfw-test",
        fit_manifest_path=fit_manifest,
        fit_manifest_sha256=sha256_file(fit_manifest),
    )
    evaluation = evaluate_rfw_frozen_codecs(
        origin_artifact_dir=output,
        codec_specs=[codec],
        output_dir=tmp_path / "evaluation",
        strict_official=False,
        bootstrap_repeats=100,
    )

    assert len(evaluation.profile_summary) == 3
    pca_rows = evaluation.profile_summary.loc[
        evaluation.profile_summary["compression_family"].eq("pca")
    ]
    assert pca_rows["codec_artifact_bytes"].gt(0).all()
    assert pca_rows["fit_on_rfw"].eq(False).all()  # noqa: E712
    assert pca_rows["macro_group_eer"].notna().all()
    assert pca_rows["group_eer_gap"].notna().all()
    assert pca_rows["eer_threshold_source"].eq(
        "heldout_fold_scores_and_labels"
    ).all()
    assert evaluation.manifest["open_set_protocol"] is False
    assert evaluation.manifest["codec_fit_on_rfw"] is False
    assert evaluation.manifest["metrics"] == ["accuracy", "tar", "far", "eer"]
    assert evaluation.manifest["eer_contract"] == {
        "threshold_source": "heldout_fold_scores_and_labels",
        "method": "heldout_scores_minimum_absolute_far_frr_v1",
        "uses_other_9_fold_accuracy_threshold": False,
    }


def test_frozen_codec_specs_are_resolved_from_explicit_completed_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "completed-run"
    workflow = run_dir / "artifacts/step2_workflow"
    codec_dir = run_dir / "artifacts/04_step2_compression_characterization/A001"
    workflow.mkdir(parents=True)
    codec_dir.mkdir(parents=True)
    (run_dir / "COMPLETED").write_text("\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"run_id": "run-1", "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    (workflow / "freeze_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "dataset_id": "lfw",
                "model_uid": "edgeface-test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    codec_path = codec_dir / "pca_2_A001.joblib"
    PCACompressor(2, random_state=42).fit(
        np.eye(4, 512, dtype=np.float32)
    ).save(codec_path)
    codec_sha256 = sha256_file(codec_path)
    codec_relative = codec_path.relative_to(run_dir).as_posix()
    codec_manifest = {
        "status": "completed",
        "artifact_type": "frozen_compression_codec_bundle",
        "fit_source_dataset": "lfw",
        "fit_source_run_id": "run-1",
        "model_uid": "edgeface-test",
        "fit_on_rfw": False,
        "codecs": [
            {
                "profile_name": "pca_2",
                "family": "pca",
                "artifact": codec_relative,
                "artifact_sha256": codec_sha256,
                "artifact_byte_count": codec_path.stat().st_size,
                "fit_seed": 42,
            }
        ],
    }
    (workflow / "frozen_codec_manifest.json").write_text(
        json.dumps(codec_manifest) + "\n", encoding="utf-8"
    )

    specs = frozen_codec_specs_from_completed_run(
        run_dir,
        expected_model_uid="edgeface-test",
        families=("pca",),
        profile_names=("pca_2",),
    )

    assert len(specs) == 1
    assert specs[0].artifact_path == codec_path.resolve()
    assert specs[0].fit_source_run_id == "run-1"


def test_rfw_evaluation_rejects_codec_from_different_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "aligned.tar.gz"
    archive.write_bytes(b"archive")
    model_spec = tmp_path / "model.json"
    model_spec.write_text("{}", encoding="utf-8")
    spec = SimpleNamespace(
        model_uid="edgeface-test",
        checkpoint=SimpleNamespace(sha256="a" * 64),
        preprocessing=SimpleNamespace(preprocess_hash="b" * 64),
        family="edgeface",
        architecture="edgeface_xs_gamma_06",
        training_dataset="webface12m",
    )
    monkeypatch.setattr(
        module,
        "inspect_rfw_aligned_bin_archive",
        lambda *args, **kwargs: SimpleNamespace(
            archive_path=archive.resolve(), archive_sha256="c" * 64
        ),
    )
    monkeypatch.setattr(module, "read_model_spec", lambda path: spec)
    monkeypatch.setattr(
        module, "create_pytorch_adapter_from_spec", lambda *args, **kwargs: _Adapter()
    )
    monkeypatch.setattr(
        module,
        "iter_rfw_aligned_pair_batches",
        lambda *args, **kwargs: iter(
            [
                RFWAlignedPairBatch(
                    faces=np.zeros((8, 112, 112, 3), dtype=np.uint8),
                    occurrences=_occurrences(),
                )
            ]
        ),
    )
    origin_dir = tmp_path / "origin"
    extract_rfw_origin_embeddings(
        aligned_bin_archive_path=archive,
        pairs=_pairs(),
        model_spec_path=model_spec,
        output_dir=origin_dir,
        expected_model_uid="edgeface-test",
        device="cpu",
        strict_official=False,
    )
    codec_path = tmp_path / "codec.joblib"
    PCACompressor(2, random_state=42).fit(
        np.eye(4, 512, dtype=np.float32)
    ).save(codec_path)
    fit_manifest = tmp_path / "fit_manifest.json"
    codec_sha256 = sha256_file(codec_path)
    fit_manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "artifact_type": "frozen_compression_codec_bundle",
                "fit_source_dataset": "lfw",
                "fit_source_run_id": "other-run",
                "fit_on_rfw": False,
                "model_uid": "arcface-test",
                "codecs": [
                    {
                        "profile_name": "pca_2",
                        "family": "pca",
                        "artifact_sha256": codec_sha256,
                        "artifact_byte_count": codec_path.stat().st_size,
                        "fit_seed": 42,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    codec = FrozenCodecSpec(
        profile_name="pca_2",
        family="pca",
        artifact_path=codec_path,
        artifact_sha256=codec_sha256,
        fit_source_dataset="lfw",
        fit_source_run_id="other-run",
        fit_manifest_path=fit_manifest,
        fit_manifest_sha256=sha256_file(fit_manifest),
    )

    with pytest.raises(ValueError, match="origin/model codec mismatch"):
        rfw_frozen_codec_evaluation_uid(
            origin_artifact_dir=origin_dir,
            codec_specs=(codec,),
            strict_official=False,
            bootstrap_repeats=100,
        )


def test_rfw_origin_embedding_loader_rejects_changed_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "_SUCCESS").write_text("invalid\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_rfw_origin_embedding_artifact(root)
