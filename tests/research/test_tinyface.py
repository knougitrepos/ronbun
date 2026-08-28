from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.io import savemat

from research.datasets import (
    build_tinyface_official_bundle,
    select_tinyface_protocol_fraction,
)
from research.evaluation import (
    evaluate_tinyface_identification,
    load_tinyface_completed_evaluation,
    paired_tinyface_deltas,
)
from research.experiments import tinyface_pipeline
from research.runtime import ProgressReporter
from research.runtime.hashing import sha256_file


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=(value, value, value)).save(path)


def _tinyface_fixture(root: Path) -> Path:
    source = root / "tinyface"
    _image(source / "Training_Set/1/train_1.jpg", 10)
    _image(source / "Training_Set/2/train_2.jpg", 20)
    _image(source / "Testing_Set/Gallery_Match/g1.jpg", 30)
    _image(source / "Testing_Set/Gallery_Match/g2.jpg", 40)
    _image(source / "Testing_Set/Gallery_Distractor/d1.jpg", 50)
    _image(source / "Testing_Set/Gallery_Distractor/d2.jpg", 60)
    _image(source / "Testing_Set/Probe/p1.jpg", 70)
    _image(source / "Testing_Set/Probe/p2.jpg", 80)
    savemat(
        source / "Testing_Set/gallery_match_img_ID_pairs.mat",
        {
            "gallery_set": np.asarray(["g1.jpg", "g2.jpg"], dtype=object),
            "gallery_ids": np.asarray([1, 2], dtype=np.uint16),
        },
    )
    savemat(
        source / "Testing_Set/probe_img_ID_pairs.mat",
        {
            "probe_set": np.asarray(["p1.jpg", "p2.jpg"], dtype=object),
            "probe_ids": np.asarray([1, 2], dtype=np.uint16),
        },
    )
    return source


def test_tinyface_bundle_preserves_official_roles_without_open_set_claim(
    tmp_path: Path,
) -> None:
    source = _tinyface_fixture(tmp_path)
    bundle = build_tinyface_official_bundle(source, strict_official=False)

    assert bundle.summary["open_set_protocol"] is False
    assert bundle.summary["non_mated_probe_count"] == 0
    assert bundle.summary["fpir_tpir_metrics_applicable"] is False
    assert bundle.summary["role_counts"] == {
        "development_pool": 2,
        "gallery_match": 2,
        "gallery_distractor": 2,
        "registered_probe": 2,
    }
    distractors = bundle.manifest.loc[
        bundle.manifest["protocol_role"].eq("gallery_distractor")
    ]
    assert distractors["identity_id"].nunique() == 2
    assert distractors["identity_id"].str.startswith("tinyface:distractor:").all()
    assert bundle.manifest["preprocessing_mode"].eq("official_face_crop_resize").all()


def test_tinyface_strict_official_rejects_partial_fixture(tmp_path: Path) -> None:
    source = _tinyface_fixture(tmp_path)
    with pytest.raises(ValueError, match="official release"):
        build_tinyface_official_bundle(source, strict_official=True)


def test_tinyface_quick_scope_is_deterministic_and_not_paper_eligible(
    tmp_path: Path,
) -> None:
    bundle = build_tinyface_official_bundle(
        _tinyface_fixture(tmp_path), strict_official=False
    )
    first = select_tinyface_protocol_fraction(
        bundle.manifest,
        data_fraction=0.5,
        seed=42,
        minimum_development_samples=1,
    )
    second = select_tinyface_protocol_fraction(
        bundle.manifest,
        data_fraction=0.5,
        seed=42,
        minimum_development_samples=1,
    )

    pd.testing.assert_frame_equal(first, second)
    assert not first["official_result_eligible"].any()
    assert not first["scope_is_full"].any()
    assert set(first["protocol_role"]) == {
        "development_pool",
        "gallery_match",
        "gallery_distractor",
        "registered_probe",
    }
    probes = first.loc[first["protocol_role"].eq("registered_probe")]
    gallery = first.loc[first["protocol_role"].eq("gallery_match")]
    assert set(probes["identity_id"]) == set(gallery["identity_id"])


def test_tinyface_streaming_evaluator_matches_matlab_trapezoid_ap() -> None:
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    gallery = np.asarray(
        [
            [1.0, 0.0],
            [0.9, np.sqrt(1.0 - 0.9**2)],
            [0.8, np.sqrt(1.0 - 0.8**2)],
            [0.7, np.sqrt(1.0 - 0.7**2)],
        ],
        dtype=np.float32,
    )
    result = evaluate_tinyface_identification(
        query,
        gallery,
        query_identity_ids=["known"],
        gallery_identity_ids=["d1", "known", "d2", "known"],
        query_image_ids=["probe"],
        query_batch_size=1,
        gallery_batch_size=2,
    )

    row = result.per_query.iloc[0]
    assert row["first_positive_rank"] == 2
    assert row["rank_1_success"] == pytest.approx(False)
    assert row["rank_5_success"] == pytest.approx(True)
    # Official compute_AP.m updates precision at every rank. Positive ranks
    # [2, 4] therefore give 1/2*(0+1/2)/2 + 1/2*(1/3+2/4)/2 = 1/3.
    assert row["average_precision"] == pytest.approx(1.0 / 3.0)
    assert result.summary["mean_average_precision"] == pytest.approx(1.0 / 3.0)
    assert result.summary["fpir_tpir_metrics_applicable"] is False


def test_tinyface_paired_deltas_preserve_query_pairing() -> None:
    origin = pd.DataFrame(
        {
            "query_image_id": ["a", "b"],
            "average_precision": [1.0, 0.5],
            **{f"rank_{rank}_success": [True, False] for rank in (1, 5, 10, 20)},
        }
    )
    candidate = origin.copy()
    candidate.loc[1, "average_precision"] = 1.0
    for rank in (1, 5, 10, 20):
        candidate.loc[1, f"rank_{rank}_success"] = True

    result = paired_tinyface_deltas(
        origin,
        candidate,
        bootstrap_seed=42,
        bootstrap_repeats=100,
    )

    assert result["compressed_minus_origin_map"] == pytest.approx(0.25)
    assert result["compressed_minus_origin_rank_20"] == pytest.approx(0.5)
    assert result["paired_bootstrap_resamples"] == 100


def test_tinyface_embedding_extraction_uses_common_progress_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for index, path in enumerate(image_paths, start=1):
        _image(path, index * 10)
    selected = pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "identity_id": ["one", "two"],
            "split": ["development", "test"],
            "protocol_role": ["development_pool", "registered_probe"],
            "protocol_index": [0, 1],
            "source_relative_path": ["a.jpg", "b.jpg"],
            "official_result_eligible": [False, False],
            "protocol_uid": ["tinyface-test", "tinyface-test"],
            "image_path": [str(path) for path in image_paths],
        }
    )
    spec = SimpleNamespace(
        embedding_dim=2,
        preprocessing=SimpleNamespace(input_height=8, input_width=8),
    )

    class Adapter:
        def embed(self, faces: np.ndarray) -> SimpleNamespace:
            embeddings = np.tile(
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                (len(faces), 1),
            )
            return SimpleNamespace(normalized_embedding=embeddings)

    model_spec_path = tmp_path / "model_spec.json"
    model_spec_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        tinyface_pipeline,
        "read_model_spec",
        lambda *args, **kwargs: spec,
    )
    monkeypatch.setattr(
        tinyface_pipeline,
        "create_pytorch_adapter_from_spec",
        lambda *args, **kwargs: Adapter(),
    )
    plan = SimpleNamespace(
        model_spec_path=model_spec_path,
        device="cpu",
        embedding_batch_size=1,
        model_uid="tinyface-test-model",
        protocol_uid="tinyface-test",
        plan_id="tinyface-test-plan",
    )
    output = StringIO()
    reporter = ProgressReporter(
        "tinyface/full/test",
        heartbeat_seconds=None,
        milestone_percent=50,
        stream=output,
    )

    embeddings, rows = tinyface_pipeline._extract_embeddings(
        plan,
        selected,
        artifact_root=tmp_path / "artifacts",
        progress=reporter.callback(key_prefix="tinyface:full:"),
    )

    assert embeddings.shape == (2, 2)
    assert rows["image_id"].tolist() == ["a", "b"]
    log = output.getvalue()
    assert "TinyFace embedding extraction" in log
    assert "progress=50%" in log
    assert "progress=100%" in log


def test_load_tinyface_completed_evaluation_validates_closed_set_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "tinyface-run"
    output_root = run_dir / "artifacts/tinyface_official"
    output_root.mkdir(parents=True)
    (run_dir / "COMPLETED").write_text("done\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"status": "completed", "run_id": "T001"}),
        encoding="utf-8",
    )
    summary_path = output_root / "condition_summary.csv"
    per_query_path = output_root / "per_query.csv"
    pd.DataFrame(
        {
            "model_uid": ["edgeface-test"],
            "fpir_tpir_metrics_applicable": [False],
            "mean_average_precision": [0.5],
        }
    ).to_csv(summary_path, index=False)
    pd.DataFrame(
        {"query_image_id": ["p1"], "average_precision": [0.5]}
    ).to_csv(per_query_path, index=False)
    outputs = {
        path.name: {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (summary_path, per_query_path)
    }
    (output_root / "tinyface_evaluation_manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "tinyface_official_compression_evaluation_v1",
                "source_run_id": "T001",
                "dataset_id": "tinyface",
                "model_uid": "edgeface-test",
                "open_set_protocol": False,
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_tinyface_completed_evaluation(run_dir)

    assert loaded.manifest["open_set_protocol"] is False
    assert len(loaded.condition_summary) == 1
    assert len(loaded.per_query) == 1
