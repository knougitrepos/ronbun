from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from research.fiqa import FIQAScoreArtifact
from research.calibration.conditional import IDENTIFICATION_METRIC_CONTRACT
from research.runtime.hashing import sha256_file
from research.experiments.fiqa_threshold_calibration import (
    _adc_condition_score_frame,
    _standard_scores,
    CalibrationComparison,
    ConditionScoreTables,
    assess_saliency_incremental_readiness,
    join_fiqa_scores,
    join_fiqa_score_artifacts,
    load_calibration_comparison_artifact,
    load_condition_score_artifact,
    run_global_vs_fiqa_calibration,
    write_calibration_comparison_artifact,
    write_condition_score_artifact,
    upgrade_condition_score_artifact,
)


def _scores(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"{prefix}-a", f"{prefix}-b"],
            "identity_id": ["id-a", "id-b"],
            "evaluation_split": [prefix, prefix],
            "is_mated": [True, False],
            "score": [-0.1, -0.2],
            "true_identity_score": [-0.1, np.nan],
            "true_identity_rank": [1.0, np.nan],
            "rank1_correct": [True, False],
            "top_k_correct": [True, False],
            "top_k": [20, 20],
            "compression_profile": ["pq_512_m128_b8"] * 2,
            "search_mode": ["pq_adc_exhaustive"] * 2,
            "protocol_uid": ["protocol"] * 2,
            "model_uid": ["arcface-test"] * 2,
            "dataset_id": ["survface"] * 2,
            "origin_embedding_artifact_uid": ["origin-test"] * 2,
            "extraction_uid": ["population-test"] * 2,
            "score_space": ["negative_squared_l2_adc"] * 2,
            "threshold_comparator": [">="] * 2,
            "aligned_content_sha256": [f"hash-{prefix}-a", f"hash-{prefix}-b"],
        }
    )


def test_adc_condition_score_frame_does_not_require_origin_retrieval() -> None:
    frame = _adc_condition_score_frame(
        np.array([[0.2, 0.7], [0.4, 0.9]], dtype=np.float32),
        np.array([[1, 0], [0, 1]], dtype=np.int64),
        query_ids=np.array(["q-a", "q-u"], dtype=object),
        query_identity_ids=np.array(["a", "unknown"], dtype=object),
        gallery_identity_ids=np.array(["a", "b"], dtype=object),
        compression_profile="pq_512_m128_b8",
        search_mode="pq_adc_exhaustive",
    )

    assert frame["compressed_top1_score"].to_numpy() == pytest.approx(
        [-0.2, -0.4]
    )
    assert frame["compressed_rank1_correct"].tolist() == [False, False]
    assert frame["compressed_top_k_correct"].tolist() == [True, False]
    assert frame["is_mated"].tolist() == [True, False]
    assert frame["compressed_score_space"].eq("negative_squared_l2_adc").all()
    assert frame.loc[0, "compressed_true_identity_score"] == pytest.approx(-0.7)
    assert frame.loc[0, "compressed_true_identity_rank"] == 2
    assert np.isnan(frame.loc[1, "compressed_true_identity_score"])
    assert np.isnan(frame.loc[1, "compressed_true_identity_rank"])


def test_fiqa_join_requires_complete_one_to_one_scores():
    condition = _scores("calibration")
    fiqa = pd.DataFrame(
        {
            "sample_id": condition["sample_id"],
            "fiqa_score": [0.1, 0.9],
            "fiqa_model_uid": ["cr-fiqa-s"] * 2,
            "aligned_content_sha256": condition[
                "aligned_content_sha256"
            ].tolist(),
        }
    )
    joined = join_fiqa_scores(condition, fiqa)
    assert joined["fiqa_score"].tolist() == [0.1, 0.9]

    mismatched = fiqa.copy()
    mismatched.loc[0, "aligned_content_sha256"] = "wrong-alignment"
    try:
        join_fiqa_scores(condition, mismatched)
    except ValueError as exc:
        assert "alignment hashes differ" in str(exc)
    else:
        raise AssertionError("per-image alignment mismatch must fail closed")

    missing = fiqa.iloc[:1]
    try:
        join_fiqa_scores(condition, missing)
    except ValueError as exc:
        assert "missing 1 FIQA scores" in str(exc)
    else:
        raise AssertionError("incomplete FIQA coverage must fail closed")


def test_test_only_saliency_supports_primary_analysis_but_blocks_calibration():
    calibration = _scores("calibration")
    test = _scores("test")
    saliency = pd.DataFrame(
        {
            "sample_id": test["sample_id"],
            "saliency_target_name": ["origin_top1_gallery_cosine"] * 2,
            "heatmap_available": [True, True],
            "gradcam_valid_heatmap": [True, True],
            "outside_face_attention": [0.2, 0.4],
            "saliency_entropy": [0.7, 0.8],
        }
    )
    readiness = assess_saliency_incremental_readiness(
        calibration,
        test,
        saliency,
    )

    assert readiness.status == "blocked"
    assert readiness.primary_analysis_supported is True
    assert readiness.secondary_calibration_supported is False
    assert readiness.calibration_coverage == 0.0
    assert readiness.test_coverage == 1.0
    assert "leak test information" in " ".join(readiness.reasons)


def test_saliency_readiness_parses_persisted_string_booleans_strictly():
    calibration = _scores("calibration")
    test = _scores("test")
    saliency = pd.DataFrame(
        {
            "sample_id": test["sample_id"],
            "saliency_target_name": ["origin_top1_gallery_cosine"] * 2,
            "heatmap_available": ["True", "False"],
            "gradcam_valid_heatmap": ["True", "False"],
            "outside_face_attention": [0.2, 0.4],
            "saliency_entropy": [0.7, 0.8],
        }
    )

    readiness = assess_saliency_incremental_readiness(
        calibration,
        test,
        saliency,
    )

    assert readiness.test_coverage == 0.5
    assert readiness.secondary_calibration_supported is False


def test_wrong_saliency_target_blocks_primary_and_secondary_uses():
    calibration = _scores("calibration")
    test = _scores("test")
    saliency = pd.DataFrame(
        {
            "sample_id": test["sample_id"],
            "saliency_target_name": ["class_label_target"] * 2,
            "heatmap_available": [True, True],
            "gradcam_valid_heatmap": [True, True],
            "outside_face_attention": [0.2, 0.4],
            "saliency_entropy": [0.7, 0.8],
        }
    )

    readiness = assess_saliency_incremental_readiness(
        calibration,
        test,
        saliency,
    )

    assert readiness.primary_analysis_supported is False
    assert readiness.secondary_calibration_supported is False
    assert "label-free" in " ".join(readiness.reasons)


def test_saliency_readiness_rejects_invalid_coverage_requirement():
    calibration = _scores("calibration")
    test = _scores("test")
    saliency = pd.DataFrame()

    for invalid in (float("nan"), 0.0, -0.1, 1.1):
        try:
            assess_saliency_incremental_readiness(
                calibration,
                test,
                saliency,
                minimum_coverage=invalid,
            )
        except ValueError as exc:
            assert "minimum_coverage" in str(exc)
        else:
            raise AssertionError("invalid coverage requirement must fail closed")


def test_condition_score_artifact_round_trip(tmp_path: Path):
    tables = ConditionScoreTables(
        calibration=_scores("calibration"),
        test=_scores("test"),
        manifest={
            "schema_version": 2,
            "metric_contract": IDENTIFICATION_METRIC_CONTRACT,
            "artifact_type": "compressed_calibration_test_score_tables",
            "status": "completed_in_memory",
            "condition_uid": "condition-test",
        },
    )
    written = write_condition_score_artifact(tmp_path / "condition", tables)
    loaded = load_condition_score_artifact(tmp_path / "condition")

    assert written.condition_uid == "condition-test"
    assert loaded.condition_uid == "condition-test"
    assert len(loaded.calibration) == 2
    assert len(loaded.test) == 2


def _calibration_rows(prefix: str, count: int) -> pd.DataFrame:
    row_ids = list(range(count))
    is_mated = [(index % 2) == 0 for index in row_ids]
    quality = [(index + 0.5) / count for index in row_ids]
    scores = [
        -0.04 - 0.02 * (index % 7) / 6
        if mated
        else -0.12 - 0.18 * quality[index] - 0.01 * (index % 5)
        for index, mated in enumerate(is_mated)
    ]
    return pd.DataFrame(
        {
            "sample_id": [f"{prefix}-{index:04d}" for index in row_ids],
            "identity_id": [
                f"{prefix}-identity-{index // 2:04d}" for index in row_ids
            ],
            "is_mated": is_mated,
            "score": scores,
            "true_identity_score": np.where(is_mated, scores, np.nan),
            "true_identity_rank": np.where(is_mated, 1.0, np.nan),
            "fiqa_score": quality,
            "top_k_correct": is_mated,
            "top_k": [20] * count,
            "evaluation_split": [prefix] * count,
            "dataset_id": ["survface"] * count,
            "model_uid": ["arcface-test"] * count,
            "compression_profile": ["pq_512_m128_b8"] * count,
            "search_mode": ["pq_adc_exhaustive"] * count,
            "score_space": ["negative_squared_l2_adc"] * count,
            "protocol_uid": ["protocol-test"] * count,
            "origin_embedding_artifact_uid": ["origin-test"] * count,
            "extraction_uid": ["population-test"] * count,
            "fiqa_model_uid": ["cr-fiqa-s-test"] * count,
            "aligned_content_sha256": [
                f"aligned-{prefix}-{index:04d}" for index in row_ids
            ],
        }
    )


def _condition_manifest() -> dict:
    return {
        "schema_version": 2,
        "metric_contract": IDENTIFICATION_METRIC_CONTRACT,
        "artifact_type": "compressed_calibration_test_score_tables",
        "status": "completed",
        "condition_uid": "condition-test",
        "dataset_id": "survface",
        "model_uid": "arcface-test",
        "compression_profile": "pq_512_m128_b8",
        "search_mode": "pq_adc_exhaustive",
        "score_space": "negative_squared_l2_adc",
        "protocol_uid": "protocol-test",
        "origin_embedding_artifact_uid": "origin-test",
        "extraction_uid": "population-test",
        "aligned_bundle_manifest_sha256": "a" * 64,
        "top_k": 20,
    }


def _fiqa_manifest() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "fiqa_score_table",
        "status": "completed",
        "fiqa_uid": "fiqa-test",
        "fiqa_model_uid": "cr-fiqa-s-test",
        "dataset_id": "survface",
        "aligned_bundle_manifest_sha256": "a" * 64,
    }


def test_global_vs_fiqa_comparison_and_artifact_round_trip(tmp_path: Path):
    comparison = run_global_vs_fiqa_calibration(
        _calibration_rows("calibration", 400),
        _calibration_rows("test", 200),
        target_fpirs=(0.10,),
        bin_count=2,
        shrinkage_strength=20.0,
        minimum_group_non_mated=5,
        safety_fraction=0.25,
        condition_manifest=_condition_manifest(),
        fiqa_manifest=_fiqa_manifest(),
    )

    assert set(comparison.method_summary["method"]) == {
        "global_empirical",
        "global_safe",
        "fiqa_2bin_conservative_shrunk_safe",
    }
    assert comparison.manifest["threshold_fit_on_test"] is False
    assert comparison.manifest["rank_k"] == 20
    assert comparison.manifest["safety_calibration"]["partition_unit"] == (
        "identity_cluster"
    )
    assert comparison.manifest["saliency_contract"]["included_in_this_comparison"] is False
    assert set(comparison.paired_comparisons["metric"]) == {
        "fpir",
        "tpir_at_rank_k",
    }

    written = write_calibration_comparison_artifact(
        tmp_path / "comparison",
        comparison,
    )
    loaded = load_calibration_comparison_artifact(tmp_path / "comparison")

    assert isinstance(written, CalibrationComparison)
    assert loaded.comparison_uid == comparison.comparison_uid
    assert len(loaded.method_summary) == 3
    assert len(loaded.thresholds) == 4
    assert len(loaded.paired_comparisons) == 4


def test_global_vs_fiqa_requires_nonzero_safety_partition():
    calibration = _calibration_rows("calibration", 40)
    test = _calibration_rows("test", 20)

    try:
        run_global_vs_fiqa_calibration(
            calibration,
            test,
            target_fpirs=(0.10,),
            minimum_group_non_mated=2,
            safety_fraction=0.0,
            condition_manifest=_condition_manifest(),
            fiqa_manifest=_fiqa_manifest(),
        )
    except ValueError as exc:
        assert "requires safety_fraction" in str(exc)
    else:
        raise AssertionError("comparison must keep a held-out safety partition")


def test_global_vs_fiqa_rejects_calibration_test_overlap():
    calibration = _calibration_rows("calibration", 40)
    test = _calibration_rows("test", 20)
    test.loc[0, "sample_id"] = calibration.loc[0, "sample_id"]

    try:
        run_global_vs_fiqa_calibration(
            calibration,
            test,
            target_fpirs=(0.10,),
            minimum_group_non_mated=2,
            safety_fraction=0.25,
            condition_manifest=_condition_manifest(),
            fiqa_manifest=_fiqa_manifest(),
        )
    except ValueError as exc:
        assert "IDs overlap" in str(exc)
    else:
        raise AssertionError("calibration/test overlap must fail closed")


def test_global_vs_fiqa_rejects_row_top_k_manifest_mismatch():
    calibration = _calibration_rows("calibration", 40)
    test = _calibration_rows("test", 20)
    test.loc[0, "top_k"] = 10

    try:
        run_global_vs_fiqa_calibration(
            calibration,
            test,
            target_fpirs=(0.10,),
            minimum_group_non_mated=2,
            safety_fraction=0.25,
            condition_manifest=_condition_manifest(),
            fiqa_manifest=_fiqa_manifest(),
        )
    except ValueError as exc:
        assert "top_k differs from its source manifest" in str(exc)
    else:
        raise AssertionError("row-level top_k mismatch must fail closed")


def test_artifact_join_rejects_different_aligned_bundle_lineage(tmp_path: Path):
    calibration = _calibration_rows("calibration", 4).drop(
        columns=["fiqa_score", "fiqa_model_uid"]
    )
    test = _calibration_rows("test", 4).drop(
        columns=["fiqa_score", "fiqa_model_uid"]
    )
    condition = ConditionScoreTables(
        calibration=calibration,
        test=test,
        manifest=_condition_manifest(),
    )
    scores = pd.concat([calibration, test], ignore_index=True)[
        ["sample_id", "aligned_content_sha256"]
    ]
    scores["fiqa_score"] = 0.5
    scores["fiqa_model_uid"] = "cr-fiqa-s-test"
    matching = FIQAScoreArtifact(
        root=tmp_path,
        scores=scores,
        manifest=_fiqa_manifest(),
    )
    joined_calibration, joined_test = join_fiqa_score_artifacts(
        condition,
        matching,
    )
    assert len(joined_calibration) == 4
    assert len(joined_test) == 4

    fiqa_manifest = _fiqa_manifest()
    fiqa_manifest["aligned_bundle_manifest_sha256"] = "b" * 64
    fiqa = FIQAScoreArtifact(
        root=tmp_path,
        scores=scores,
        manifest=fiqa_manifest,
    )

    try:
        join_fiqa_score_artifacts(condition, fiqa)
    except ValueError as exc:
        assert "aligned-bundle lineage differs" in str(exc)
    else:
        raise AssertionError("different aligned bundles must fail closed")


def _raw_test_scores() -> pd.DataFrame:
    frame = _scores("test").drop(columns=["evaluation_split", "aligned_content_sha256"])
    return frame.rename(columns={
        "sample_id": "query_id", "identity_id": "query_identity_id",
        "score": "compressed_top1_score", "rank1_correct": "compressed_rank1_correct",
        "top_k_correct": "compressed_top_k_correct", "score_space": "compressed_score_space",
        "true_identity_score": "compressed_true_identity_score",
        "true_identity_rank": "compressed_true_identity_rank",
    })


def test_standard_scores_preserves_genuine_score_and_fails_without_it():
    raw = _raw_test_scores()
    raw.loc[0, "compressed_true_identity_score"] = -0.7
    raw.loc[0, "compressed_true_identity_rank"] = 2
    raw.loc[0, "compressed_rank1_correct"] = False
    frame = _standard_scores(raw, split="test", expected_top_k=20)
    assert frame.loc[0, "score"] == -0.1
    assert frame.loc[0, "true_identity_score"] == -0.7
    assert frame.loc[0, "true_identity_rank"] == 2
    with pytest.raises(ValueError, match="missing columns"):
        _standard_scores(raw.drop(columns="compressed_true_identity_score"), split="test", expected_top_k=20)


def test_adc_genuine_rank_uses_first_matching_identity_and_handles_rank_exit():
    frame = _adc_condition_score_frame(
        np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
        np.array([[0, 1, 2], [0, 1, 2]]),
        query_ids=np.array(["q-a", "q-c"]),
        query_identity_ids=np.array(["a", "c"]),
        gallery_identity_ids=np.array(["b", "a", "a", "c"]),
        compression_profile="pq", search_mode="pq_adc_exhaustive",
    )
    assert frame["is_mated"].tolist() == [True, True]
    assert frame.loc[0, "compressed_true_identity_rank"] == 2
    assert frame.loc[0, "compressed_true_identity_score"] == -0.2
    assert not frame.loc[1, "compressed_top_k_correct"]
    assert np.isnan(frame.loc[1, "compressed_true_identity_score"])


@pytest.mark.parametrize("artifact_type,loader", [
    ("compressed_calibration_test_score_tables", load_condition_score_artifact),
    ("global_vs_fiqa_threshold_calibration", load_calibration_comparison_artifact),
])
def test_legacy_artifacts_are_rejected_before_reuse(tmp_path, artifact_type, loader):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "artifact_type": artifact_type, "status": "completed",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy/incompatible FIQA metric artifact"):
        loader(tmp_path)


def _legacy_fixture(tmp_path):
    run = tmp_path / "run"
    workflow = run / "artifacts" / "step2_workflow"
    ledger = workflow / "retrieval_ledger"
    ledger.mkdir(parents=True)
    (run / "COMPLETED").touch()
    (run / "run_manifest.json").write_text(json.dumps({
        "run_id": "run-test", "status": "completed",
        "config": {"dataset_id": "survface", "model_uid": "arcface-test"},
    }), encoding="utf-8")
    (workflow / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    raw = _raw_test_scores()
    raw.to_parquet(ledger / "core.parquet", index=False)
    core_hash = sha256_file(ledger / "core.parquet")
    (ledger / "manifest.json").write_text(json.dumps({"conditions": [{
        "condition": {"compression_profile": "pq_512_m128_b8", "search_mode": "pq_adc_exhaustive", "evaluation_split": "test"},
        "core": {"path": "core.parquet", "sha256": core_hash},
    }]}), encoding="utf-8")
    legacy = tmp_path / "v1"
    legacy.mkdir()
    manifest = _condition_manifest()
    manifest.update({
        "schema_version": 1, "source_run_id": "run-test",
        "source_run_manifest_sha256": sha256_file(run / "run_manifest.json"),
        "source_freeze_manifest_sha256": sha256_file(workflow / "freeze_manifest.json"),
        "persisted_test_core_sha256": core_hash,
    })
    manifest.pop("metric_contract")
    manifest["files"] = {}
    for split in ("calibration", "test"):
        frame = _scores(split).drop(columns=["true_identity_score", "true_identity_rank"])
        if split == "calibration":
            frame["identity_id"] = "cal-" + frame["identity_id"]
        name = f"{split}_scores.parquet"
        frame.to_parquet(legacy / name, index=False)
        manifest["files"][name] = {"sha256": sha256_file(legacy / name), "row_count": len(frame)}
    (legacy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return legacy, run


def test_upgrade_reuses_calibration_and_preserves_v1_artifacts(tmp_path):
    legacy, run = _legacy_fixture(tmp_path)
    before = {path.name: sha256_file(path) for path in legacy.iterdir()}
    upgraded = upgrade_condition_score_artifact(legacy, run)
    assert upgraded.manifest["schema_version"] == 2
    assert upgraded.manifest["upgrade_provenance"]["search_replayed"] is False
    assert "true_identity_score" not in upgraded.calibration
    assert upgraded.test.loc[0, "true_identity_score"] == -0.1
    loaded = write_condition_score_artifact(tmp_path / "v2", upgraded)
    pd.testing.assert_frame_equal(loaded.test, upgraded.test)
    assert before == {path.name: sha256_file(path) for path in legacy.iterdir()}


def test_upgrade_rejects_modified_cached_maxima_even_with_updated_file_hash(tmp_path):
    legacy, run = _legacy_fixture(tmp_path)
    path = legacy / "test_scores.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "score"] = -0.05
    frame.to_parquet(path, index=False)
    manifest_path = legacy / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][path.name]["sha256"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="scores/labels differ"):
        upgrade_condition_score_artifact(legacy, run)


def test_comparison_corrects_summary_and_paired_tpir_not_thresholds():
    calibration = _calibration_rows("calibration", 400)
    test = _calibration_rows("test", 200)
    kwargs = dict(target_fpirs=(0.1,), minimum_group_non_mated=5,
                  condition_manifest=_condition_manifest(), fiqa_manifest=_fiqa_manifest())
    before = run_global_vs_fiqa_calibration(calibration, test, **kwargs)
    test.loc[test["is_mated"], "true_identity_score"] = -10.0
    after = run_global_vs_fiqa_calibration(calibration, test, **kwargs)
    pd.testing.assert_frame_equal(before.thresholds, after.thresholds)
    assert after.method_summary["tpir_at_rank_k"].eq(0).all()
    assert before.method_summary["realized_fpir"].equals(after.method_summary["realized_fpir"])
    paired = after.paired_comparisons.query("metric == 'tpir_at_rank_k'")
    assert paired["reference_successes"].eq(0).all()
    assert paired["candidate_successes"].eq(0).all()
