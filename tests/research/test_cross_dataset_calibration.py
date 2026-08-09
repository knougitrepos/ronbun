from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from research.experiments.cross_dataset_calibration import (
    CalibrationThresholdArtifact,
    CalibrationTransferCompatibilityError,
    CodecProvenance,
    IdentityOverlapAudit,
    MAXIMUM_GALLERY_SCORE,
    OPEN_SET_1_TO_N,
    PAIR_SCORE,
    PAIR_VERIFICATION_1_TO_1,
    ProtocolScoreContract,
    RFW_EXTERNAL_DIAGNOSTIC,
    RFW_INTERNAL_9FOLD,
    RFW_OFFICIAL_PAIR_ROLE,
    SAME_DOMAIN_DIAGNOSTIC,
    STRICT_EXTERNAL_TRANSFER,
    VerifiedArtifact,
    build_cross_dataset_calibration_plan,
    calibration_evaluation_manifest,
    evaluate_external_calibration_transfer,
    evaluate_rfw_official_internal_baseline,
)
from research.runtime.hashing import sha256_file


def _artifact(
    root: Path,
    name: str,
    artifact_type: str,
    *,
    artifact_uid: str | None = None,
    **extra: object,
) -> VerifiedArtifact:
    path = root / f"{name}.json"
    payload = {
        "schema_version": 1,
        "status": "completed",
        "artifact_type": artifact_type,
        **extra,
    }
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return VerifiedArtifact(
        path=path,
        sha256=sha256_file(path),
        artifact_type=artifact_type,
        artifact_uid=artifact_uid or name,
    )


def _contract(
    root: Path,
    name: str,
    *,
    dataset_id: str,
    physical_dataset_id: str,
    population_uid: str,
    protocol_kind: str = OPEN_SET_1_TO_N,
    protocol_role: str = "custom_open_set",
    score_statistic: str = MAXIMUM_GALLERY_SCORE,
    gallery_size: int | None = 100,
    score_space: str = "cosine_similarity",
    checkpoint_training_identity_overlap_status: str = "UNKNOWN",
) -> ProtocolScoreContract:
    model_uid = "edgeface-test"
    codec = CodecProvenance.origin(model_uid=model_uid)
    run_id = f"{name}-run"
    protocol_uid = f"{name}-protocol"
    score_artifact = _artifact(
        root,
        f"{name}-scores",
        "score_rows",
        dataset_id=dataset_id,
        physical_dataset_id=physical_dataset_id,
        source_population_uid=population_uid,
        run_id=run_id,
        model_uid=model_uid,
        protocol_uid=protocol_uid,
        protocol_kind=protocol_kind,
        protocol_role=protocol_role,
        score_statistic=score_statistic,
        search_mode="reconstruction_cosine",
        score_space=score_space,
        checkpoint_training_identity_overlap_status=(
            checkpoint_training_identity_overlap_status
        ),
        gallery_size=gallery_size,
        codec_fingerprint=codec.fingerprint(),
    )
    return ProtocolScoreContract(
        dataset_id=dataset_id,
        physical_dataset_id=physical_dataset_id,
        source_population_uid=population_uid,
        run_id=run_id,
        model_uid=model_uid,
        protocol_uid=protocol_uid,
        protocol_kind=protocol_kind,
        protocol_role=protocol_role,
        score_statistic=score_statistic,
        search_mode="reconstruction_cosine",
        score_space=score_space,
        checkpoint_training_identity_overlap_status=(
            checkpoint_training_identity_overlap_status
        ),
        score_artifact=score_artifact,
        codec=codec,
        gallery_size=gallery_size,
    )


def _calibration(
    root: Path,
    name: str,
    source: ProtocolScoreContract,
    *,
    threshold: float = 0.5,
) -> CalibrationThresholdArtifact:
    operating_point_kind = (
        "fpir" if source.protocol_kind == OPEN_SET_1_TO_N else "far"
    )
    manifest = _artifact(
        root,
        f"{name}-calibration",
        "calibration_threshold",
        calibration_uid=name,
        calibration_source_dataset_ids=[source.dataset_id],
        model_uid=source.model_uid,
        protocol_kind=source.protocol_kind,
        score_statistic=source.score_statistic,
        search_mode=source.search_mode,
        score_space=source.score_space,
        gallery_size=source.gallery_size,
        codec_fingerprint=source.codec.fingerprint(),
        threshold=threshold,
        threshold_comparator=">=",
        threshold_policy="frozen_source_threshold",
        operating_point_kind=operating_point_kind,
        operating_point_value=0.1,
    )
    return CalibrationThresholdArtifact(
        calibration_uid=name,
        source_contracts=(source,),
        threshold=threshold,
        threshold_comparator=">=",
        threshold_policy="frozen_source_threshold",
        operating_point_kind=operating_point_kind,
        operating_point_value=0.1,
        manifest=manifest,
    )


def _overlap_audit(
    root: Path,
    source: ProtocolScoreContract,
    target: ProtocolScoreContract,
    status: str,
) -> IdentityOverlapAudit:
    artifact = None
    if status != "unknown":
        artifact = _artifact(
            root,
            f"overlap-{source.source_population_uid}-{target.source_population_uid}",
            "identity_overlap_audit",
            source_population_uid=source.source_population_uid,
            target_population_uid=target.source_population_uid,
            identity_overlap_status=status,
        )
    return IdentityOverlapAudit(
        source_population_uid=source.source_population_uid,
        target_population_uid=target.source_population_uid,
        status=status,
        audit_artifact=artifact,
    )


def _pair_contract(
    root: Path,
    name: str,
    *,
    dataset_id: str,
    physical_dataset_id: str,
    population_uid: str,
    role: str,
) -> ProtocolScoreContract:
    return _contract(
        root,
        name,
        dataset_id=dataset_id,
        physical_dataset_id=physical_dataset_id,
        population_uid=population_uid,
        protocol_kind=PAIR_VERIFICATION_1_TO_1,
        protocol_role=role,
        score_statistic=PAIR_SCORE,
        gallery_size=None,
    )


def test_plan_separates_calibration_sources_from_target_dataset_ids(tmp_path: Path):
    lfw_source = _contract(
        tmp_path,
        "lfw-source",
        dataset_id="lfw",
        physical_dataset_id="lfw-v1",
        population_uid="lfw-calibration-identities-v1",
    )
    survface_target = _contract(
        tmp_path,
        "survface-target",
        dataset_id="survface",
        physical_dataset_id="qmul-survface-v1",
        population_uid="survface-official-test-v1",
    )
    unselected = _contract(
        tmp_path,
        "rfw-custom-target",
        dataset_id="rfw_custom",
        physical_dataset_id="rfw-v1",
        population_uid="rfw-custom-open-set-v1",
    )
    calibration = _calibration(tmp_path, "lfw-threshold", lfw_source)
    audit = _overlap_audit(
        tmp_path,
        lfw_source,
        survface_target,
        "disjoint_verified",
    )

    plan = build_cross_dataset_calibration_plan(
        calibration_sources=(calibration,),
        targets=(survface_target, unselected),
        target_dataset_ids=("survface",),
        identity_overlap_audits=(audit,),
    )

    assert plan.calibration_source_dataset_ids == ("lfw",)
    assert plan.target_dataset_ids == ("survface",)
    assert len(plan.cases) == 1
    case = plan.as_dict()["cases"][0]
    assert case["transfer_scope"] == STRICT_EXTERNAL_TRANSFER
    assert case["strict_external_transfer_eligible"] is True
    assert case["calibration"]["calibration_source_dataset_ids"] == ["lfw"]
    assert case["target"]["dataset_id"] == "survface"
    assert case["target"]["codec"]["profile_name"] == "origin_512"


def test_open_set_maximum_score_cannot_transfer_to_rfw_pair_score(tmp_path: Path):
    source = _contract(
        tmp_path,
        "lfw-open-source",
        dataset_id="lfw",
        physical_dataset_id="lfw-v1",
        population_uid="lfw-open-calibration-v1",
    )
    target = _pair_contract(
        tmp_path,
        "rfw-official-target",
        dataset_id="rfw",
        physical_dataset_id="rfw-v1",
        population_uid="rfw-official-test-v1",
        role=RFW_OFFICIAL_PAIR_ROLE,
    )
    calibration = _calibration(tmp_path, "lfw-open-threshold", source)
    audit = _overlap_audit(tmp_path, source, target, "disjoint_verified")

    with pytest.raises(
        CalibrationTransferCompatibilityError,
        match="maximum-gallery scores and 1:1 pair scores",
    ):
        build_cross_dataset_calibration_plan(
            calibration_sources=(calibration,),
            targets=(target,),
            target_dataset_ids=("rfw",),
            identity_overlap_audits=(audit,),
        )


def test_strict_external_transfer_rejects_unknown_identity_overlap(tmp_path: Path):
    source = _pair_contract(
        tmp_path,
        "lfw-pair-source",
        dataset_id="lfw",
        physical_dataset_id="lfw-v1",
        population_uid="lfw-pairs-v1",
        role="lfw_official_pairs",
    )
    target = _pair_contract(
        tmp_path,
        "rfw-pair-target",
        dataset_id="rfw",
        physical_dataset_id="rfw-v1",
        population_uid="rfw-official-test-v1",
        role=RFW_OFFICIAL_PAIR_ROLE,
    )
    calibration = _calibration(tmp_path, "lfw-pair-threshold", source)
    unknown = _overlap_audit(tmp_path, source, target, "unknown")

    with pytest.raises(
        CalibrationTransferCompatibilityError,
        match="verified identity disjointness",
    ):
        build_cross_dataset_calibration_plan(
            calibration_sources=(calibration,),
            targets=(target,),
            target_dataset_ids=("rfw",),
            identity_overlap_audits=(unknown,),
        )

    with pytest.raises(
        CalibrationTransferCompatibilityError,
        match="verified identity disjointness",
    ):
        build_cross_dataset_calibration_plan(
            calibration_sources=(calibration,),
            targets=(target,),
            target_dataset_ids=("rfw",),
            identity_overlap_audits=(unknown,),
            allow_same_domain_diagnostic=True,
        )


def test_rfw_shared_population_is_only_named_same_domain_diagnostic(tmp_path: Path):
    source = _pair_contract(
        tmp_path,
        "rfw-custom-pair-source",
        dataset_id="rfw_custom",
        physical_dataset_id="rfw-v1",
        population_uid="rfw-test-images-and-identities-v1",
        role="rfw_custom_pair_calibration",
    )
    target = _pair_contract(
        tmp_path,
        "rfw-official-pair-target",
        dataset_id="rfw",
        physical_dataset_id="rfw-v1",
        population_uid="rfw-test-images-and-identities-v1",
        role=RFW_OFFICIAL_PAIR_ROLE,
    )
    calibration = _calibration(tmp_path, "rfw-custom-threshold", source)
    overlap = _overlap_audit(tmp_path, source, target, "overlap_verified")

    with pytest.raises(
        CalibrationTransferCompatibilityError,
        match="different physical datasets",
    ):
        build_cross_dataset_calibration_plan(
            calibration_sources=(calibration,),
            targets=(target,),
            target_dataset_ids=("rfw",),
            identity_overlap_audits=(overlap,),
        )

    plan = build_cross_dataset_calibration_plan(
        calibration_sources=(calibration,),
        targets=(target,),
        target_dataset_ids=("rfw",),
        identity_overlap_audits=(overlap,),
        allow_same_domain_diagnostic=True,
    )
    case = plan.cases[0]
    assert case.transfer_scope == SAME_DOMAIN_DIAGNOSTIC
    assert case.as_dict()["evaluation_mode"] == SAME_DOMAIN_DIAGNOSTIC
    assert case.as_dict()["strict_external_transfer_eligible"] is False


def test_rfw_internal_baseline_and_external_diagnostic_are_distinct(tmp_path: Path):
    source = _pair_contract(
        tmp_path,
        "lfw-pair-calibration",
        dataset_id="lfw",
        physical_dataset_id="lfw-v1",
        population_uid="lfw-pair-calibration-v1",
        role="lfw_official_pairs",
    )
    target = _pair_contract(
        tmp_path,
        "rfw-official",
        dataset_id="rfw",
        physical_dataset_id="rfw-v1",
        population_uid="rfw-official-test-v1",
        role=RFW_OFFICIAL_PAIR_ROLE,
    )
    calibration = _calibration(tmp_path, "lfw-fixed-pair-threshold", source)
    audit = _overlap_audit(tmp_path, source, target, "disjoint_verified")
    plan = build_cross_dataset_calibration_plan(
        calibration_sources=(calibration,),
        targets=(target,),
        target_dataset_ids=("rfw",),
        identity_overlap_audits=(audit,),
    )
    frame = pd.DataFrame(
        {
            "pair_id": ["a0g", "a0i", "a1g", "a1i"],
            "rfw_group": ["African"] * 4,
            "fold_index": [0, 0, 1, 1],
            "left_image_id": ["a", "c", "e", "g"],
            "right_image_id": ["b", "d", "f", "h"],
            "is_genuine": [True, False, True, False],
            "score": [0.9, 0.1, 0.8, 0.2],
        }
    )

    external = evaluate_external_calibration_transfer(
        plan.cases[0],
        frame,
        strict_rfw_official=False,
    )
    internal = evaluate_rfw_official_internal_baseline(
        target,
        frame.drop(columns="score"),
        scores=frame["score"],
        thresholds=(-0.5, 0.5, 1.0),
        strict_official=False,
        bootstrap_repeats=100,
    )

    assert external.evaluation_mode == RFW_EXTERNAL_DIAGNOSTIC
    assert external.summary["threshold_fit_on_target"] is False
    assert external.summary["official_internal_9fold_baseline"] is False
    assert (
        external.summary["checkpoint_training_identity_overlap_status"]
        == "UNKNOWN"
    )
    assert external.summary["strict_unseen_identity_evidence"] is False
    assert external.summary["eer"] == pytest.approx(0.0)
    assert external.summary["eer_threshold_source"] == (
        "target_pair_scores_and_labels"
    )
    assert external.group_summary["eer"].eq(0.0).all()
    assert internal.evaluation_mode == RFW_INTERNAL_9FOLD
    assert internal.summary["threshold_fit_on_target"] is True
    assert internal.summary["official_internal_9fold_baseline"] is True
    assert (
        internal.summary["checkpoint_training_identity_overlap_status"]
        == "UNKNOWN"
    )
    assert internal.summary["strict_unseen_identity_evidence"] is False
    assert internal.summary["eer_threshold_source"] == (
        "heldout_fold_scores_and_labels"
    )
    manifest = calibration_evaluation_manifest(external)
    assert manifest["summary"]["calibration_source_dataset_ids"] == ["lfw"]
    assert manifest["summary"]["target_dataset_id"] == "rfw"


def test_content_addressed_score_lineage_fails_after_tampering(tmp_path: Path):
    source = _contract(
        tmp_path,
        "tampered-source",
        dataset_id="lfw",
        physical_dataset_id="lfw-v1",
        population_uid="lfw-calibration-v1",
    )
    source.score_artifact.path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        source.verify()


def test_codec_source_and_artifact_sha_are_verified_fail_closed(tmp_path: Path):
    codec_path = tmp_path / "pca_128.pkl"
    codec_path.write_bytes(b"frozen-pca-codec")
    codec_sha = sha256_file(codec_path)
    codec_artifact = VerifiedArtifact(
        path=codec_path,
        sha256=codec_sha,
        artifact_type="frozen_pca_codec",
        artifact_uid="pca-codec-uid",
    )
    codec_manifest = _artifact(
        tmp_path,
        "codec-manifest",
        "frozen_compression_codec_bundle",
        fit_source_dataset="lfw",
        fit_source_run_id="lfw-codec-run",
        model_uid="edgeface-test",
        fit_on_rfw=False,
        codecs=[
            {
                "family": "pca",
                "profile_name": "pca_128",
                "artifact_sha256": codec_sha,
            }
        ],
    )
    codec = CodecProvenance(
        family="pca",
        profile_name="pca_128",
        model_uid="edgeface-test",
        fit_source_dataset_id="lfw",
        fit_source_run_id="lfw-codec-run",
        manifest=codec_manifest,
        artifact=codec_artifact,
    )

    codec.verify()
    assert codec.as_dict()["fit_source_dataset_id"] == "lfw"
    codec_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        codec.verify()
