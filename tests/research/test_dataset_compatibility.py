from __future__ import annotations

import pytest

from research.datasets import (
    DatasetIntegrityError,
    audit_source_identity_overlap,
    check_rfw_checkpoint_eligibility,
    require_no_source_identity_overlap,
    require_rfw_checkpoint_eligibility,
    validate_dataset_operation,
)


def test_source_identity_overlap_uses_unprefixed_provider_ids():
    audit = audit_source_identity_overlap(
        {"m.one", "m.two"},
        {"m.two", "m.three"},
    )

    assert audit.overlap_identities == ("m.two",)
    assert audit.overlap_identity_count == 1
    with pytest.raises(DatasetIntegrityError, match="source identity overlap"):
        require_no_source_identity_overlap(audit)


def test_ms1mv2_checkpoint_fails_closed_for_rfw():
    result = check_rfw_checkpoint_eligibility(
        "arcface",
        training_dataset="ms1mv2",
        training_dataset_verified=True,
    )

    assert result.eligible is False
    with pytest.raises(DatasetIntegrityError, match="RFW checkpoint eligibility"):
        require_rfw_checkpoint_eligibility(
            "arcface",
            training_dataset="ms1mv2",
            training_dataset_verified=True,
        )


def test_documented_overlap_removed_checkpoint_can_pass_rfw_gate():
    result = require_rfw_checkpoint_eligibility(
        "controlled-model",
        training_dataset="MS1M_wo_RFW",
        training_dataset_verified=True,
        overlap_control={
            "status": "official_ms1m_wo_rfw",
            "evidence": "provider dataset card and checkpoint manifest",
        },
    )

    assert result.eligible is True


def test_non_ms1m_checkpoint_without_disjointness_evidence_fails_closed():
    result = check_rfw_checkpoint_eligibility(
        "vggface2-model",
        training_dataset="vggface2",
        training_dataset_verified=True,
    )

    assert result.eligible is False
    assert "non-overlap evidence" in result.reason


def test_unknown_checkpoint_and_invalid_dataset_roles_are_rejected():
    assert (
        check_rfw_checkpoint_eligibility(
            "unknown",
            training_dataset=None,
            training_dataset_verified=False,
        ).eligible
        is False
    )
    with pytest.raises(DatasetIntegrityError):
        validate_dataset_operation("rfw-v1", "pca_fit")
    with pytest.raises(DatasetIntegrityError):
        validate_dataset_operation(
            "bupt-balancedface-equalizedface",
            "final_evaluation",
        )


def test_checkpoint_gate_rejects_string_boolean_and_null_evidence():
    string_boolean = check_rfw_checkpoint_eligibility(
        "bad-types",
        training_dataset="vggface2",
        training_dataset_verified="false",  # type: ignore[arg-type]
        overlap_control={
            "status": "verified_non_overlap_training_source",
            "evidence": "document",
        },
    )
    null_evidence = check_rfw_checkpoint_eligibility(
        "null-evidence",
        training_dataset="vggface2",
        training_dataset_verified=True,
        overlap_control={
            "status": "verified_non_overlap_training_source",
            "evidence": None,  # type: ignore[dict-item]
        },
    )

    assert string_boolean.eligible is False
    assert null_evidence.eligible is False
