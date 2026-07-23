from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from research.datasets.sources import DatasetIntegrityError


_MS1M_PATTERN = re.compile(
    r"(?:ms[\s_-]?celeb[\s_-]?1m|ms1m|ms1mv[123])",
    flags=re.IGNORECASE,
)
@dataclass(frozen=True)
class IdentityOverlapAudit:
    """Exact source-identity overlap before dataset-specific prefixing."""

    left_identity_count: int
    right_identity_count: int
    overlap_identities: tuple[str, ...]
    overlap_sha256: str

    @property
    def overlap_identity_count(self) -> int:
        return len(self.overlap_identities)


@dataclass(frozen=True)
class RFWCheckpointEligibility:
    """Fail-closed result for using a checkpoint on RFW."""

    checkpoint_name: str
    training_dataset: str | None
    eligible: bool
    overlap_control_status: str
    reason: str


def audit_source_identity_overlap(
    left_identity_ids: Iterable[str],
    right_identity_ids: Iterable[str],
) -> IdentityOverlapAudit:
    """Compare raw provider identity IDs, not dataset-prefixed IDs."""

    left = {str(value).strip() for value in left_identity_ids}
    right = {str(value).strip() for value in right_identity_ids}
    if "" in left or "" in right:
        raise ValueError("source identity IDs must not be empty")
    overlap = tuple(sorted(left.intersection(right)))
    payload = "\n".join(overlap).encode("utf-8")
    return IdentityOverlapAudit(
        left_identity_count=len(left),
        right_identity_count=len(right),
        overlap_identities=overlap,
        overlap_sha256=hashlib.sha256(payload).hexdigest().upper(),
    )


def require_no_source_identity_overlap(audit: IdentityOverlapAudit) -> None:
    """Reject a leakage-sensitive join until overlapping identities are removed."""

    if audit.overlap_identity_count:
        raise DatasetIntegrityError(
            "source identity overlap detected: "
            f"count={audit.overlap_identity_count}, sha256={audit.overlap_sha256}"
        )


def check_rfw_checkpoint_eligibility(
    checkpoint_name: str,
    *,
    training_dataset: str | None,
    training_dataset_verified: bool,
    overlap_control: Mapping[str, str] | None = None,
) -> RFWCheckpointEligibility:
    """Determine whether RFW may be used as a headline checkpoint evaluation.

    Unknown provenance and MS1M-family training fail closed. An MS1M-family
    checkpoint is allowed only with a documented provider-side overlap-control
    status and non-empty evidence reference.
    """

    raw_status = (overlap_control or {}).get("status", "none")
    raw_evidence = (overlap_control or {}).get("evidence", "")
    status = raw_status.strip() if isinstance(raw_status, str) else "invalid"
    evidence = raw_evidence.strip() if isinstance(raw_evidence, str) else ""
    valid_training_dataset = (
        isinstance(training_dataset, str) and bool(training_dataset.strip())
    )
    if not valid_training_dataset or training_dataset_verified is not True:
        return RFWCheckpointEligibility(
            checkpoint_name=checkpoint_name,
            training_dataset=training_dataset,
            eligible=False,
            overlap_control_status=status,
            reason="training dataset provenance is missing or unverified",
        )

    assert isinstance(training_dataset, str)
    is_ms1m_family = _MS1M_PATTERN.search(training_dataset) is not None
    if is_ms1m_family:
        if (
            status
            not in {"official_ms1m_wo_rfw", "verified_rfw_identity_removal"}
            or not evidence
        ):
            return RFWCheckpointEligibility(
                checkpoint_name=checkpoint_name,
                training_dataset=training_dataset,
                eligible=False,
                overlap_control_status=status,
                reason=(
                    "MS1M-family checkpoint has no verified RFW identity-removal evidence"
                ),
            )
        return RFWCheckpointEligibility(
            checkpoint_name=checkpoint_name,
            training_dataset=training_dataset,
            eligible=True,
            overlap_control_status=status,
            reason="documented MS1M/RFW overlap control is present",
        )

    if status != "verified_non_overlap_training_source" or not evidence:
        return RFWCheckpointEligibility(
            checkpoint_name=checkpoint_name,
            training_dataset=training_dataset,
            eligible=False,
            overlap_control_status=status,
            reason=(
                "non-MS1M training source has no documented RFW non-overlap evidence"
            ),
        )
    return RFWCheckpointEligibility(
        checkpoint_name=checkpoint_name,
        training_dataset=training_dataset,
        eligible=True,
        overlap_control_status=status,
        reason="documented non-MS1M/RFW non-overlap evidence is present",
    )


def require_rfw_checkpoint_eligibility(
    checkpoint_name: str,
    *,
    training_dataset: str | None,
    training_dataset_verified: bool,
    overlap_control: Mapping[str, str] | None = None,
) -> RFWCheckpointEligibility:
    """Return the eligibility result or raise before an RFW headline run."""

    result = check_rfw_checkpoint_eligibility(
        checkpoint_name,
        training_dataset=training_dataset,
        training_dataset_verified=training_dataset_verified,
        overlap_control=overlap_control,
    )
    if not result.eligible:
        raise DatasetIntegrityError(
            f"RFW checkpoint eligibility failed for {checkpoint_name}: {result.reason}"
        )
    return result


def validate_dataset_operation(dataset: str, operation: str) -> None:
    """Enforce the current research roles for RFW and BalancedFace."""

    normalized_dataset = dataset.strip().lower()
    normalized_operation = operation.strip().lower()
    allowed = {
        "rfw-v1": {
            "source_inventory",
            "protocol_manifest",
            "image_materialization",
            "embedding_extraction",
            "frozen_compression_apply",
            "pair_scoring",
            "external_threshold_evaluation",
            "diagnostic_9fold_threshold_fit",
            "official_10fold_evaluation",
        },
        "bupt-balancedface-equalizedface": {
            "source_inventory",
            "source_index",
            "identity_overlap_exclusion",
            "image_materialization",
            "embedding_extraction",
            "compressor_fit",
            "pca_fit",
            "pq_fit",
            "threshold_calibration",
        },
    }
    if normalized_dataset not in allowed:
        raise DatasetIntegrityError(f"unknown dataset role policy: {dataset!r}")
    if normalized_operation not in allowed[normalized_dataset]:
        raise DatasetIntegrityError(
            f"operation {operation!r} is not allowed for dataset {dataset!r}"
        )
