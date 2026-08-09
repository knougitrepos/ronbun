from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np
import pandas as pd

from research.datasets.rfw import (
    RFW_GROUPS,
    RFW_OFFICIAL_FOLD_COUNT,
    RFW_OFFICIAL_GENUINE_PER_FOLD,
)
from research.evaluation.rfw_verification import (
    empirical_equal_error_rate,
    evaluate_rfw_pair_scores,
)
from research.runtime.hashing import canonical_sha256, sha256_file


ProtocolKind = Literal["open_set_1toN", "pair_verification_1to1"]
ScoreStatistic = Literal["maximum_gallery_score", "pair_score"]
OperatingPointKind = Literal["fpir", "far", "accuracy"]
IdentityOverlapStatus = Literal["disjoint_verified", "overlap_verified", "unknown"]
CheckpointTrainingIdentityOverlapStatus = Literal[
    "DISJOINT_VERIFIED",
    "OVERLAP_VERIFIED",
    "UNKNOWN",
]

OPEN_SET_1_TO_N = "open_set_1toN"
PAIR_VERIFICATION_1_TO_1 = "pair_verification_1to1"
MAXIMUM_GALLERY_SCORE = "maximum_gallery_score"
PAIR_SCORE = "pair_score"
RFW_OFFICIAL_PAIR_ROLE = "rfw_official_pairs"
EXTERNAL_FIXED_THRESHOLD = "external_frozen_threshold"
RFW_INTERNAL_9FOLD = "rfw_official_internal_9fold"
RFW_EXTERNAL_DIAGNOSTIC = "rfw_official_external_threshold_diagnostic"
STRICT_EXTERNAL_TRANSFER = "strict_external_transfer"
SAME_DOMAIN_DIAGNOSTIC = "same_domain_overlap_diagnostic"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_COMPARATOR = ">="

__all__ = [
    "CalibrationEvaluation",
    "CalibrationThresholdArtifact",
    "CalibrationTransferCase",
    "CalibrationTransferCompatibilityError",
    "CodecProvenance",
    "CrossDatasetCalibrationPlan",
    "EXTERNAL_FIXED_THRESHOLD",
    "IdentityOverlapAudit",
    "MAXIMUM_GALLERY_SCORE",
    "OPEN_SET_1_TO_N",
    "PAIR_SCORE",
    "PAIR_VERIFICATION_1_TO_1",
    "ProtocolScoreContract",
    "RFW_EXTERNAL_DIAGNOSTIC",
    "RFW_INTERNAL_9FOLD",
    "RFW_OFFICIAL_PAIR_ROLE",
    "SAME_DOMAIN_DIAGNOSTIC",
    "STRICT_EXTERNAL_TRANSFER",
    "VerifiedArtifact",
    "build_cross_dataset_calibration_plan",
    "calibration_evaluation_manifest",
    "evaluate_external_calibration_transfer",
    "evaluate_rfw_official_internal_baseline",
    "validate_calibration_transfer_compatibility",
]


class CalibrationTransferCompatibilityError(ValueError):
    """Raised before an incompatible calibration threshold can be applied."""


def _required_text(value: str, *, field: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} must be a non-empty string")
    return normalized


def _dataset_id(value: str, *, field: str = "dataset_id") -> str:
    return _required_text(value, field=field).lower()


def _sha256(value: str, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return normalized


def _content_identity(value: Any) -> Any:
    """Remove location-only fields before deriving stable scientific UIDs."""

    if isinstance(value, dict):
        return {
            key: _content_identity(item)
            for key, item in value.items()
            if key != "path"
        }
    if isinstance(value, list):
        return [_content_identity(item) for item in value]
    return value


@dataclass(frozen=True)
class VerifiedArtifact:
    """A content-addressed completed artifact used by a transfer contract."""

    path: Path
    sha256: str
    artifact_type: str
    artifact_uid: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        object.__setattr__(
            self,
            "sha256",
            _sha256(self.sha256, field="artifact sha256"),
        )
        object.__setattr__(
            self,
            "artifact_type",
            _required_text(self.artifact_type, field="artifact_type"),
        )
        object.__setattr__(
            self,
            "artifact_uid",
            _required_text(self.artifact_uid, field="artifact_uid"),
        )

    def verify(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        actual_sha256 = sha256_file(self.path)
        if actual_sha256 != self.sha256:
            raise ValueError(
                f"artifact SHA-256 mismatch: {self.path}; "
                f"expected={self.sha256}, actual={actual_sha256}"
            )
        if self.path.suffix.lower() != ".json":
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"artifact manifest must contain a JSON object: {self.path}")
        if payload.get("status") != "completed":
            raise ValueError(f"artifact manifest is not completed: {self.path}")
        if payload.get("artifact_type") != self.artifact_type:
            raise ValueError(
                "artifact manifest type mismatch: "
                f"expected={self.artifact_type!r}, "
                f"actual={payload.get('artifact_type')!r}"
            )
        return payload

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "artifact_type": self.artifact_type,
            "artifact_uid": self.artifact_uid,
        }


@dataclass(frozen=True)
class IdentityOverlapAudit:
    """Pairwise identity-overlap evidence for two physical populations."""

    source_population_uid: str
    target_population_uid: str
    status: IdentityOverlapStatus
    audit_artifact: VerifiedArtifact | None = None

    def __post_init__(self) -> None:
        for field in ("source_population_uid", "target_population_uid"):
            object.__setattr__(
                self,
                field,
                _required_text(str(getattr(self, field)), field=field),
            )
        if self.status not in {"disjoint_verified", "overlap_verified", "unknown"}:
            raise ValueError(f"unsupported identity overlap status: {self.status!r}")
        if self.status in {"disjoint_verified", "overlap_verified"}:
            if self.audit_artifact is None:
                raise ValueError(
                    f"{self.status} requires a content-addressed audit artifact"
                )

    def verify(self) -> None:
        if self.audit_artifact is not None:
            if self.audit_artifact.artifact_type != "identity_overlap_audit":
                raise ValueError(
                    "identity overlap evidence must use identity_overlap_audit type"
                )
            payload = self.audit_artifact.verify()
            if payload is None:
                raise ValueError("identity overlap audit must be a JSON manifest")
            expected = {
                "source_population_uid": self.source_population_uid,
                "target_population_uid": self.target_population_uid,
                "identity_overlap_status": self.status,
            }
            mismatches = {
                key: {"expected": value, "actual": payload.get(key)}
                for key, value in expected.items()
                if payload.get(key) != value
            }
            if mismatches:
                raise ValueError(f"identity overlap audit mismatch: {mismatches}")

    def as_dict(self) -> dict[str, object]:
        return {
            "source_population_uid": self.source_population_uid,
            "target_population_uid": self.target_population_uid,
            "status": self.status,
            "audit_artifact": (
                None
                if self.audit_artifact is None
                else self.audit_artifact.as_dict()
            ),
        }


@dataclass(frozen=True)
class CodecProvenance:
    """Exact representation lineage shared by calibration and target scores."""

    family: str
    profile_name: str
    model_uid: str
    fit_source_dataset_id: str | None = None
    fit_source_run_id: str | None = None
    manifest: VerifiedArtifact | None = None
    artifact: VerifiedArtifact | None = None

    def __post_init__(self) -> None:
        family = _required_text(self.family, field="codec family").lower()
        profile = _required_text(self.profile_name, field="compression profile")
        model_uid = _required_text(self.model_uid, field="codec model_uid")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "profile_name", profile)
        object.__setattr__(self, "model_uid", model_uid)
        if family == "origin":
            unexpected = {
                "fit_source_dataset_id": self.fit_source_dataset_id,
                "fit_source_run_id": self.fit_source_run_id,
                "manifest": self.manifest,
                "artifact": self.artifact,
            }
            if any(value is not None for value in unexpected.values()):
                raise ValueError("origin representation cannot declare fitted codec lineage")
            return
        if family not in {"pca", "pq"}:
            raise ValueError("codec family must be origin, pca, or pq")
        if None in (
            self.fit_source_dataset_id,
            self.fit_source_run_id,
            self.manifest,
            self.artifact,
        ):
            raise ValueError("compressed representation requires complete codec lineage")
        object.__setattr__(
            self,
            "fit_source_dataset_id",
            _dataset_id(
                str(self.fit_source_dataset_id),
                field="codec fit_source_dataset_id",
            ),
        )
        object.__setattr__(
            self,
            "fit_source_run_id",
            _required_text(
                str(self.fit_source_run_id),
                field="codec fit_source_run_id",
            ),
        )

    @classmethod
    def origin(cls, *, model_uid: str, profile_name: str = "origin_512") -> CodecProvenance:
        return cls(
            family="origin",
            profile_name=profile_name,
            model_uid=model_uid,
        )

    def verify(self) -> None:
        if self.family == "origin":
            return
        assert self.manifest is not None
        assert self.artifact is not None
        manifest = self.manifest.verify()
        self.artifact.verify()
        if manifest is None:
            raise ValueError("codec manifest must be JSON")
        expected = {
            "fit_source_dataset": self.fit_source_dataset_id,
            "fit_source_run_id": self.fit_source_run_id,
            "model_uid": self.model_uid,
        }
        mismatches = {
            key: {"expected": value, "actual": manifest.get(key)}
            for key, value in expected.items()
            if str(manifest.get(key)) != str(value)
        }
        if mismatches:
            raise ValueError(f"codec manifest lineage mismatch: {mismatches}")
        if manifest.get("fit_on_rfw") is True:
            raise ValueError("RFW-fitted codecs are forbidden for transfer diagnostics")
        matching = [
            entry
            for entry in manifest.get("codecs", [])
            if str(entry.get("family", "")).lower() == self.family
            and str(entry.get("profile_name", "")) == self.profile_name
        ]
        if len(matching) != 1:
            raise ValueError(
                "codec manifest must contain exactly one matching family/profile"
            )
        registered_sha = str(
            matching[0].get("artifact_sha256", matching[0].get("sha256", ""))
        ).lower()
        if registered_sha != self.artifact.sha256:
            raise ValueError("codec artifact SHA differs from codec manifest")

    def fingerprint(self) -> str:
        payload: dict[str, object] = {
            "family": self.family,
            "profile_name": self.profile_name,
            "model_uid": self.model_uid,
            "fit_source_dataset_id": self.fit_source_dataset_id,
            "fit_source_run_id": self.fit_source_run_id,
        }
        if self.manifest is not None:
            payload["manifest_sha256"] = self.manifest.sha256
            payload["manifest_artifact_uid"] = self.manifest.artifact_uid
        if self.artifact is not None:
            payload["artifact_sha256"] = self.artifact.sha256
            payload["artifact_uid"] = self.artifact.artifact_uid
        return canonical_sha256(payload)

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "profile_name": self.profile_name,
            "model_uid": self.model_uid,
            "fit_source_dataset_id": self.fit_source_dataset_id,
            "fit_source_run_id": self.fit_source_run_id,
            "manifest": None if self.manifest is None else self.manifest.as_dict(),
            "artifact": None if self.artifact is None else self.artifact.as_dict(),
        }


@dataclass(frozen=True)
class ProtocolScoreContract:
    """Semantic and artifact identity of one calibration or target score set."""

    dataset_id: str
    physical_dataset_id: str
    source_population_uid: str
    run_id: str
    model_uid: str
    protocol_uid: str
    protocol_kind: ProtocolKind
    protocol_role: str
    score_statistic: ScoreStatistic
    search_mode: str
    score_space: str
    checkpoint_training_identity_overlap_status: (
        CheckpointTrainingIdentityOverlapStatus
    )
    score_artifact: VerifiedArtifact
    codec: CodecProvenance
    gallery_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _dataset_id(self.dataset_id))
        object.__setattr__(
            self,
            "physical_dataset_id",
            _dataset_id(self.physical_dataset_id, field="physical_dataset_id"),
        )
        for field in (
            "source_population_uid",
            "run_id",
            "model_uid",
            "protocol_uid",
            "protocol_role",
            "search_mode",
            "score_space",
        ):
            object.__setattr__(
                self,
                field,
                _required_text(str(getattr(self, field)), field=field),
            )
        if self.protocol_kind not in {OPEN_SET_1_TO_N, PAIR_VERIFICATION_1_TO_1}:
            raise ValueError(f"unsupported protocol_kind: {self.protocol_kind!r}")
        if self.score_statistic not in {MAXIMUM_GALLERY_SCORE, PAIR_SCORE}:
            raise ValueError(f"unsupported score_statistic: {self.score_statistic!r}")
        checkpoint_overlap = str(
            self.checkpoint_training_identity_overlap_status
        ).strip().upper()
        if checkpoint_overlap not in {
            "DISJOINT_VERIFIED",
            "OVERLAP_VERIFIED",
            "UNKNOWN",
        }:
            raise ValueError(
                "unsupported checkpoint-training identity overlap status: "
                f"{self.checkpoint_training_identity_overlap_status!r}"
            )
        object.__setattr__(
            self,
            "checkpoint_training_identity_overlap_status",
            checkpoint_overlap,
        )
        if self.model_uid != self.codec.model_uid:
            raise ValueError("score contract model_uid differs from codec model_uid")
        if self.protocol_kind == OPEN_SET_1_TO_N:
            if self.score_statistic != MAXIMUM_GALLERY_SCORE:
                raise ValueError(
                    "1:N open-set protocols require maximum_gallery_score"
                )
            if self.gallery_size is None or int(self.gallery_size) <= 0:
                raise ValueError("1:N open-set protocols require positive gallery_size")
            object.__setattr__(self, "gallery_size", int(self.gallery_size))
        else:
            if self.score_statistic != PAIR_SCORE:
                raise ValueError("1:1 verification protocols require pair_score")
            if self.gallery_size is not None:
                raise ValueError("1:1 pair verification cannot declare gallery_size")
        if self.protocol_role == RFW_OFFICIAL_PAIR_ROLE:
            if self.dataset_id != "rfw":
                raise ValueError("rfw_official_pairs role requires dataset_id='rfw'")
            if self.protocol_kind != PAIR_VERIFICATION_1_TO_1:
                raise ValueError("RFW official pairs are a 1:1 verification protocol")

    def verify(self) -> None:
        self.codec.verify()
        payload = self.score_artifact.verify()
        if payload is None:
            raise ValueError("score_artifact must be a completed JSON manifest")
        expected = {
            "dataset_id": self.dataset_id,
            "physical_dataset_id": self.physical_dataset_id,
            "source_population_uid": self.source_population_uid,
            "run_id": self.run_id,
            "model_uid": self.model_uid,
            "protocol_uid": self.protocol_uid,
            "protocol_kind": self.protocol_kind,
            "protocol_role": self.protocol_role,
            "score_statistic": self.score_statistic,
            "search_mode": self.search_mode,
            "score_space": self.score_space,
            "checkpoint_training_identity_overlap_status": (
                self.checkpoint_training_identity_overlap_status
            ),
            "gallery_size": self.gallery_size,
            "codec_fingerprint": self.codec.fingerprint(),
        }
        mismatches = {
            key: {"expected": value, "actual": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(f"score artifact contract mismatch: {mismatches}")

    def compatibility_signature(self) -> dict[str, object]:
        return {
            "model_uid": self.model_uid,
            "protocol_kind": self.protocol_kind,
            "score_statistic": self.score_statistic,
            "search_mode": self.search_mode,
            "score_space": self.score_space,
            "checkpoint_training_identity_overlap_status": (
                self.checkpoint_training_identity_overlap_status
            ),
            "gallery_size": self.gallery_size,
            "codec_fingerprint": self.codec.fingerprint(),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "physical_dataset_id": self.physical_dataset_id,
            "source_population_uid": self.source_population_uid,
            "run_id": self.run_id,
            "model_uid": self.model_uid,
            "protocol_uid": self.protocol_uid,
            "protocol_kind": self.protocol_kind,
            "protocol_role": self.protocol_role,
            "score_statistic": self.score_statistic,
            "search_mode": self.search_mode,
            "score_space": self.score_space,
            "gallery_size": self.gallery_size,
            "score_artifact": self.score_artifact.as_dict(),
            "codec": self.codec.as_dict(),
        }


@dataclass(frozen=True)
class CalibrationThresholdArtifact:
    """One frozen threshold, optionally fitted from pooled source datasets."""

    calibration_uid: str
    source_contracts: tuple[ProtocolScoreContract, ...]
    threshold: float
    threshold_comparator: str
    threshold_policy: str
    operating_point_kind: OperatingPointKind
    operating_point_value: float | None
    manifest: VerifiedArtifact

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "calibration_uid",
            _required_text(self.calibration_uid, field="calibration_uid"),
        )
        sources = tuple(self.source_contracts)
        if not sources:
            raise ValueError("at least one calibration source contract is required")
        object.__setattr__(self, "source_contracts", sources)
        threshold = float(self.threshold)
        if not np.isfinite(threshold):
            raise ValueError("calibration threshold must be finite")
        object.__setattr__(self, "threshold", threshold)
        if self.threshold_comparator != _SUPPORTED_COMPARATOR:
            raise ValueError("only the '>=' threshold comparator is supported")
        object.__setattr__(
            self,
            "threshold_policy",
            _required_text(self.threshold_policy, field="threshold_policy"),
        )
        if self.operating_point_kind not in {"fpir", "far", "accuracy"}:
            raise ValueError(
                f"unsupported operating_point_kind: {self.operating_point_kind!r}"
            )
        if self.operating_point_value is not None:
            value = float(self.operating_point_value)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("operating_point_value must be inside [0, 1]")
            object.__setattr__(self, "operating_point_value", value)

        keys = [
            (source.dataset_id, source.run_id, source.protocol_uid)
            for source in sources
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("calibration source contracts contain duplicates")
        reference = sources[0].compatibility_signature()
        inconsistent = [
            source.dataset_id
            for source in sources[1:]
            if source.compatibility_signature() != reference
        ]
        if inconsistent:
            raise CalibrationTransferCompatibilityError(
                "pooled calibration sources have incompatible score contracts: "
                f"{inconsistent}"
            )
        if sources[0].protocol_kind == OPEN_SET_1_TO_N:
            if self.operating_point_kind != "fpir":
                raise ValueError("1:N open-set calibration requires an FPIR target")
        elif self.operating_point_kind == "fpir":
            raise ValueError("1:1 pair verification cannot use an FPIR operating point")

    @property
    def source_dataset_ids(self) -> tuple[str, ...]:
        return tuple(source.dataset_id for source in self.source_contracts)

    def verify(self) -> None:
        for source in self.source_contracts:
            source.verify()
        payload = self.manifest.verify()
        if payload is None:
            raise ValueError("calibration threshold manifest must be JSON")
        reference = self.source_contracts[0]
        expected = {
                "calibration_uid": self.calibration_uid,
                "calibration_source_dataset_ids": list(self.source_dataset_ids),
                "model_uid": reference.model_uid,
                "protocol_kind": reference.protocol_kind,
                "score_statistic": reference.score_statistic,
                "search_mode": reference.search_mode,
                "score_space": reference.score_space,
                "gallery_size": reference.gallery_size,
                "codec_fingerprint": reference.codec.fingerprint(),
                "threshold": self.threshold,
                "threshold_comparator": self.threshold_comparator,
                "threshold_policy": self.threshold_policy,
                "operating_point_kind": self.operating_point_kind,
                "operating_point_value": self.operating_point_value,
        }
        mismatches = {
            key: {"expected": value, "actual": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"calibration threshold manifest mismatch: {mismatches}"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "calibration_uid": self.calibration_uid,
            "calibration_source_dataset_ids": list(self.source_dataset_ids),
            "source_contracts": [source.as_dict() for source in self.source_contracts],
            "threshold": self.threshold,
            "threshold_comparator": self.threshold_comparator,
            "threshold_policy": self.threshold_policy,
            "operating_point_kind": self.operating_point_kind,
            "operating_point_value": self.operating_point_value,
            "manifest": self.manifest.as_dict(),
        }


def _compatibility_mismatches(
    source: CalibrationThresholdArtifact,
    target: ProtocolScoreContract,
) -> dict[str, dict[str, object]]:
    reference = source.source_contracts[0]
    expected = reference.compatibility_signature()
    actual = target.compatibility_signature()
    return {
        key: {"calibration": expected[key], "target": actual[key]}
        for key in expected
        if expected[key] != actual[key]
    }


def validate_calibration_transfer_compatibility(
    source: CalibrationThresholdArtifact,
    target: ProtocolScoreContract,
) -> None:
    """Fail closed unless the threshold and target have identical score semantics."""

    source.verify()
    target.verify()
    mismatches = _compatibility_mismatches(source, target)
    if mismatches:
        if "score_statistic" in mismatches or "protocol_kind" in mismatches:
            raise CalibrationTransferCompatibilityError(
                "1:N maximum-gallery scores and 1:1 pair scores are not "
                f"interchangeable: {mismatches}"
            )
        raise CalibrationTransferCompatibilityError(
            f"calibration/target score contract mismatch: {mismatches}"
        )


@dataclass(frozen=True)
class CalibrationTransferCase:
    source: CalibrationThresholdArtifact
    target: ProtocolScoreContract
    identity_overlap_audits: tuple[IdentityOverlapAudit, ...]
    transfer_scope: str = STRICT_EXTERNAL_TRANSFER
    evaluation_mode: str = EXTERNAL_FIXED_THRESHOLD

    def __post_init__(self) -> None:
        if self.evaluation_mode != EXTERNAL_FIXED_THRESHOLD:
            raise ValueError(
                "cross-dataset cases must apply one externally frozen threshold"
            )
        validate_calibration_transfer_compatibility(self.source, self.target)
        audits = tuple(self.identity_overlap_audits)
        object.__setattr__(self, "identity_overlap_audits", audits)
        if self.transfer_scope not in {
            STRICT_EXTERNAL_TRANSFER,
            SAME_DOMAIN_DIAGNOSTIC,
        }:
            raise ValueError(f"unsupported transfer_scope: {self.transfer_scope!r}")
        expected_pairs = {
            (source.source_population_uid, self.target.source_population_uid)
            for source in self.source.source_contracts
        }
        observed_pairs = {
            (audit.source_population_uid, audit.target_population_uid)
            for audit in audits
        }
        if observed_pairs != expected_pairs or len(audits) != len(expected_pairs):
            raise CalibrationTransferCompatibilityError(
                "one identity-overlap audit is required for every calibration "
                f"population/target population pair: expected={sorted(expected_pairs)}, "
                f"actual={sorted(observed_pairs)}"
            )
        audit_by_pair = {
            (audit.source_population_uid, audit.target_population_uid): audit
            for audit in audits
        }
        for source_contract in self.source.source_contracts:
            key = (
                source_contract.source_population_uid,
                self.target.source_population_uid,
            )
            audit = audit_by_pair[key]
            audit.verify()
            same_population = (
                source_contract.source_population_uid
                == self.target.source_population_uid
            )
            if same_population and audit.status == "disjoint_verified":
                raise CalibrationTransferCompatibilityError(
                    "the same source_population_uid cannot be identity-disjoint"
                )
            if self.transfer_scope == STRICT_EXTERNAL_TRANSFER:
                if (
                    source_contract.physical_dataset_id
                    == self.target.physical_dataset_id
                ):
                    raise CalibrationTransferCompatibilityError(
                        "strict external transfer requires different physical datasets; "
                        f"both use {self.target.physical_dataset_id!r}"
                    )
                if audit.status != "disjoint_verified":
                    raise CalibrationTransferCompatibilityError(
                        "strict external transfer requires verified identity disjointness; "
                        f"status={audit.status!r} for {key}"
                    )

    @property
    def case_uid(self) -> str:
        return canonical_sha256(_content_identity(self.as_dict()))[:24]

    def as_dict(self) -> dict[str, object]:
        if self.transfer_scope == SAME_DOMAIN_DIAGNOSTIC:
            mode = SAME_DOMAIN_DIAGNOSTIC
        elif self.target.protocol_role == RFW_OFFICIAL_PAIR_ROLE:
            mode = RFW_EXTERNAL_DIAGNOSTIC
        else:
            mode = self.evaluation_mode
        return {
            "evaluation_mode": mode,
            "transfer_scope": self.transfer_scope,
            "strict_external_transfer_eligible": (
                self.transfer_scope == STRICT_EXTERNAL_TRANSFER
            ),
            "identity_overlap_audits": [
                audit.as_dict() for audit in self.identity_overlap_audits
            ],
            "calibration": self.source.as_dict(),
            "target": self.target.as_dict(),
        }


@dataclass(frozen=True)
class CrossDatasetCalibrationPlan:
    calibration_sources: tuple[CalibrationThresholdArtifact, ...]
    targets: tuple[ProtocolScoreContract, ...]
    target_dataset_ids: tuple[str, ...]
    cases: tuple[CalibrationTransferCase, ...]

    @property
    def plan_uid(self) -> str:
        return canonical_sha256(_content_identity(self.as_dict()))[:24]

    @property
    def calibration_source_dataset_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    dataset_id
                    for source in self.calibration_sources
                    for dataset_id in source.source_dataset_ids
                }
            )
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_type": "cross_dataset_calibration_transfer_plan",
            "schema_version": 1,
            "calibration_source_dataset_ids": list(
                self.calibration_source_dataset_ids
            ),
            "target_dataset_ids": list(self.target_dataset_ids),
            "case_count": len(self.cases),
            "cases": [
                {"case_uid": case.case_uid, **case.as_dict()} for case in self.cases
            ],
        }


def _selected_dataset_ids(values: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(_dataset_id(value, field="target DATASET_IDS") for value in values)
    if not selected:
        raise ValueError("target DATASET_IDS must not be empty")
    if len(set(selected)) != len(selected):
        raise ValueError(f"target DATASET_IDS contains duplicates: {selected}")
    return selected


def build_cross_dataset_calibration_plan(
    *,
    calibration_sources: Sequence[CalibrationThresholdArtifact],
    targets: Sequence[ProtocolScoreContract],
    target_dataset_ids: Sequence[str],
    identity_overlap_audits: Sequence[IdentityOverlapAudit],
    allow_same_domain_diagnostic: bool = False,
) -> CrossDatasetCalibrationPlan:
    """Build an explicit calibration-source × selected-target transfer matrix."""

    sources = tuple(calibration_sources)
    available_targets = tuple(targets)
    audits = tuple(identity_overlap_audits)
    selected_ids = _selected_dataset_ids(target_dataset_ids)
    if not sources:
        raise ValueError("calibration_sources must not be empty")
    selected_targets = tuple(
        target for target in available_targets if target.dataset_id in selected_ids
    )
    missing = sorted(set(selected_ids) - {target.dataset_id for target in selected_targets})
    if missing:
        raise ValueError(f"selected target datasets have no score contract: {missing}")
    target_keys = [
        (target.dataset_id, target.run_id, target.protocol_uid, target.search_mode)
        for target in selected_targets
    ]
    if len(set(target_keys)) != len(target_keys):
        raise ValueError("target score contracts contain duplicates")

    cases: list[CalibrationTransferCase] = []
    for source in sources:
        for target in selected_targets:
            expected_pairs = {
                (member.source_population_uid, target.source_population_uid)
                for member in source.source_contracts
            }
            case_audits = tuple(
                audit
                for audit in audits
                if (audit.source_population_uid, audit.target_population_uid)
                in expected_pairs
            )
            same_domain = any(
                member.physical_dataset_id == target.physical_dataset_id
                or member.source_population_uid == target.source_population_uid
                for member in source.source_contracts
            )
            scope = (
                SAME_DOMAIN_DIAGNOSTIC
                if same_domain and allow_same_domain_diagnostic
                else STRICT_EXTERNAL_TRANSFER
            )
            cases.append(
                CalibrationTransferCase(
                    source=source,
                    target=target,
                    identity_overlap_audits=case_audits,
                    transfer_scope=scope,
                )
            )
    if not cases:
        raise ValueError("calibration transfer plan contains no cases")
    return CrossDatasetCalibrationPlan(
        calibration_sources=sources,
        targets=selected_targets,
        target_dataset_ids=selected_ids,
        cases=tuple(cases),
    )


@dataclass(frozen=True)
class CalibrationEvaluation:
    evaluation_mode: str
    evaluation_uid: str
    scored_rows: pd.DataFrame
    group_summary: pd.DataFrame
    summary: dict[str, Any]


def _validated_scores(
    frame: pd.DataFrame,
    *,
    score_column: str,
    score_space: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if score_column not in frame:
        raise ValueError(f"target scores are missing column: {score_column}")
    if frame.empty:
        raise ValueError("target score frame must not be empty")
    scored = frame.copy().reset_index(drop=True)
    values = scored[score_column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("target scores must be finite")
    if "cosine" in score_space.lower() and (
        np.any(values < -1.000001) or np.any(values > 1.000001)
    ):
        raise ValueError("cosine scores must be inside [-1, 1]")
    return scored, values


def _rfw_pair_structure(scored: pd.DataFrame, *, strict_official: bool) -> None:
    required = {"pair_id", "rfw_group", "fold_index", "is_genuine"}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"RFW official pair scores are missing columns: {missing}")
    if scored["pair_id"].astype(str).duplicated().any():
        raise ValueError("RFW pair_id values must be unique")
    if not strict_official:
        return
    if set(scored["rfw_group"].astype(str)) != set(RFW_GROUPS):
        raise ValueError("strict RFW evaluation requires all four official groups")
    counts = scored.groupby(["rfw_group", "fold_index", "is_genuine"]).size()
    expected_index = pd.MultiIndex.from_product(
        [RFW_GROUPS, range(RFW_OFFICIAL_FOLD_COUNT), [False, True]],
        names=["rfw_group", "fold_index", "is_genuine"],
    )
    expected = pd.Series(
        RFW_OFFICIAL_GENUINE_PER_FOLD,
        index=expected_index,
        dtype=np.int64,
    )
    if not counts.reindex(expected_index, fill_value=0).equals(expected):
        raise ValueError(
            "strict RFW evaluation requires 300 genuine and 300 impostor "
            "pairs per group/fold"
        )


def _base_external_summary(case: CalibrationTransferCase) -> dict[str, Any]:
    payload = case.as_dict()
    return {
        "schema_version": 1,
        "artifact_type": "cross_dataset_calibration_transfer_evaluation",
        "case_uid": case.case_uid,
        "evaluation_mode": payload["evaluation_mode"],
        "externally_calibrated": (
            case.transfer_scope == STRICT_EXTERNAL_TRANSFER
        ),
        "same_domain_diagnostic": (
            case.transfer_scope == SAME_DOMAIN_DIAGNOSTIC
        ),
        "strict_external_transfer_eligible": (
            case.transfer_scope == STRICT_EXTERNAL_TRANSFER
        ),
        "threshold_fit_on_target": False,
        "threshold": case.source.threshold,
        "threshold_comparator": case.source.threshold_comparator,
        "threshold_policy": case.source.threshold_policy,
        "calibration_source_dataset_ids": list(
            case.source.source_dataset_ids
        ),
        "target_dataset_id": case.target.dataset_id,
        "protocol_kind": case.target.protocol_kind,
        "protocol_uid": case.target.protocol_uid,
        "score_statistic": case.target.score_statistic,
        "search_mode": case.target.search_mode,
        "score_space": case.target.score_space,
        "checkpoint_training_identity_overlap_status": (
            case.target.checkpoint_training_identity_overlap_status
        ),
        "strict_unseen_identity_evidence": (
            case.target.checkpoint_training_identity_overlap_status
            == "DISJOINT_VERIFIED"
        ),
        "codec_source": case.target.codec.as_dict(),
        "lineage": payload,
    }


def evaluate_external_calibration_transfer(
    case: CalibrationTransferCase,
    target_scores: pd.DataFrame,
    *,
    score_column: str = "score",
    strict_rfw_official: bool = True,
) -> CalibrationEvaluation:
    """Apply a compatible frozen threshold without fitting on target scores."""

    validate_calibration_transfer_compatibility(case.source, case.target)
    scored, values = _validated_scores(
        target_scores,
        score_column=score_column,
        score_space=case.target.score_space,
    )
    accepted = values >= case.source.threshold
    scored["accepted"] = accepted
    summary = _base_external_summary(case)
    group_summary = pd.DataFrame()

    if case.target.protocol_kind == OPEN_SET_1_TO_N:
        required = {"is_mated", "top1_correct"}
        missing = sorted(required - set(scored.columns))
        if missing:
            raise ValueError(f"open-set target scores are missing columns: {missing}")
        is_mated = scored["is_mated"].to_numpy(dtype=bool)
        top1_correct = scored["top1_correct"].to_numpy(dtype=bool)
        non_mated = ~is_mated
        if not is_mated.any() or not non_mated.any():
            raise ValueError("open-set evaluation requires mated and non-mated probes")
        scored["correct_identification"] = accepted & is_mated & top1_correct
        summary.update(
            {
                "probe_count": int(len(scored)),
                "mated_probe_count": int(is_mated.sum()),
                "non_mated_probe_count": int(non_mated.sum()),
                "dir_rank1": float(
                    np.mean(accepted[is_mated] & top1_correct[is_mated])
                ),
                "fpir": float(np.mean(accepted[non_mated])),
                "open_set_protocol": True,
            }
        )
    else:
        if "is_genuine" not in scored:
            raise ValueError("1:1 target scores require is_genuine")
        genuine = scored["is_genuine"].to_numpy(dtype=bool)
        impostor = ~genuine
        if not genuine.any() or not impostor.any():
            raise ValueError("1:1 evaluation requires genuine and impostor pairs")
        scored["decision_correct"] = accepted == genuine
        overall_eer = empirical_equal_error_rate(values, genuine)
        summary.update(
            {
                "pair_count": int(len(scored)),
                "accuracy": float(np.mean(accepted == genuine)),
                "tar": float(np.mean(accepted[genuine])),
                "far": float(np.mean(accepted[impostor])),
                "eer": overall_eer.eer,
                "eer_threshold": overall_eer.threshold,
                "eer_far": overall_eer.far,
                "eer_frr": overall_eer.frr,
                "eer_threshold_source": "target_pair_scores_and_labels",
                "eer_method": overall_eer.method,
                "open_set_protocol": False,
            }
        )
        if case.target.protocol_role == RFW_OFFICIAL_PAIR_ROLE:
            _rfw_pair_structure(scored, strict_official=strict_rfw_official)
            group_rows: list[dict[str, object]] = []
            for group_name, group_frame in scored.groupby(
                "rfw_group", sort=True
            ):
                group_scores = group_frame[score_column].to_numpy(
                    dtype=np.float64
                )
                group_genuine = group_frame["is_genuine"].to_numpy(dtype=bool)
                group_accepted = group_scores >= case.source.threshold
                group_impostor = ~group_genuine
                group_eer = empirical_equal_error_rate(
                    group_scores,
                    group_genuine,
                )
                group_rows.append(
                    {
                        "rfw_group": str(group_name),
                        "pair_count": int(len(group_scores)),
                        "accuracy": float(
                            np.mean(group_accepted == group_genuine)
                        ),
                        "tar": float(np.mean(group_accepted[group_genuine])),
                        "far": float(np.mean(group_accepted[group_impostor])),
                        "eer": group_eer.eer,
                        "eer_threshold": group_eer.threshold,
                        "eer_far": group_eer.far,
                        "eer_frr": group_eer.frr,
                        "eer_threshold_source": (
                            "target_group_pair_scores_and_labels"
                        ),
                        "eer_method": group_eer.method,
                    }
                )
            group_summary = pd.DataFrame.from_records(group_rows)
            summary.update(
                {
                    "protocol": "rfw_official_pairs_external_threshold_diagnostic",
                    "official_internal_9fold_baseline": False,
                    "rfw_threshold_fit_scope": "external_source_only",
                }
            )

    evaluation_uid = canonical_sha256(_content_identity(summary))[:24]
    summary["evaluation_uid"] = evaluation_uid
    return CalibrationEvaluation(
        evaluation_mode=str(summary["evaluation_mode"]),
        evaluation_uid=evaluation_uid,
        scored_rows=scored,
        group_summary=group_summary,
        summary=summary,
    )


def evaluate_rfw_official_internal_baseline(
    target: ProtocolScoreContract,
    pairs: pd.DataFrame,
    *,
    scores: Sequence[float],
    thresholds: Sequence[float],
    strict_official: bool = True,
    bootstrap_seed: int = 42,
    bootstrap_repeats: int = 2000,
) -> CalibrationEvaluation:
    """Run the canonical RFW other-nine-fold threshold baseline only."""

    target.verify()
    if target.protocol_role != RFW_OFFICIAL_PAIR_ROLE:
        raise ValueError("RFW internal baseline requires rfw_official_pairs target")
    if target.protocol_kind != PAIR_VERIFICATION_1_TO_1:
        raise ValueError("RFW internal baseline requires 1:1 pair scores")
    result = evaluate_rfw_pair_scores(
        pairs,
        scores=scores,
        score_space=target.score_space,
        thresholds=thresholds,
        strict_official=strict_official,
        bootstrap_seed=bootstrap_seed,
        bootstrap_repeats=bootstrap_repeats,
    )
    summary = {
        **result.summary,
        "schema_version": 1,
        "artifact_type": "rfw_official_internal_9fold_baseline",
        "evaluation_mode": RFW_INTERNAL_9FOLD,
        "externally_calibrated": False,
        "official_internal_9fold_baseline": True,
        "threshold_fit_on_target": True,
        "rfw_threshold_fit_scope": "same_group_other_9_folds",
        "calibration_source_dataset_ids": ["rfw"],
        "target_dataset_id": "rfw",
        "checkpoint_training_identity_overlap_status": (
            target.checkpoint_training_identity_overlap_status
        ),
        "strict_unseen_identity_evidence": (
            target.checkpoint_training_identity_overlap_status
            == "DISJOINT_VERIFIED"
        ),
        "target": target.as_dict(),
        "codec_source": target.codec.as_dict(),
    }
    evaluation_uid = canonical_sha256(_content_identity(summary))[:24]
    summary["evaluation_uid"] = evaluation_uid
    return CalibrationEvaluation(
        evaluation_mode=RFW_INTERNAL_9FOLD,
        evaluation_uid=evaluation_uid,
        scored_rows=result.pair_scores,
        group_summary=result.group_summary,
        summary=summary,
    )


def calibration_evaluation_manifest(
    evaluation: CalibrationEvaluation,
) -> dict[str, Any]:
    """Return a compact manifest payload retaining all scientific lineage."""

    return {
        "schema_version": 1,
        "status": "completed",
        "artifact_type": "cross_dataset_calibration_evaluation_manifest",
        "evaluation_uid": evaluation.evaluation_uid,
        "evaluation_mode": evaluation.evaluation_mode,
        "summary": evaluation.summary,
        "tables": {
            "scored_rows": {
                "row_count": int(len(evaluation.scored_rows)),
                "columns": list(evaluation.scored_rows.columns),
            },
            "group_summary": {
                "row_count": int(len(evaluation.group_summary)),
                "columns": list(evaluation.group_summary.columns),
            },
        },
    }
