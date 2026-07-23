from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


LOO_TARGET_NAME = "origin_leave_one_out_identity_cosine"
ELIGIBLE_REASON = "eligible"
MISSING_IDENTITY_REASON = "missing_identity"
SINGLETON_IDENTITY_REASON = "singleton_identity"
ZERO_RESIDUAL_REASON = "zero_norm_leave_one_out_template"


def _string_identifiers(
    values: Sequence[Any] | np.ndarray,
    *,
    name: str,
    length: int,
    allow_missing: bool,
    require_unique: bool,
) -> np.ndarray:
    materialized = list(values)
    if len(materialized) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    normalized: list[str] = []
    for value in materialized:
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if missing:
            if not allow_missing:
                raise ValueError(f"{name} must not contain missing values")
            normalized.append("")
            continue
        text = str(value).strip()
        if not text and not allow_missing:
            raise ValueError(f"{name} must not contain empty values")
        normalized.append(text)
    if require_unique and len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} values must be unique")
    width = max((len(value) for value in normalized), default=1)
    return np.asarray(normalized, dtype=f"<U{max(width, 1)}")


def _unit_embedding_matrix(
    embeddings: np.ndarray,
    *,
    unit_tolerance: float,
) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("normalized_embeddings must be a non-empty 2D matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("normalized_embeddings must contain only finite values")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("normalized_embeddings must not contain zero-norm rows")
    if not np.allclose(norms, 1.0, atol=unit_tolerance, rtol=0.0):
        maximum_error = float(np.max(np.abs(norms - 1.0)))
        raise ValueError(
            "normalized_embeddings must already have unit row norms; "
            f"maximum absolute norm error={maximum_error:.6g}"
        )
    return matrix


@dataclass(frozen=True)
class LeaveOneOutTemplateBundle:
    sample_ids: np.ndarray
    identity_ids: np.ndarray
    scope_ids: np.ndarray
    templates: np.ndarray
    template_member_counts: np.ndarray
    eligible: np.ndarray
    exclusion_reasons: np.ndarray
    target_scores: np.ndarray
    model_uid: str
    target_name: str = LOO_TARGET_NAME

    def __post_init__(self) -> None:
        row_count = len(self.sample_ids)
        arrays = {
            "identity_ids": self.identity_ids,
            "scope_ids": self.scope_ids,
            "template_member_counts": self.template_member_counts,
            "eligible": self.eligible,
            "exclusion_reasons": self.exclusion_reasons,
            "target_scores": self.target_scores,
        }
        for name, values in arrays.items():
            if len(values) != row_count:
                raise ValueError(f"{name} must align with sample_ids")
        if self.templates.ndim != 2 or len(self.templates) != row_count:
            raise ValueError("templates must have shape [sample_count, dimension]")
        if not str(self.model_uid).strip():
            raise ValueError("model_uid must not be empty")
        if not str(self.target_name).strip():
            raise ValueError("target_name must not be empty")

    @property
    def eligible_count(self) -> int:
        return int(np.count_nonzero(self.eligible))

    @property
    def coverage(self) -> float:
        return float(self.eligible_count / len(self.sample_ids))

    def metadata_frame(self) -> pd.DataFrame:
        scores = self.target_scores.astype(np.float64, copy=False)
        return pd.DataFrame(
            {
                "sample_id": self.sample_ids.astype(str),
                "identity_id": self.identity_ids.astype(str),
                "template_scope_id": self.scope_ids.astype(str),
                "model_uid": str(self.model_uid),
                "saliency_target_name": str(self.target_name),
                "template_member_count": self.template_member_counts.astype(
                    np.int64,
                    copy=False,
                ),
                "saliency_target_eligible": self.eligible.astype(bool, copy=False),
                "saliency_target_status": self.exclusion_reasons.astype(str),
                "identity_template_cosine": scores,
            }
        )

    def coverage_summary(self) -> pd.DataFrame:
        frame = self.metadata_frame()
        return (
            frame.groupby(
                [
                    "template_scope_id",
                    "saliency_target_eligible",
                    "saliency_target_status",
                ],
                dropna=False,
                sort=True,
            )
            .size()
            .rename("sample_count")
            .reset_index()
        )


def build_leave_one_out_identity_templates(
    sample_ids: Sequence[Any] | np.ndarray,
    identity_ids: Sequence[Any] | np.ndarray,
    normalized_embeddings: np.ndarray,
    *,
    model_uid: str,
    scope_ids: Sequence[Any] | np.ndarray | None = None,
    unit_tolerance: float = 1e-4,
    residual_epsilon: float = 1e-10,
    require_all_eligible: bool = False,
) -> LeaveOneOutTemplateBundle:
    """Build one detached same-identity template per sample.

    Templates never cross ``scope_ids``. Within each ``(scope, identity)``
    group, unit embeddings are accumulated in float64 and the current sample
    is removed before L2 normalization:

    ``t_i = normalize(sum_{j != i} e_j)``.

    The divisor ``N-1`` is omitted because the result is normalized. Samples
    without a usable template remain in the bundle with a status and NaN
    target score; no alternate target is silently substituted.
    """

    if not np.isfinite(float(unit_tolerance)) or float(unit_tolerance) <= 0.0:
        raise ValueError("unit_tolerance must be positive and finite")
    if not np.isfinite(float(residual_epsilon)) or float(residual_epsilon) <= 0.0:
        raise ValueError("residual_epsilon must be positive and finite")
    embeddings = _unit_embedding_matrix(
        normalized_embeddings,
        unit_tolerance=float(unit_tolerance),
    )
    row_count, dimension = embeddings.shape
    samples = _string_identifiers(
        sample_ids,
        name="sample_ids",
        length=row_count,
        allow_missing=False,
        require_unique=True,
    )
    identities = _string_identifiers(
        identity_ids,
        name="identity_ids",
        length=row_count,
        allow_missing=True,
        require_unique=False,
    )
    if scope_ids is None:
        scopes = np.full(row_count, "default", dtype="<U7")
    else:
        scopes = _string_identifiers(
            scope_ids,
            name="scope_ids",
            length=row_count,
            allow_missing=False,
            require_unique=False,
        )
    normalized_model_uid = str(model_uid).strip()
    if not normalized_model_uid:
        raise ValueError("model_uid must not be empty")

    templates = np.full((row_count, dimension), np.nan, dtype=np.float32)
    member_counts = np.zeros(row_count, dtype=np.int32)
    eligible = np.zeros(row_count, dtype=bool)
    reasons = np.full(
        row_count,
        MISSING_IDENTITY_REASON,
        dtype=f"<U{len(ZERO_RESIDUAL_REASON)}",
    )
    scores = np.full(row_count, np.nan, dtype=np.float32)

    groups: dict[tuple[str, str], list[int]] = {}
    for index, (scope_id, identity_id) in enumerate(zip(scopes, identities)):
        if not identity_id:
            continue
        groups.setdefault((str(scope_id), str(identity_id)), []).append(index)

    accumulation = embeddings.astype(np.float64)
    for indices in groups.values():
        group_count = len(indices)
        if group_count < 2:
            reasons[indices[0]] = SINGLETON_IDENTITY_REASON
            continue
        group_sum = np.sum(accumulation[indices], axis=0, dtype=np.float64)
        for index in indices:
            residual = group_sum - accumulation[index]
            residual_norm = float(np.linalg.norm(residual))
            member_counts[index] = group_count - 1
            if not np.isfinite(residual_norm) or residual_norm <= residual_epsilon:
                reasons[index] = ZERO_RESIDUAL_REASON
                continue
            template = residual / residual_norm
            templates[index] = template.astype(np.float32)
            scores[index] = np.float32(
                np.clip(
                    np.dot(accumulation[index], template),
                    -1.0,
                    1.0,
                )
            )
            eligible[index] = True
            reasons[index] = ELIGIBLE_REASON

    if require_all_eligible and not bool(np.all(eligible)):
        counts = (
            pd.Series(reasons[~eligible])
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
        raise ValueError(
            "not every selected sample has a leave-one-out identity template; "
            f"ineligible={counts}"
        )

    return LeaveOneOutTemplateBundle(
        sample_ids=samples,
        identity_ids=identities,
        scope_ids=scopes,
        templates=templates,
        template_member_counts=member_counts,
        eligible=eligible,
        exclusion_reasons=reasons,
        target_scores=scores,
        model_uid=normalized_model_uid,
    )
