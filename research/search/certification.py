from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class SimilarityBounds:
    approximate_scores: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    approximate_angles: np.ndarray


@dataclass(frozen=True)
class CertifiedDecision:
    decision: str
    selected_index: int | None
    selected_identity: str | None
    rank_certified: bool
    threshold: float
    bounds: SimilarityBounds


@dataclass(frozen=True)
class ExactDecision:
    decision: str
    selected_index: int | None
    selected_identity: str | None
    top1_index: int
    top1_identity: str
    top1_score: float
    threshold: float
    scores: np.ndarray


@dataclass(frozen=True)
class CertificationSummary:
    counts: dict[str, int]
    certification_coverage: float
    accept_coverage: float
    reject_coverage: float
    defer_rate: float


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    if value.ndim != 1:
        raise ValueError("query must be a 1D vector")
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("query vector must be non-zero")
    return value / norm


def _unit_matrix(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("compressed_templates must be a 2D matrix")
    if len(matrix) == 0:
        raise ValueError("at least one compressed template is required")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("compressed template vectors must be non-zero")
    return matrix / norms


def _angular_errors(values: np.ndarray, expected_length: int) -> np.ndarray:
    errors = np.asarray(values, dtype=np.float32)
    if errors.shape != (expected_length,):
        raise ValueError("template_angular_errors must have one value per template")
    if np.any(~np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError("template_angular_errors must be finite non-negative angles")
    return np.minimum(errors, np.pi)


def compute_similarity_bounds(
    query: np.ndarray,
    compressed_templates: np.ndarray,
    template_angular_errors: np.ndarray,
    query_angular_error: float = 0.0,
) -> SimilarityBounds:
    """Bound true query-template cosine scores from compressed template angles.

    For unit vectors, if the compressed template is within angular error
    theta_i of the true template and the approximate query-template angle is
    phi_i, then spherical triangle inequality gives:

        max(0, phi_i - theta_i) <= alpha_i <= min(pi, phi_i + theta_i)

    Cosine is decreasing on [0, pi], so these angle bounds become cosine lower
    and upper bounds. ``query_angular_error`` is added when the query vector is
    also compressed; leave it at zero for the usual asymmetric search case.
    """

    q = _unit_vector(query)
    templates = _unit_matrix(compressed_templates)
    errors = _angular_errors(template_angular_errors, len(templates))
    query_error = float(query_angular_error)
    if not np.isfinite(query_error) or query_error < 0.0:
        raise ValueError("query_angular_error must be a finite non-negative angle")

    approximate_scores = templates @ q
    approximate_angles = np.arccos(np.clip(approximate_scores, -1.0, 1.0))
    total_errors = np.minimum(errors + query_error, np.pi)

    true_angle_lower = np.maximum(0.0, approximate_angles - total_errors)
    true_angle_upper = np.minimum(np.pi, approximate_angles + total_errors)
    lower_bounds = np.cos(true_angle_upper)
    upper_bounds = np.cos(true_angle_lower)

    return SimilarityBounds(
        approximate_scores=approximate_scores.astype(np.float32),
        lower_bounds=lower_bounds.astype(np.float32),
        upper_bounds=upper_bounds.astype(np.float32),
        approximate_angles=approximate_angles.astype(np.float32),
    )


def certify_open_set_decision(
    query: np.ndarray,
    compressed_templates: np.ndarray,
    template_identities: Sequence[str],
    template_angular_errors: np.ndarray,
    threshold: float,
    query_angular_error: float = 0.0,
) -> CertifiedDecision:
    bounds = compute_similarity_bounds(
        query=query,
        compressed_templates=compressed_templates,
        template_angular_errors=template_angular_errors,
        query_angular_error=query_angular_error,
    )
    identities = [str(identity) for identity in template_identities]
    if len(identities) != len(bounds.approximate_scores):
        raise ValueError("template_identities must have one value per template")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite")

    if float(np.max(bounds.upper_bounds)) < threshold_value:
        return CertifiedDecision(
            decision="reject",
            selected_index=None,
            selected_identity=None,
            rank_certified=False,
            threshold=threshold_value,
            bounds=bounds,
        )

    selected_index = int(np.argmax(bounds.approximate_scores))
    selected_lower = float(bounds.lower_bounds[selected_index])
    other_upper = np.delete(bounds.upper_bounds, selected_index)
    rank_certified = len(other_upper) == 0 or selected_lower > float(np.max(other_upper))

    decision = "accept" if rank_certified and selected_lower >= threshold_value else "defer"
    return CertifiedDecision(
        decision=decision,
        selected_index=selected_index,
        selected_identity=identities[selected_index],
        rank_certified=bool(rank_certified),
        threshold=threshold_value,
        bounds=bounds,
    )


def exact_open_set_decision(
    query: np.ndarray,
    templates: np.ndarray,
    template_identities: Sequence[str],
    threshold: float,
) -> ExactDecision:
    q = _unit_vector(query)
    template_matrix = _unit_matrix(templates)
    identities = [str(identity) for identity in template_identities]
    if len(identities) != len(template_matrix):
        raise ValueError("template_identities must have one value per template")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite")

    scores = template_matrix @ q
    top1_index = int(np.argmax(scores))
    top1_score = float(scores[top1_index])
    top1_identity = identities[top1_index]
    if top1_score >= threshold_value:
        return ExactDecision(
            decision="accept",
            selected_index=top1_index,
            selected_identity=top1_identity,
            top1_index=top1_index,
            top1_identity=top1_identity,
            top1_score=top1_score,
            threshold=threshold_value,
            scores=scores.astype(np.float32),
        )
    return ExactDecision(
        decision="reject",
        selected_index=None,
        selected_identity=None,
        top1_index=top1_index,
        top1_identity=top1_identity,
        top1_score=top1_score,
        threshold=threshold_value,
        scores=scores.astype(np.float32),
    )


def summarize_certified_decisions(decisions: Sequence[CertifiedDecision]) -> CertificationSummary:
    if not decisions:
        raise ValueError("at least one certified decision is required")

    counts = {"accept": 0, "reject": 0, "defer": 0}
    for decision in decisions:
        if decision.decision not in counts:
            raise ValueError(f"unknown certified decision: {decision.decision}")
        counts[decision.decision] += 1

    total = float(len(decisions))
    accept = counts["accept"] / total
    reject = counts["reject"] / total
    defer = counts["defer"] / total
    return CertificationSummary(
        counts=counts,
        certification_coverage=accept + reject,
        accept_coverage=accept,
        reject_coverage=reject,
        defer_rate=defer,
    )
