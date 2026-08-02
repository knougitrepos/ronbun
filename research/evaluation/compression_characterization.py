from __future__ import annotations

from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


ProgressCallback = Callable[[str, dict[str, object]], None]


PAIRED_EMBEDDING_COLUMNS = (
    "sample_id",
    "compression_family",
    "compression_profile",
    "origin_dimension",
    "search_dimension",
    "reconstruction_available",
    "metric_vector_source",
    "reconstruction_mse",
    "angular_error_rad",
    "cosine_to_origin",
    "origin_fallback_used",
)


RETRIEVAL_COMPARISON_COLUMNS = (
    "query_id",
    "query_identity_id",
    "is_mated",
    "compression_family",
    "compression_profile",
    "top_k",
    "search_mode",
    "origin_score_space",
    "compressed_score_space",
    "score_spaces_comparable",
    "frozen_origin_threshold_applicable",
    "latency_measurement_repeats",
    "latency_timer",
    "compressed_index_build_latency_ms",
    "compressed_gallery_add_latency_ms",
    "compressed_gallery_encode_latency_ms",
    "origin_search_latency_ms_total",
    "compressed_search_latency_ms_total",
    "compressed_search_latency_ms_per_query",
    "compressed_search_queries_per_second",
    "origin_top1_gallery_id",
    "compressed_top1_gallery_id",
    "origin_top1_identity_id",
    "compressed_top1_identity_id",
    "origin_top1_score",
    "compressed_top1_score",
    "compressed_score_at_origin_top1",
    "origin_score_at_compressed_top1",
    "top1_score_drift",
    "origin_winner_score_drift",
    "agreement_with_origin",
    "top_k_identity_agreement",
    "top_k_identity_jaccard",
    "origin_rank1_correct",
    "compressed_rank1_correct",
    "origin_top_k_correct",
    "compressed_top_k_correct",
    "decision_threshold",
    "origin_decision_threshold",
    "compressed_decision_threshold",
    "origin_accepted",
    "compressed_accepted",
    "threshold_crossing",
    "threshold_crossing_direction",
    "origin_decision_correct",
    "compressed_decision_correct",
    "origin_top_k_gallery_ids",
    "compressed_top_k_gallery_ids",
    "origin_top_k_identity_ids",
    "compressed_top_k_identity_ids",
    "origin_top_k_scores",
    "compressed_top_k_scores",
    "origin_fallback_used",
)


ORIGIN_RETRIEVAL_COLUMNS = (
    "query_id",
    "query_identity_id",
    "is_mated",
    "top_k",
    "origin_top1_gallery_id",
    "origin_top1_identity_id",
    "origin_top1_score",
    "origin_rank1_correct",
    "origin_top_k_correct",
)


def _as_float_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _validate_profile(compression_family: str, compression_profile: str) -> tuple[str, str]:
    family = str(compression_family).strip()
    profile = str(compression_profile).strip()
    if not family:
        raise ValueError("compression_family must not be empty")
    if not profile:
        raise ValueError("compression_profile must not be empty")
    return family, profile


def _as_identifier_array(
    values: Sequence[Any] | np.ndarray | None,
    *,
    name: str,
    length: int,
    default_prefix: str | None = None,
    require_unique: bool = False,
) -> np.ndarray:
    if values is None:
        if default_prefix is None:
            raise ValueError(f"{name} is required")
        identifiers = np.asarray(
            [f"{default_prefix}_{index}" for index in range(length)],
            dtype=object,
        )
    else:
        identifiers = np.asarray(list(values), dtype=object)
    if identifiers.ndim != 1 or len(identifiers) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    for value in identifiers.tolist():
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if missing:
            raise ValueError(f"{name} must not contain missing values")
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name} values must be hashable") from exc
    if require_unique and len(set(identifiers.tolist())) != length:
        raise ValueError(f"{name} values must be unique")
    return identifiers


def _row_normalize(matrix: np.ndarray, *, name: str) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError(f"{name} must not contain zero-norm vectors")
    return matrix / norms


def paired_embedding_metrics(
    original_embeddings: np.ndarray,
    search_embeddings: np.ndarray,
    *,
    compression_family: str,
    compression_profile: str,
    sample_ids: Sequence[Any] | np.ndarray | None = None,
    reconstructed_embeddings: np.ndarray | None = None,
) -> pd.DataFrame:
    """Characterize paired original and compressed embeddings per sample.

    Metrics are computed against ``reconstructed_embeddings`` when supplied.
    Otherwise, a same-dimensional search representation is directly compared
    with the original.  A lower-dimensional search representation without a
    reconstruction remains a valid search artifact, but its reconstruction
    metrics are reported as missing rather than silently substituting the
    original embedding.
    """

    family, profile = _validate_profile(compression_family, compression_profile)
    original = _as_float_matrix(original_embeddings, name="original_embeddings")
    search = _as_float_matrix(search_embeddings, name="search_embeddings")
    if search.shape[0] != original.shape[0]:
        raise ValueError(
            "original_embeddings and search_embeddings must have the same row count"
        )
    _row_normalize(original, name="original_embeddings")
    _row_normalize(search, name="search_embeddings")

    identifiers = _as_identifier_array(
        sample_ids,
        name="sample_ids",
        length=len(original),
        default_prefix="sample",
        require_unique=True,
    )

    comparison: np.ndarray | None
    if reconstructed_embeddings is not None:
        reconstructed = _as_float_matrix(
            reconstructed_embeddings,
            name="reconstructed_embeddings",
        )
        if reconstructed.shape != original.shape:
            raise ValueError(
                "reconstructed_embeddings must have the same shape as original_embeddings"
            )
        comparison = reconstructed
        vector_source = "reconstruction"
        reconstruction_available = True
    elif search.shape[1] == original.shape[1]:
        comparison = search
        vector_source = "search"
        reconstruction_available = False
    else:
        comparison = None
        vector_source = "unavailable"
        reconstruction_available = False

    if comparison is None:
        mse = np.full(len(original), np.nan, dtype=np.float64)
        angular = np.full(len(original), np.nan, dtype=np.float64)
        cosine = np.full(len(original), np.nan, dtype=np.float64)
    else:
        original_unit = _row_normalize(original, name="original_embeddings")
        comparison_unit = _row_normalize(comparison, name=vector_source)
        cosine = np.sum(original_unit * comparison_unit, axis=1, dtype=np.float64)
        cosine = np.clip(cosine, -1.0, 1.0)
        angular = np.arccos(cosine)
        difference = original.astype(np.float64) - comparison.astype(np.float64)
        mse = np.mean(np.square(difference), axis=1)

    rows = pd.DataFrame(
        {
            "sample_id": identifiers,
            "compression_family": family,
            "compression_profile": profile,
            "origin_dimension": int(original.shape[1]),
            "search_dimension": int(search.shape[1]),
            "reconstruction_available": reconstruction_available,
            "metric_vector_source": vector_source,
            "reconstruction_mse": mse,
            "angular_error_rad": angular,
            "cosine_to_origin": cosine,
            "origin_fallback_used": False,
        }
    )
    return rows.loc[:, PAIRED_EMBEDDING_COLUMNS]


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _batched_cosine_top_k(
    queries: np.ndarray,
    gallery: np.ndarray,
    *,
    top_k: int,
    query_batch_size: int,
    gallery_batch_size: int,
    progress: ProgressCallback | None = None,
    progress_message: str = "cosine retrieval",
    progress_offset: int = 0,
    progress_total: int | None = None,
    progress_details: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine top-k without materializing the full query-gallery matrix."""

    query_count = len(queries)
    gallery_count = len(gallery)
    result_indices = np.empty((query_count, top_k), dtype=np.int64)
    result_scores = np.empty((query_count, top_k), dtype=np.float32)

    for query_start in range(0, query_count, query_batch_size):
        query_stop = min(query_start + query_batch_size, query_count)
        query_block = queries[query_start:query_stop]
        block_count = len(query_block)
        best_scores = np.full((block_count, top_k), -np.inf, dtype=np.float32)
        best_indices = np.full((block_count, top_k), gallery_count, dtype=np.int64)

        for gallery_start in range(0, gallery_count, gallery_batch_size):
            gallery_stop = min(gallery_start + gallery_batch_size, gallery_count)
            scores = query_block @ gallery[gallery_start:gallery_stop].T
            indices = np.broadcast_to(
                np.arange(gallery_start, gallery_stop, dtype=np.int64),
                scores.shape,
            )
            candidate_scores = np.concatenate((best_scores, scores), axis=1)
            candidate_indices = np.concatenate((best_indices, indices), axis=1)

            # Stable sorting makes equal-score ties deterministic by retaining
            # the earlier gallery row, since gallery chunks are visited in row
            # order and the retained top-k rows are already ordered.
            order = np.argsort(-candidate_scores, axis=1, kind="stable")[:, :top_k]
            best_scores = np.take_along_axis(candidate_scores, order, axis=1)
            best_indices = np.take_along_axis(candidate_indices, order, axis=1)

        result_indices[query_start:query_stop] = best_indices
        result_scores[query_start:query_stop] = best_scores
        if progress is not None:
            progress(
                progress_message,
                {
                    "processed": int(progress_offset + query_stop),
                    "total": int(progress_total or query_count),
                    **(progress_details or {}),
                },
            )

    return result_indices, result_scores


def _identity_jaccard(left: tuple[Any, ...], right: tuple[Any, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0


def _cosine_threshold(value: float, *, name: str) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError(f"{name} must be a finite cosine score in [-1, 1]")
    return threshold


def origin_cosine_retrieval(
    queries: np.ndarray,
    gallery: np.ndarray,
    *,
    query_ids: Sequence[Any] | np.ndarray,
    gallery_ids: Sequence[Any] | np.ndarray,
    query_identity_ids: Sequence[Any] | np.ndarray,
    gallery_identity_ids: Sequence[Any] | np.ndarray,
    top_k: int = 1,
    query_batch_size: int = 128,
    gallery_batch_size: int = 4096,
    max_pairwise_elements: int = 1_000_000,
    progress: ProgressCallback | None = None,
    progress_message: str = "origin cosine retrieval",
    progress_details: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Search one cosine space once and return origin-prefixed audit fields.

    This path is intentionally separate from ``compare_cosine_retrieval`` so
    calibration diagnostics do not perform the same origin search once per
    compression profile (or twice by comparing origin with itself).
    """

    query_matrix = _as_float_matrix(queries, name="queries")
    gallery_matrix = _as_float_matrix(gallery, name="gallery")
    if query_matrix.shape[1] != gallery_matrix.shape[1]:
        raise ValueError("query and gallery dimensions must match")

    top_k_value = _positive_integer(top_k, name="top_k")
    query_batch = _positive_integer(query_batch_size, name="query_batch_size")
    gallery_batch = _positive_integer(gallery_batch_size, name="gallery_batch_size")
    pair_limit = _positive_integer(max_pairwise_elements, name="max_pairwise_elements")
    if top_k_value > len(gallery_matrix):
        raise ValueError("top_k must not exceed the gallery row count")
    effective_query_batch = min(query_batch, len(query_matrix))
    effective_gallery_batch = min(gallery_batch, len(gallery_matrix))
    largest_score_block = effective_query_batch * effective_gallery_batch
    largest_merge_block = effective_query_batch * (
        effective_gallery_batch + top_k_value
    )
    if max(largest_score_block, largest_merge_block) > pair_limit:
        raise ValueError(
            "batched retrieval working set exceeds max_pairwise_elements "
            f"for this input ({largest_merge_block} > {pair_limit}); reduce "
            "query_batch_size, gallery_batch_size, or top_k"
        )

    query_identifiers = _as_identifier_array(
        query_ids,
        name="query_ids",
        length=len(query_matrix),
        require_unique=True,
    )
    gallery_identifiers = _as_identifier_array(
        gallery_ids,
        name="gallery_ids",
        length=len(gallery_matrix),
        require_unique=True,
    )
    query_identities = _as_identifier_array(
        query_identity_ids,
        name="query_identity_ids",
        length=len(query_matrix),
    )
    gallery_identities = _as_identifier_array(
        gallery_identity_ids,
        name="gallery_identity_ids",
        length=len(gallery_matrix),
    )
    query_unit = _row_normalize(query_matrix, name="queries")
    gallery_unit = _row_normalize(gallery_matrix, name="gallery")
    indices, scores = _batched_cosine_top_k(
        query_unit,
        gallery_unit,
        top_k=top_k_value,
        query_batch_size=query_batch,
        gallery_batch_size=gallery_batch,
        progress=progress,
        progress_message=progress_message,
        progress_total=len(query_matrix),
        progress_details=progress_details,
    )

    ranked_identities = gallery_identities[indices]
    top1_identities = ranked_identities[:, 0]
    gallery_identity_set = set(gallery_identities.tolist())
    is_mated = np.fromiter(
        (identity in gallery_identity_set for identity in query_identities),
        dtype=bool,
        count=len(query_identities),
    )
    rank1_correct = top1_identities == query_identities
    top_k_correct = np.any(
        ranked_identities == query_identities[:, np.newaxis],
        axis=1,
    )
    result = pd.DataFrame(
        {
            "query_id": query_identifiers,
            "query_identity_id": query_identities,
            "is_mated": is_mated,
            "top_k": top_k_value,
            "origin_top1_gallery_id": gallery_identifiers[indices[:, 0]],
            "origin_top1_identity_id": top1_identities,
            "origin_top1_score": scores[:, 0].astype(np.float64),
            "origin_rank1_correct": rank1_correct,
            "origin_top_k_correct": top_k_correct,
        }
    )
    return result.loc[:, ORIGIN_RETRIEVAL_COLUMNS]


def compare_cosine_retrieval(
    original_queries: np.ndarray,
    original_gallery: np.ndarray,
    compressed_queries: np.ndarray,
    compressed_gallery: np.ndarray,
    *,
    query_ids: Sequence[Any] | np.ndarray,
    gallery_ids: Sequence[Any] | np.ndarray,
    query_identity_ids: Sequence[Any] | np.ndarray,
    gallery_identity_ids: Sequence[Any] | np.ndarray,
    compression_family: str,
    compression_profile: str,
    top_k: int = 1,
    threshold: float | None = None,
    origin_threshold: float | None = None,
    compressed_threshold: float | None = None,
    query_batch_size: int = 128,
    gallery_batch_size: int = 4096,
    max_pairwise_elements: int = 1_000_000,
    progress: ProgressCallback | None = None,
    progress_message: str = "cosine retrieval",
    progress_offset: int = 0,
    progress_total: int | None = None,
    progress_details: dict[str, object] | None = None,
    search_mode: str = "cosine_search",
) -> pd.DataFrame:
    """Compare origin and compressed cosine retrieval without fallback.

    The two representations are searched independently.  ``agreement_with_origin``
    records rank-1 identity agreement only; ground-truth correctness is kept in
    separate ``*_rank1_correct`` and ``*_decision_correct`` fields.  ``threshold``
    is the shorthand for one shared operating threshold.  To compare a frozen
    origin threshold with a recalibrated compressed threshold, pass both
    ``origin_threshold`` and ``compressed_threshold`` instead.  In that mode,
    ``threshold_crossing`` means that the two representations make different
    decisions under their respective operating thresholds.

    Query and gallery batching bounds every temporary score matrix.  The
    function rejects batch settings above ``max_pairwise_elements`` instead of
    allocating an unbounded all-pairs matrix.
    """

    family, profile = _validate_profile(compression_family, compression_profile)
    search_mode_value = str(search_mode).strip()
    if not search_mode_value:
        raise ValueError("search_mode must not be empty")
    origin_query = _as_float_matrix(original_queries, name="original_queries")
    origin_gallery = _as_float_matrix(original_gallery, name="original_gallery")
    compressed_query = _as_float_matrix(compressed_queries, name="compressed_queries")
    compressed_gallery_matrix = _as_float_matrix(
        compressed_gallery,
        name="compressed_gallery",
    )

    if origin_query.shape[1] != origin_gallery.shape[1]:
        raise ValueError("original query and gallery dimensions must match")
    if compressed_query.shape[1] != compressed_gallery_matrix.shape[1]:
        raise ValueError("compressed query and gallery dimensions must match")
    if origin_query.shape[0] != compressed_query.shape[0]:
        raise ValueError("original and compressed query row counts must match")
    if origin_gallery.shape[0] != compressed_gallery_matrix.shape[0]:
        raise ValueError("original and compressed gallery row counts must match")

    top_k_value = _positive_integer(top_k, name="top_k")
    query_batch = _positive_integer(query_batch_size, name="query_batch_size")
    gallery_batch = _positive_integer(gallery_batch_size, name="gallery_batch_size")
    pair_limit = _positive_integer(max_pairwise_elements, name="max_pairwise_elements")
    if top_k_value > len(origin_gallery):
        raise ValueError("top_k must not exceed the gallery row count")
    effective_query_batch = min(query_batch, len(origin_query))
    effective_gallery_batch = min(gallery_batch, len(origin_gallery))
    largest_score_block = effective_query_batch * effective_gallery_batch
    largest_merge_block = effective_query_batch * (
        effective_gallery_batch + top_k_value
    )
    if max(largest_score_block, largest_merge_block) > pair_limit:
        raise ValueError(
            "batched retrieval working set exceeds max_pairwise_elements "
            f"for this input ({largest_merge_block} > {pair_limit}); reduce "
            "query_batch_size, gallery_batch_size, or top_k"
        )
    progress_offset_value = int(progress_offset)
    if progress_offset_value < 0:
        raise ValueError("progress_offset must be non-negative")
    required_progress_total = progress_offset_value + 2 * len(origin_query)
    progress_total_value = (
        required_progress_total
        if progress_total is None
        else int(progress_total)
    )
    if progress_total_value < required_progress_total:
        raise ValueError(
            "progress_total must cover both origin and compressed retrieval"
        )

    if threshold is not None:
        if origin_threshold is not None or compressed_threshold is not None:
            raise ValueError(
                "threshold cannot be combined with origin_threshold or "
                "compressed_threshold"
            )
        threshold_value = _cosine_threshold(threshold, name="threshold")
        origin_threshold_value = threshold_value
        compressed_threshold_value = threshold_value
    elif origin_threshold is None and compressed_threshold is None:
        threshold_value = None
        origin_threshold_value = None
        compressed_threshold_value = None
    elif origin_threshold is None or compressed_threshold is None:
        raise ValueError(
            "origin_threshold and compressed_threshold must be provided together"
        )
    else:
        origin_threshold_value = _cosine_threshold(
            origin_threshold,
            name="origin_threshold",
        )
        compressed_threshold_value = _cosine_threshold(
            compressed_threshold,
            name="compressed_threshold",
        )
        threshold_value = (
            origin_threshold_value
            if origin_threshold_value == compressed_threshold_value
            else None
        )

    query_identifiers = _as_identifier_array(
        query_ids,
        name="query_ids",
        length=len(origin_query),
        require_unique=True,
    )
    gallery_identifiers = _as_identifier_array(
        gallery_ids,
        name="gallery_ids",
        length=len(origin_gallery),
        require_unique=True,
    )
    query_identities = _as_identifier_array(
        query_identity_ids,
        name="query_identity_ids",
        length=len(origin_query),
    )
    gallery_identities = _as_identifier_array(
        gallery_identity_ids,
        name="gallery_identity_ids",
        length=len(origin_gallery),
    )

    origin_query_unit = _row_normalize(origin_query, name="original_queries")
    origin_gallery_unit = _row_normalize(origin_gallery, name="original_gallery")
    compressed_query_unit = _row_normalize(
        compressed_query,
        name="compressed_queries",
    )
    compressed_gallery_unit = _row_normalize(
        compressed_gallery_matrix,
        name="compressed_gallery",
    )

    origin_search_started = perf_counter()
    origin_indices, origin_scores = _batched_cosine_top_k(
        origin_query_unit,
        origin_gallery_unit,
        top_k=top_k_value,
        query_batch_size=query_batch,
        gallery_batch_size=gallery_batch,
        progress=progress,
        progress_message=progress_message,
        progress_offset=progress_offset_value,
        progress_total=progress_total_value,
        progress_details={
            **(progress_details or {}),
            "representation": "origin",
        },
    )
    origin_search_elapsed = perf_counter() - origin_search_started
    compressed_search_started = perf_counter()
    compressed_indices, compressed_scores = _batched_cosine_top_k(
        compressed_query_unit,
        compressed_gallery_unit,
        top_k=top_k_value,
        query_batch_size=query_batch,
        gallery_batch_size=gallery_batch,
        progress=progress,
        progress_message=progress_message,
        progress_offset=progress_offset_value + len(origin_query),
        progress_total=progress_total_value,
        progress_details={
            **(progress_details or {}),
            "representation": "compressed",
        },
    )
    compressed_search_elapsed = perf_counter() - compressed_search_started
    compressed_search_ms = float(compressed_search_elapsed * 1000.0)
    origin_search_ms = float(origin_search_elapsed * 1000.0)
    compressed_search_ms_per_query = float(
        compressed_search_ms / len(origin_query)
    )
    compressed_search_qps = float(
        len(origin_query) / compressed_search_elapsed
        if compressed_search_elapsed > 0.0
        else np.inf
    )

    row_indices = np.arange(len(origin_query))
    origin_top1_indices = origin_indices[:, 0]
    compressed_top1_indices = compressed_indices[:, 0]
    compressed_score_at_origin_top1 = np.sum(
        compressed_query_unit
        * compressed_gallery_unit[origin_top1_indices],
        axis=1,
        dtype=np.float64,
    )
    origin_score_at_compressed_top1 = np.sum(
        origin_query_unit * origin_gallery_unit[compressed_top1_indices],
        axis=1,
        dtype=np.float64,
    )

    gallery_identity_set = set(gallery_identities.tolist())
    records: list[dict[str, Any]] = []
    for row_index in row_indices:
        query_identity = query_identities[row_index]
        origin_gallery_rows = tuple(gallery_identifiers[origin_indices[row_index]].tolist())
        compressed_gallery_rows = tuple(
            gallery_identifiers[compressed_indices[row_index]].tolist()
        )
        origin_identity_ranking = tuple(
            gallery_identities[origin_indices[row_index]].tolist()
        )
        compressed_identity_ranking = tuple(
            gallery_identities[compressed_indices[row_index]].tolist()
        )
        origin_top1_identity = origin_identity_ranking[0]
        compressed_top1_identity = compressed_identity_ranking[0]
        origin_rank1_correct = origin_top1_identity == query_identity
        compressed_rank1_correct = compressed_top1_identity == query_identity
        is_mated = query_identity in gallery_identity_set
        origin_top1_score = float(origin_scores[row_index, 0])
        compressed_top1_score = float(compressed_scores[row_index, 0])

        if origin_threshold_value is None:
            origin_accepted: bool | None = None
            compressed_accepted: bool | None = None
            crossing: bool | None = None
            crossing_direction: str | None = None
            origin_decision_correct: bool | None = None
            compressed_decision_correct: bool | None = None
        else:
            origin_accepted = origin_top1_score >= origin_threshold_value
            compressed_accepted = (
                compressed_top1_score >= compressed_threshold_value
            )
            crossing = origin_accepted != compressed_accepted
            if origin_accepted and not compressed_accepted:
                crossing_direction = "accept_to_reject"
            elif not origin_accepted and compressed_accepted:
                crossing_direction = "reject_to_accept"
            else:
                crossing_direction = "none"
            origin_decision_correct = (
                origin_accepted and origin_rank1_correct
                if is_mated
                else not origin_accepted
            )
            compressed_decision_correct = (
                compressed_accepted and compressed_rank1_correct
                if is_mated
                else not compressed_accepted
            )

        records.append(
            {
                "query_id": query_identifiers[row_index],
                "query_identity_id": query_identity,
                "is_mated": bool(is_mated),
                "compression_family": family,
                "compression_profile": profile,
                "top_k": top_k_value,
                "search_mode": search_mode_value,
                "origin_score_space": "cosine_similarity",
                "compressed_score_space": "cosine_similarity",
                "score_spaces_comparable": True,
                "frozen_origin_threshold_applicable": True,
                "latency_measurement_repeats": 1,
                "latency_timer": "time.perf_counter",
                "compressed_index_build_latency_ms": np.nan,
                "compressed_gallery_add_latency_ms": np.nan,
                "compressed_gallery_encode_latency_ms": np.nan,
                "origin_search_latency_ms_total": origin_search_ms,
                "compressed_search_latency_ms_total": compressed_search_ms,
                "compressed_search_latency_ms_per_query": (
                    compressed_search_ms_per_query
                ),
                "compressed_search_queries_per_second": compressed_search_qps,
                "origin_top1_gallery_id": origin_gallery_rows[0],
                "compressed_top1_gallery_id": compressed_gallery_rows[0],
                "origin_top1_identity_id": origin_top1_identity,
                "compressed_top1_identity_id": compressed_top1_identity,
                "origin_top1_score": origin_top1_score,
                "compressed_top1_score": compressed_top1_score,
                "compressed_score_at_origin_top1": float(
                    compressed_score_at_origin_top1[row_index]
                ),
                "origin_score_at_compressed_top1": float(
                    origin_score_at_compressed_top1[row_index]
                ),
                "top1_score_drift": compressed_top1_score - origin_top1_score,
                "origin_winner_score_drift": float(
                    compressed_score_at_origin_top1[row_index] - origin_top1_score
                ),
                "agreement_with_origin": bool(
                    compressed_top1_identity == origin_top1_identity
                ),
                "top_k_identity_agreement": bool(
                    compressed_identity_ranking == origin_identity_ranking
                ),
                "top_k_identity_jaccard": _identity_jaccard(
                    origin_identity_ranking,
                    compressed_identity_ranking,
                ),
                "origin_rank1_correct": bool(origin_rank1_correct),
                "compressed_rank1_correct": bool(compressed_rank1_correct),
                "origin_top_k_correct": bool(query_identity in origin_identity_ranking),
                "compressed_top_k_correct": bool(
                    query_identity in compressed_identity_ranking
                ),
                "decision_threshold": threshold_value,
                "origin_decision_threshold": origin_threshold_value,
                "compressed_decision_threshold": compressed_threshold_value,
                "origin_accepted": origin_accepted,
                "compressed_accepted": compressed_accepted,
                "threshold_crossing": crossing,
                "threshold_crossing_direction": crossing_direction,
                "origin_decision_correct": origin_decision_correct,
                "compressed_decision_correct": compressed_decision_correct,
                "origin_top_k_gallery_ids": origin_gallery_rows,
                "compressed_top_k_gallery_ids": compressed_gallery_rows,
                "origin_top_k_identity_ids": origin_identity_ranking,
                "compressed_top_k_identity_ids": compressed_identity_ranking,
                "origin_top_k_scores": tuple(
                    float(value) for value in origin_scores[row_index]
                ),
                "compressed_top_k_scores": tuple(
                    float(value) for value in compressed_scores[row_index]
                ),
                "origin_fallback_used": False,
            }
        )

    return pd.DataFrame.from_records(records, columns=RETRIEVAL_COMPARISON_COLUMNS)


def compare_pq_adc_retrieval(
    origin_comparison: pd.DataFrame,
    original_queries: np.ndarray,
    original_gallery: np.ndarray,
    adc_distances: np.ndarray,
    adc_indices: np.ndarray,
    *,
    query_ids: Sequence[Any] | np.ndarray,
    gallery_ids: Sequence[Any] | np.ndarray,
    query_identity_ids: Sequence[Any] | np.ndarray,
    gallery_identity_ids: Sequence[Any] | np.ndarray,
    compression_profile: str,
    search_metrics: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Compare origin cosine ranking with exhaustive PQ ADC squared-L2 ranking.

    ADC distances are transformed to negative squared L2 only so larger values
    remain better for threshold calibration. They are never treated as cosine,
    and cross-space score drift fields remain missing by construction.
    """

    _, profile = _validate_profile("pq", compression_profile)
    query_matrix = _as_float_matrix(original_queries, name="original_queries")
    gallery_matrix = _as_float_matrix(original_gallery, name="original_gallery")
    if query_matrix.shape[1] != gallery_matrix.shape[1]:
        raise ValueError("original query and gallery dimensions must match")
    distances = np.asarray(adc_distances, dtype=np.float32)
    indices = np.asarray(adc_indices, dtype=np.int64)
    if distances.ndim != 2 or indices.shape != distances.shape:
        raise ValueError("ADC distances and indices must be equal-shape 2D arrays")
    if distances.shape[0] != len(query_matrix) or distances.shape[1] == 0:
        raise ValueError("ADC result shape must match the query count and top-k")
    if not np.all(np.isfinite(distances)) or np.any(distances < -1e-5):
        raise ValueError("ADC squared-L2 distances must be finite and non-negative")
    if np.any(indices < 0) or np.any(indices >= len(gallery_matrix)):
        raise ValueError("ADC indices are outside the gallery")
    if len(origin_comparison) != len(query_matrix):
        raise ValueError("origin comparison row count must match ADC queries")

    query_identifiers = _as_identifier_array(
        query_ids,
        name="query_ids",
        length=len(query_matrix),
        require_unique=True,
    )
    gallery_identifiers = _as_identifier_array(
        gallery_ids,
        name="gallery_ids",
        length=len(gallery_matrix),
        require_unique=True,
    )
    query_identities = _as_identifier_array(
        query_identity_ids,
        name="query_identity_ids",
        length=len(query_matrix),
    )
    gallery_identities = _as_identifier_array(
        gallery_identity_ids,
        name="gallery_identity_ids",
        length=len(gallery_matrix),
    )
    if not np.array_equal(
        origin_comparison["query_id"].astype(str).to_numpy(),
        query_identifiers.astype(str),
    ):
        raise ValueError("origin comparison query order differs from ADC queries")
    top_k_value = int(distances.shape[1])
    if not origin_comparison["top_k"].eq(top_k_value).all():
        raise ValueError("origin comparison and ADC top-k values differ")

    original_query_unit = _row_normalize(query_matrix, name="original_queries")
    original_gallery_unit = _row_normalize(gallery_matrix, name="original_gallery")
    origin_score_at_adc_top1 = np.sum(
        original_query_unit * original_gallery_unit[indices[:, 0]],
        axis=1,
        dtype=np.float64,
    )
    adc_scores = -distances.astype(np.float64)
    metrics = dict(search_metrics or {})
    latency_repeats = int(metrics.get("latency_measurement_repeats", 1))
    latency_timer = str(metrics.get("latency_timer", "not_measured"))
    index_build_latency = float(
        metrics.get("compressed_index_build_latency_ms", np.nan)
    )
    gallery_add_latency = float(
        metrics.get("compressed_gallery_add_latency_ms", np.nan)
    )
    gallery_encode_latency = float(
        metrics.get("compressed_gallery_encode_latency_ms", np.nan)
    )
    search_latency = float(
        metrics.get("compressed_search_latency_ms_total", np.nan)
    )
    search_latency_per_query = float(
        metrics.get("compressed_search_latency_ms_per_query", np.nan)
    )
    search_qps = float(
        metrics.get("compressed_search_queries_per_second", np.nan)
    )
    gallery_identity_set = set(gallery_identities.tolist())
    records: list[dict[str, Any]] = []
    for row_index in range(len(query_matrix)):
        origin_row = origin_comparison.iloc[row_index]
        query_identity = query_identities[row_index]
        origin_gallery_rows = tuple(origin_row["origin_top_k_gallery_ids"])
        origin_identity_ranking = tuple(origin_row["origin_top_k_identity_ids"])
        origin_score_ranking = tuple(origin_row["origin_top_k_scores"])
        adc_gallery_rows = tuple(
            gallery_identifiers[indices[row_index]].tolist()
        )
        adc_identity_ranking = tuple(
            gallery_identities[indices[row_index]].tolist()
        )
        adc_score_ranking = tuple(float(value) for value in adc_scores[row_index])
        origin_top1_identity = origin_identity_ranking[0]
        adc_top1_identity = adc_identity_ranking[0]
        is_mated = query_identity in gallery_identity_set
        records.append(
            {
                "query_id": query_identifiers[row_index],
                "query_identity_id": query_identity,
                "is_mated": bool(is_mated),
                "compression_family": "pq",
                "compression_profile": profile,
                "top_k": top_k_value,
                "search_mode": "pq_adc_exhaustive",
                "origin_score_space": "cosine_similarity",
                "compressed_score_space": "negative_squared_l2_adc",
                "score_spaces_comparable": False,
                "frozen_origin_threshold_applicable": False,
                "latency_measurement_repeats": latency_repeats,
                "latency_timer": latency_timer,
                "compressed_index_build_latency_ms": index_build_latency,
                "compressed_gallery_add_latency_ms": gallery_add_latency,
                "compressed_gallery_encode_latency_ms": gallery_encode_latency,
                "origin_search_latency_ms_total": float(
                    origin_row["origin_search_latency_ms_total"]
                ),
                "compressed_search_latency_ms_total": search_latency,
                "compressed_search_latency_ms_per_query": (
                    search_latency_per_query
                ),
                "compressed_search_queries_per_second": search_qps,
                "origin_top1_gallery_id": origin_gallery_rows[0],
                "compressed_top1_gallery_id": adc_gallery_rows[0],
                "origin_top1_identity_id": origin_top1_identity,
                "compressed_top1_identity_id": adc_top1_identity,
                "origin_top1_score": float(origin_row["origin_top1_score"]),
                "compressed_top1_score": float(adc_scores[row_index, 0]),
                "compressed_score_at_origin_top1": np.nan,
                "origin_score_at_compressed_top1": float(
                    origin_score_at_adc_top1[row_index]
                ),
                "top1_score_drift": np.nan,
                "origin_winner_score_drift": np.nan,
                "agreement_with_origin": bool(
                    adc_top1_identity == origin_top1_identity
                ),
                "top_k_identity_agreement": bool(
                    adc_identity_ranking == origin_identity_ranking
                ),
                "top_k_identity_jaccard": _identity_jaccard(
                    origin_identity_ranking,
                    adc_identity_ranking,
                ),
                "origin_rank1_correct": bool(
                    origin_top1_identity == query_identity
                ),
                "compressed_rank1_correct": bool(
                    adc_top1_identity == query_identity
                ),
                "origin_top_k_correct": bool(
                    query_identity in origin_identity_ranking
                ),
                "compressed_top_k_correct": bool(
                    query_identity in adc_identity_ranking
                ),
                "decision_threshold": None,
                "origin_decision_threshold": None,
                "compressed_decision_threshold": None,
                "origin_accepted": None,
                "compressed_accepted": None,
                "threshold_crossing": None,
                "threshold_crossing_direction": None,
                "origin_decision_correct": None,
                "compressed_decision_correct": None,
                "origin_top_k_gallery_ids": origin_gallery_rows,
                "compressed_top_k_gallery_ids": adc_gallery_rows,
                "origin_top_k_identity_ids": origin_identity_ranking,
                "compressed_top_k_identity_ids": adc_identity_ranking,
                "origin_top_k_scores": origin_score_ranking,
                "compressed_top_k_scores": adc_score_ranking,
                "origin_fallback_used": False,
            }
        )
    return pd.DataFrame.from_records(records, columns=RETRIEVAL_COMPARISON_COLUMNS)


def apply_retrieval_thresholds(
    comparison: pd.DataFrame,
    *,
    origin_threshold: float,
    compressed_threshold: float,
) -> pd.DataFrame:
    """Apply a new threshold policy without repeating cosine retrieval."""

    required = {
        "is_mated",
        "origin_top1_score",
        "compressed_top1_score",
        "origin_rank1_correct",
        "compressed_rank1_correct",
    }
    missing = sorted(required.difference(comparison.columns))
    if missing:
        raise ValueError(f"retrieval comparison is missing columns: {missing}")
    origin_value = _cosine_threshold(
        origin_threshold,
        name="origin_threshold",
    )
    compressed_value = float(compressed_threshold)
    if not np.isfinite(compressed_value):
        raise ValueError("compressed_threshold must be finite")
    compressed_spaces = (
        comparison["compressed_score_space"].astype(str).unique().tolist()
        if "compressed_score_space" in comparison
        else ["cosine_similarity"]
    )
    if compressed_spaces == ["cosine_similarity"]:
        compressed_value = _cosine_threshold(
            compressed_value,
            name="compressed_threshold",
        )
    result = comparison.copy()
    origin_accepted = (
        result["origin_top1_score"].to_numpy(dtype=float) >= origin_value
    )
    compressed_accepted = (
        result["compressed_top1_score"].to_numpy(dtype=float)
        >= compressed_value
    )
    mated = result["is_mated"].to_numpy(dtype=bool)
    origin_correct = result["origin_rank1_correct"].to_numpy(dtype=bool)
    compressed_correct = result["compressed_rank1_correct"].to_numpy(dtype=bool)
    crossing = origin_accepted != compressed_accepted

    result["decision_threshold"] = (
        origin_value if origin_value == compressed_value else None
    )
    result["origin_decision_threshold"] = origin_value
    result["compressed_decision_threshold"] = compressed_value
    result["origin_accepted"] = origin_accepted
    result["compressed_accepted"] = compressed_accepted
    result["threshold_crossing"] = crossing
    result["threshold_crossing_direction"] = np.select(
        [
            origin_accepted & ~compressed_accepted,
            ~origin_accepted & compressed_accepted,
        ],
        ["accept_to_reject", "reject_to_accept"],
        default="none",
    )
    result["origin_decision_correct"] = np.where(
        mated,
        origin_accepted & origin_correct,
        ~origin_accepted,
    )
    result["compressed_decision_correct"] = np.where(
        mated,
        compressed_accepted & compressed_correct,
        ~compressed_accepted,
    )
    return result.loc[:, RETRIEVAL_COMPARISON_COLUMNS]
