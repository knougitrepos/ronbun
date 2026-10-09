from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd

from research.evaluation.metrics import (
    paired_binary_rate_difference_bootstrap_interval,
    wilson_score_interval,
)
from research.runtime.hashing import sha256_file


TinyFaceScoreKind = Literal["cosine", "negative_squared_l2"]
TINYFACE_RANKS = (1, 5, 10, 20)
TINYFACE_NATIVE_PQ_AUDIT_BOOLEAN_COLUMNS = tuple(
    column
    for rank in TINYFACE_RANKS
    for column in (
        f"native_rank_{rank}_success",
        f"decoded_native_rank_{rank}_mismatch",
    )
)


@dataclass(frozen=True)
class TinyFaceIdentificationResult:
    per_query: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class TinyFaceCompletedEvaluation:
    root: Path
    manifest: dict[str, Any]
    condition_summary: pd.DataFrame
    per_query: pd.DataFrame


def normalize_tinyface_per_query_audit_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Use nullable booleans for audit columns absent from non-native searches."""

    normalized = frame.copy()
    for column in TINYFACE_NATIVE_PQ_AUDIT_BOOLEAN_COLUMNS:
        if column not in normalized.columns:
            continue
        try:
            normalized[column] = normalized[column].astype("boolean")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TinyFace per-query audit column {column!r} must contain only "
                "boolean or missing values"
            ) from exc
    return normalized


def _matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) == 0 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a non-empty finite 2D float matrix")
    return matrix


def _identities(values: list[str] | tuple[str, ...] | np.ndarray, *, count: int, name: str) -> np.ndarray:
    identities = np.asarray(values, dtype=object).reshape(-1)
    if len(identities) != count:
        raise ValueError(f"{name} length does not match its embedding matrix")
    normalized = np.asarray([str(value).strip() for value in identities], dtype=object)
    if any(not value for value in normalized):
        raise ValueError(f"{name} contains an empty identity")
    return normalized


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ValueError("cosine vectors must be non-zero")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _score_block(
    queries: np.ndarray,
    gallery: np.ndarray,
    *,
    score_kind: TinyFaceScoreKind,
) -> np.ndarray:
    products = queries @ gallery.T
    if score_kind == "cosine":
        return products
    if score_kind == "negative_squared_l2":
        query_sq = np.sum(queries * queries, axis=1, keepdims=True)
        gallery_sq = np.sum(gallery * gallery, axis=1, keepdims=True).T
        return -(query_sq + gallery_sq - 2.0 * products)
    raise ValueError(f"unsupported TinyFace score_kind: {score_kind!r}")


def _average_precision_from_positive_ranks(positive_ranks: np.ndarray) -> float:
    """Match the trapezoidal AP integration in TinyFace's MATLAB evaluator."""

    ranks = np.sort(np.asarray(positive_ranks, dtype=np.int64))
    if ranks.ndim != 1 or len(ranks) == 0 or np.any(ranks < 1):
        raise ValueError("positive_ranks must contain positive 1-based ranks")
    ap = 0.0
    relevant_count = len(ranks)
    for found, rank in enumerate(ranks, start=1):
        previous_precision = 1.0 if int(rank) == 1 else (found - 1) / (int(rank) - 1)
        precision = found / int(rank)
        ap += (1.0 / relevant_count) * (previous_precision + precision) / 2.0
    return float(ap)


def evaluate_tinyface_identification(
    query_vectors: np.ndarray,
    gallery_vectors: np.ndarray,
    *,
    query_identity_ids: list[str] | tuple[str, ...] | np.ndarray,
    gallery_identity_ids: list[str] | tuple[str, ...] | np.ndarray,
    query_image_ids: list[str] | tuple[str, ...] | np.ndarray | None = None,
    query_fiqa_scores: Mapping[str, float] | Sequence[float] | None = None,
    score_kind: TinyFaceScoreKind = "cosine",
    query_batch_size: int = 32,
    gallery_batch_size: int = 8_192,
    compute_device: str = "cpu",
) -> TinyFaceIdentificationResult:
    """Evaluate all official positives against the full distractor gallery.

    Ranking uses a deterministic stable-index tie break and retains no full
    query-by-gallery score matrix.  This makes the official 3,728 x 157,871
    comparison practical on the project's 64 GB workstation.
    """

    queries = _matrix(query_vectors, name="query_vectors")
    gallery = _matrix(gallery_vectors, name="gallery_vectors")
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery dimensions must match")
    query_ids = _identities(query_identity_ids, count=len(queries), name="query_identity_ids")
    gallery_ids = _identities(gallery_identity_ids, count=len(gallery), name="gallery_identity_ids")
    if query_image_ids is None:
        image_ids = np.asarray([f"query:{index}" for index in range(len(queries))], dtype=object)
    else:
        image_ids = _identities(query_image_ids, count=len(queries), name="query_image_ids")
    if query_fiqa_scores is not None and not isinstance(query_fiqa_scores, Mapping):
        fiqa_seq = np.asarray(query_fiqa_scores, dtype=np.float64)
        if len(fiqa_seq) != len(queries):
            raise ValueError(
                f"query_fiqa_scores length ({len(fiqa_seq)}) does not match query count ({len(queries)})"
            )
        if not np.isfinite(fiqa_seq).all():
            raise ValueError("query_fiqa_scores contains non-finite values")
    if isinstance(query_batch_size, bool) or int(query_batch_size) < 1:
        raise ValueError("query_batch_size must be a positive integer")
    if isinstance(gallery_batch_size, bool) or int(gallery_batch_size) < 1:
        raise ValueError("gallery_batch_size must be a positive integer")
    query_batch = int(query_batch_size)
    gallery_batch = int(gallery_batch_size)

    gallery_by_identity: dict[str, np.ndarray] = {}
    for identity in sorted(set(gallery_ids.tolist())):
        indexes = np.flatnonzero(gallery_ids == identity).astype(np.int64)
        gallery_by_identity[str(identity)] = indexes
    missing = sorted(set(query_ids.tolist()) - set(gallery_by_identity))
    if missing:
        raise ValueError(
            "TinyFace closed-set queries are missing matching gallery identities: "
            f"{missing[:5]}"
        )

    if score_kind == "cosine":
        queries = _normalize(queries)
        gallery = _normalize(gallery)
    device = str(compute_device).strip().lower()
    if not device:
        raise ValueError("compute_device must be non-empty")
    if device == "cpu":

        def score_slices(
            query_start: int,
            query_stop: int,
            gallery_start: int,
            gallery_stop: int,
        ) -> np.ndarray:
            return _score_block(
                queries[query_start:query_stop],
                gallery[gallery_start:gallery_stop],
                score_kind=score_kind,
            )

        def score_selected(query_index: int, indexes: np.ndarray) -> np.ndarray:
            return _score_block(
                queries[query_index : query_index + 1],
                gallery[indexes],
                score_kind=score_kind,
            ).reshape(-1)

        compute_backend = "numpy_cpu"
    else:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for TinyFace GPU ranking") from exc
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("TinyFace GPU ranking requested but CUDA is unavailable")
        torch_queries = torch.from_numpy(np.ascontiguousarray(queries)).to(torch_device)
        torch_gallery = torch.from_numpy(np.ascontiguousarray(gallery)).to(torch_device)

        def torch_scores(query_tensor: Any, gallery_tensor: Any) -> Any:
            products = query_tensor @ gallery_tensor.T
            if score_kind == "cosine":
                return products
            query_sq = torch.sum(query_tensor * query_tensor, dim=1, keepdim=True)
            gallery_sq = torch.sum(
                gallery_tensor * gallery_tensor, dim=1, keepdim=True
            ).T
            return -(query_sq + gallery_sq - 2.0 * products)

        def score_slices(
            query_start: int,
            query_stop: int,
            gallery_start: int,
            gallery_stop: int,
        ) -> np.ndarray:
            with torch.inference_mode():
                scores = torch_scores(
                    torch_queries[query_start:query_stop],
                    torch_gallery[gallery_start:gallery_stop],
                )
            return scores.detach().to("cpu").float().numpy()

        def score_selected(query_index: int, indexes: np.ndarray) -> np.ndarray:
            torch_indexes = torch.as_tensor(indexes, dtype=torch.long, device=torch_device)
            with torch.inference_mode():
                scores = torch_scores(
                    torch_queries[query_index : query_index + 1],
                    torch_gallery.index_select(0, torch_indexes),
                )
            return scores.detach().to("cpu").float().numpy().reshape(-1)

        compute_backend = f"pytorch_{torch_device.type}"
    started = perf_counter()
    records: list[dict[str, Any]] = []
    for query_start in range(0, len(queries), query_batch):
        query_stop = min(len(queries), query_start + query_batch)
        positive_indexes = [
            gallery_by_identity[str(identity)]
            for identity in query_ids[query_start:query_stop]
        ]
        positive_scores = [
            score_selected(query_start + local_index, indexes)
            for local_index, indexes in enumerate(positive_indexes)
        ]
        greater_counts = [np.zeros(len(indexes), dtype=np.int64) for indexes in positive_indexes]
        equal_earlier_counts = [np.zeros(len(indexes), dtype=np.int64) for indexes in positive_indexes]
        top1_scores = np.full(query_stop - query_start, -np.inf, dtype=np.float64)
        top2_scores = np.full(query_stop - query_start, -np.inf, dtype=np.float64)
        for gallery_start in range(0, len(gallery), gallery_batch):
            gallery_stop = min(len(gallery), gallery_start + gallery_batch)
            scores = score_slices(
                query_start,
                query_stop,
                gallery_start,
                gallery_stop,
            )
            block_indexes = np.arange(gallery_start, gallery_stop, dtype=np.int64)
            for local_index, (indexes, expected_scores) in enumerate(
                zip(positive_indexes, positive_scores, strict=True)
            ):
                row = scores[local_index]
                # Use the exact same stored positive score as the rank
                # threshold. BLAS may accumulate a 1xP and QxG product in a
                # slightly different order; replacing only the known positive
                # entries prevents a true match from outranking itself.
                for positive_index, positive_score in zip(
                    indexes, expected_scores, strict=True
                ):
                    if gallery_start <= positive_index < gallery_stop:
                        row[int(positive_index - gallery_start)] = positive_score

                # Update top1 and top2 running maximums from canonical scores
                if len(row) >= 2:
                    p_sorted = np.sort(np.partition(row, -2)[-2:])
                    cands = np.array([top1_scores[local_index], top2_scores[local_index], p_sorted[1], p_sorted[0]], dtype=np.float64)
                    top_two = np.sort(cands)[-2:]
                    top1_scores[local_index] = top_two[1]
                    top2_scores[local_index] = top_two[0]
                elif len(row) == 1:
                    cands = np.array([top1_scores[local_index], top2_scores[local_index], row[0]], dtype=np.float64)
                    top_two = np.sort(cands)[-2:]
                    top1_scores[local_index] = top_two[1]
                    top2_scores[local_index] = top_two[0]
                for positive_offset, (positive_index, positive_score) in enumerate(
                    zip(indexes, expected_scores, strict=True)
                ):
                    greater_counts[local_index][positive_offset] += int(
                        np.count_nonzero(row > positive_score)
                    )
                    equal_earlier_counts[local_index][positive_offset] += int(
                        np.count_nonzero(
                            (row == positive_score) & (block_indexes < positive_index)
                        )
                    )
        for local_index, indexes in enumerate(positive_indexes):
            global_index = query_start + local_index
            ranks = (
                1
                + greater_counts[local_index]
                + equal_earlier_counts[local_index]
            )
            first_rank = int(np.min(ranks))
            t1 = float(top1_scores[local_index])
            t2 = float(top2_scores[local_index])
            margin = float(t1 - t2) if np.isfinite(t1) and np.isfinite(t2) else 0.0
            record: dict[str, Any] = {
                "query_index": global_index,
                "query_image_id": str(image_ids[global_index]),
                "identity_id": str(query_ids[global_index]),
                "relevant_gallery_count": int(len(indexes)),
                "first_positive_rank": first_rank,
                "average_precision": _average_precision_from_positive_ranks(ranks),
                "top1_score": t1,
                "top2_score": t2,
                "score_margin": margin,
            }
            if query_fiqa_scores is not None:
                img_key = str(image_ids[global_index])
                if isinstance(query_fiqa_scores, Mapping):
                    record["fiqa_score"] = float(query_fiqa_scores.get(img_key, np.nan))
                else:
                    record["fiqa_score"] = float(query_fiqa_scores[global_index])
            for rank in TINYFACE_RANKS:
                record[f"rank_{rank}_success"] = bool(first_rank <= rank)
            records.append(record)
    elapsed = perf_counter() - started
    per_query = pd.DataFrame.from_records(records)
    total = int(len(per_query))
    summary: dict[str, Any] = {
        "protocol": "tinyface_official_closed_set_v1",
        "open_set_protocol": False,
        "fpir_tpir_metrics_applicable": False,
        "score_kind": score_kind,
        "query_count": total,
        "gallery_count": int(len(gallery)),
        "match_gallery_count": int(sum(len(value) for value in gallery_by_identity.values()) - sum(str(identity).startswith("tinyface:distractor:") for identity in gallery_ids)),
        "mean_average_precision": float(per_query["average_precision"].mean()),
        "mean_score_margin": float(per_query["score_margin"].mean()),
        "search_latency_ms_total": float(elapsed * 1_000.0),
        "search_latency_ms_per_query": float(elapsed * 1_000.0 / total),
        "search_queries_per_second": float(total / elapsed if elapsed > 0 else np.inf),
        "ranking_implementation": "streaming_exact_all_gallery_stable_index_ties",
        "compute_backend": compute_backend,
        "confidence_interval_contract": "probe_level_wilson_95",
    }
    for rank in TINYFACE_RANKS:
        successes = int(per_query[f"rank_{rank}_success"].sum())
        low, high = wilson_score_interval(successes, total)
        summary.update(
            {
                f"rank_{rank}": successes / total,
                f"rank_{rank}_success_count": successes,
                f"rank_{rank}_denominator": total,
                f"rank_{rank}_wilson95_low": low,
                f"rank_{rank}_wilson95_high": high,
            }
        )
    return TinyFaceIdentificationResult(per_query=per_query, summary=summary)


def paired_tinyface_deltas(
    origin: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    bootstrap_seed: int = 8972,
    bootstrap_repeats: int = 2_000,
) -> dict[str, Any]:
    """Return candidate-minus-origin paired probe bootstrap evidence."""

    key = "query_image_id"
    required = {key, "average_precision", *(f"rank_{rank}_success" for rank in TINYFACE_RANKS)}
    for name, frame in (("origin", origin), ("candidate", candidate)):
        missing = sorted(required - set(frame.columns))
        if missing or frame[key].duplicated().any():
            raise ValueError(f"{name} TinyFace rows are invalid: missing={missing}")
    joined = origin[list(required)].merge(
        candidate[list(required)], on=key, how="inner", validate="one_to_one", suffixes=("_origin", "_candidate")
    )
    if len(joined) != len(origin) or len(joined) != len(candidate):
        raise ValueError("origin/candidate TinyFace query sets differ")
    rng = np.random.default_rng(int(bootstrap_seed))
    ap_difference = (
        joined["average_precision_candidate"].to_numpy(dtype=np.float64)
        - joined["average_precision_origin"].to_numpy(dtype=np.float64)
    )
    draws = rng.integers(0, len(joined), size=(int(bootstrap_repeats), len(joined)))
    ap_bootstrap = ap_difference[draws].mean(axis=1)
    result: dict[str, Any] = {
        "compressed_minus_origin_map": float(ap_difference.mean()),
        "compressed_minus_origin_map_paired_bootstrap95_low": float(np.quantile(ap_bootstrap, 0.025)),
        "compressed_minus_origin_map_paired_bootstrap95_high": float(np.quantile(ap_bootstrap, 0.975)),
        "paired_bootstrap_resamples": int(bootstrap_repeats),
        "paired_bootstrap_random_seed": int(bootstrap_seed),
    }
    for rank in TINYFACE_RANKS:
        origin_success = joined[f"rank_{rank}_success_origin"].astype(bool).to_numpy()
        candidate_success = joined[f"rank_{rank}_success_candidate"].astype(bool).to_numpy()
        low, high = paired_binary_rate_difference_bootstrap_interval(
            int(origin_success.sum()),
            int(candidate_success.sum()),
            int(np.logical_and(origin_success, candidate_success).sum()),
            len(joined),
            resamples=int(bootstrap_repeats),
            random_seed=int(bootstrap_seed),
        )
        result.update(
            {
                f"compressed_minus_origin_rank_{rank}": float(candidate_success.mean() - origin_success.mean()),
                f"compressed_minus_origin_rank_{rank}_paired_bootstrap95_low": low,
                f"compressed_minus_origin_rank_{rank}_paired_bootstrap95_high": high,
            }
        )
    return result


def load_tinyface_completed_evaluation(run_dir: str | Path) -> TinyFaceCompletedEvaluation:
    root = Path(run_dir).expanduser().resolve()
    run_manifest_path = root / "run_manifest.json"
    completed = root / "COMPLETED"
    evaluation_root = root / "artifacts" / "tinyface_official"
    manifest_path = evaluation_root / "tinyface_evaluation_manifest.json"
    if not completed.is_file() or not run_manifest_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"TinyFace completed evaluation is incomplete: {root}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "completed":
        raise ValueError(f"TinyFace run is not completed: {root}")
    if manifest.get("artifact_type") != "tinyface_official_compression_evaluation_v1":
        raise ValueError(f"unexpected TinyFace evaluation artifact: {manifest_path}")
    if manifest.get("source_run_id") != run_manifest.get("run_id"):
        raise ValueError("TinyFace run/evaluation identity mismatch")
    if manifest.get("dataset_id") != "tinyface" or manifest.get("open_set_protocol") is not False:
        raise ValueError("TinyFace artifact escaped its official closed-set boundary")
    outputs = manifest.get("outputs", {})
    validated: dict[str, Path] = {}
    for name in ("condition_summary.csv", "per_query.csv"):
        entry = outputs.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"TinyFace manifest is missing output {name}")
        path = evaluation_root / str(entry.get("path", name))
        if (
            not path.is_file()
            or path.stat().st_size != int(entry.get("bytes", -1))
            or sha256_file(path) != str(entry.get("sha256", ""))
        ):
            raise ValueError(f"TinyFace output failed checksum validation: {path}")
        validated[name] = path
    condition_summary = pd.read_csv(validated["condition_summary.csv"])
    try:
        per_query = pd.read_csv(
            validated["per_query.csv"],
            dtype={
                column: "boolean"
                for column in TINYFACE_NATIVE_PQ_AUDIT_BOOLEAN_COLUMNS
            },
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "TinyFace per-query audit columns must contain only boolean or missing values"
        ) from exc
    per_query = normalize_tinyface_per_query_audit_dtypes(per_query)
    if condition_summary.empty or per_query.empty:
        raise ValueError("TinyFace evaluation tables must not be empty")
    if condition_summary["model_uid"].astype(str).nunique() != 1:
        raise ValueError("TinyFace condition summary must contain one model UID")
    if set(condition_summary.get("fpir_tpir_metrics_applicable", [])) != {False}:
        raise ValueError("TinyFace summary must not claim FPIR/TPIR applicability")
    return TinyFaceCompletedEvaluation(
        root=root,
        manifest=manifest,
        condition_summary=condition_summary,
        per_query=per_query,
    )


# =========================================================================
# EXTENSION MODULE: Optional Exploratory Fallback Analysis
# =========================================================================
# NOTE: The dataclass and functions below provide an optional exploratory fallback
# analysis, separated from the core TinyFace 1:N ranking evaluation protocol
# (Rank-1/5/10/20 & mAP). They are NOT part of the primary benchmark pipeline or
# mandatory workflow. The core TinyFace evaluation remains purely evaluate_tinyface_identification.
# Query-dependent thresholds or fallback routines do not alter official 1:N rank metrics.


@dataclass(frozen=True)
class TinyFaceFallbackResult:
    """Optional extension: TinyFace 1:N closed-set exact fallback rejection evaluation result."""

    per_query: pd.DataFrame
    summary: dict[str, Any]


def simulate_tinyface_fallback_rejection(
    origin_result: TinyFaceIdentificationResult | pd.DataFrame,
    compressed_result: TinyFaceIdentificationResult | pd.DataFrame,
    *,
    fiqa_scores: Mapping[str, float] | pd.Series | None = None,
    score_margins: Mapping[str, float] | pd.Series | None = None,
    fiqa_threshold: float | None = None,
    margin_threshold: float | None = None,
    fallback_budget_fraction: float | None = None,
    fallback_mask: Sequence[bool] | np.ndarray | pd.Series | None = None,
) -> TinyFaceFallbackResult:
    """Optional extension: simulate exact 512D fallback retrieval from compressed PQ ADC results.

    Separated from the core TinyFace 1:N ranking evaluation protocol.
    Uncertainty can be conditioned on query FIQA quality, retrieval score margin (top1 - top2),
    explicit thresholds, fallback budget fraction, or an explicit fallback mask.
    Computes resulting closed-set metrics (Rank-1, 5, 10, 20, mAP), fallback rate,
    recovery rate (복원율), and precision.
    """
    origin_df = (
        origin_result.per_query.copy()
        if isinstance(origin_result, TinyFaceIdentificationResult)
        else origin_result.copy()
    )
    compressed_df = (
        compressed_result.per_query.copy()
        if isinstance(compressed_result, TinyFaceIdentificationResult)
        else compressed_result.copy()
    )

    if origin_df.empty or compressed_df.empty:
        raise ValueError("origin and compressed results must not be empty")

    key = "query_image_id"
    required = {key, "average_precision", *(f"rank_{rank}_success" for rank in TINYFACE_RANKS)}
    for name, frame in (("origin", origin_df), ("compressed", compressed_df)):
        missing = sorted(required - set(frame.columns))
        if missing or frame[key].duplicated().any():
            raise ValueError(f"{name} TinyFace rows are invalid: missing={missing}")

    merged = pd.merge(
        origin_df,
        compressed_df,
        on=key,
        suffixes=("_origin", "_compressed"),
        validate="one_to_one",
    )
    if len(merged) != len(compressed_df) or len(merged) != len(origin_df):
        raise ValueError("origin and compressed queries must match one-to-one")

    n = len(merged)

    # Resolve FIQA scores
    if fiqa_scores is not None:
        if isinstance(fiqa_scores, (pd.Series, Mapping)):
            fiqa = merged[key].astype(str).map(fiqa_scores).to_numpy(dtype=np.float64)
        else:
            fiqa = np.asarray(fiqa_scores, dtype=np.float64)
            if len(fiqa) != n:
                raise ValueError(f"fiqa_scores length ({len(fiqa)}) does not match query count ({n})")
        if not np.isfinite(fiqa).all():
            raise ValueError("FIQA scores contain missing or non-finite values")
    elif "fiqa_score_compressed" in merged.columns:
        fiqa = merged["fiqa_score_compressed"].to_numpy(dtype=np.float64)
    elif "fiqa_score" in compressed_df.columns:
        fiqa = merged["fiqa_score"].to_numpy(dtype=np.float64)
    else:
        fiqa = None

    # Resolve score margins
    if score_margins is not None:
        if isinstance(score_margins, (pd.Series, Mapping)):
            margins = merged[key].astype(str).map(score_margins).to_numpy(dtype=np.float64)
        else:
            margins = np.asarray(score_margins, dtype=np.float64)
            if len(margins) != n:
                raise ValueError(f"score_margins length ({len(margins)}) does not match query count ({n})")
        if not np.isfinite(margins).all():
            raise ValueError("score margins contain missing or non-finite values")
    elif "score_margin_compressed" in merged.columns:
        margins = merged["score_margin_compressed"].to_numpy(dtype=np.float64)
    elif "score_margin" in compressed_df.columns:
        margins = merged["score_margin"].to_numpy(dtype=np.float64)
    else:
        margins = None

    # Determine fallback mask
    if fallback_mask is not None:
        mask = np.asarray(fallback_mask, dtype=bool)
        if len(mask) != n:
            raise ValueError(f"fallback_mask length ({len(mask)}) does not match query count ({n})")
        should_fallback = mask
    elif fallback_budget_fraction is not None:
        budget = float(fallback_budget_fraction)
        if not (0.0 <= budget <= 1.0):
            raise ValueError("fallback_budget_fraction must be in [0.0, 1.0]")
        k_fallback = int(np.ceil(budget * n))
        if k_fallback == 0:
            should_fallback = np.zeros(n, dtype=bool)
        elif k_fallback >= n:
            should_fallback = np.ones(n, dtype=bool)
        else:
            if margins is None and fiqa is None:
                raise ValueError("cannot compute fallback budget without fiqa_scores or score_margins")
            uncert = np.zeros(n, dtype=np.float64)
            if margins is not None:
                if not np.isfinite(margins).all():
                    raise ValueError("score margins contain non-finite values")
                m_scale = np.std(margins)
                m_scale = m_scale if m_scale > 1e-8 else 1.0
                uncert -= (margins - np.mean(margins)) / m_scale
            if fiqa is not None:
                if not np.isfinite(fiqa).all():
                    raise ValueError("FIQA scores contain non-finite values")
                q_scale = np.std(fiqa)
                q_scale = q_scale if q_scale > 1e-8 else 1.0
                uncert -= (fiqa - np.mean(fiqa)) / q_scale
            order = np.argsort(-uncert, kind="stable")
            should_fallback = np.zeros(n, dtype=bool)
            should_fallback[order[:k_fallback]] = True
    elif fiqa_threshold is not None or margin_threshold is not None:
        mask = np.zeros(n, dtype=bool)
        if fiqa_threshold is not None:
            if fiqa is None or not np.isfinite(fiqa).all():
                raise ValueError("fiqa_threshold provided but FIQA scores are missing or non-finite")
            mask |= (fiqa < float(fiqa_threshold))
        if margin_threshold is not None:
            if margins is None or not np.isfinite(margins).all():
                raise ValueError("margin_threshold provided but score margins are missing or non-finite")
            mask |= (margins < float(margin_threshold))
        should_fallback = mask
    else:
        should_fallback = np.zeros(n, dtype=bool)

    # Outcomes
    comp_rank = (
        merged["first_positive_rank_compressed"].to_numpy(dtype=np.int64)
        if "first_positive_rank_compressed" in merged.columns
        else merged["first_positive_rank"].to_numpy(dtype=np.int64)
        if "first_positive_rank" in merged.columns
        else np.ones(n, dtype=np.int64)
    )
    orig_rank = (
        merged["first_positive_rank_origin"].to_numpy(dtype=np.int64)
        if "first_positive_rank_origin" in merged.columns
        else merged["first_positive_rank"].to_numpy(dtype=np.int64)
        if "first_positive_rank" in merged.columns
        else np.ones(n, dtype=np.int64)
    )
    comp_ap = merged["average_precision_compressed"].to_numpy(dtype=np.float64)
    orig_ap = merged["average_precision_origin"].to_numpy(dtype=np.float64)

    effective_rank = np.where(should_fallback, orig_rank, comp_rank)
    effective_ap = np.where(should_fallback, orig_ap, comp_ap)

    per_query_data: dict[str, Any] = {
        "query_image_id": merged[key],
        "is_fallback": should_fallback,
        "compressed_first_positive_rank": comp_rank,
        "origin_first_positive_rank": orig_rank,
        "effective_first_positive_rank": effective_rank,
        "compressed_average_precision": comp_ap,
        "origin_average_precision": orig_ap,
        "effective_average_precision": effective_ap,
    }
    if fiqa is not None:
        per_query_data["fiqa_score"] = fiqa
    if margins is not None:
        per_query_data["score_margin"] = margins

    for r in TINYFACE_RANKS:
        comp_succ = merged[f"rank_{r}_success_compressed"].to_numpy(dtype=bool)
        orig_succ = merged[f"rank_{r}_success_origin"].to_numpy(dtype=bool)
        eff_succ = np.where(should_fallback, orig_succ, comp_succ)
        per_query_data[f"compressed_rank_{r}_success"] = comp_succ
        per_query_data[f"origin_rank_{r}_success"] = orig_succ
        per_query_data[f"effective_rank_{r}_success"] = eff_succ

    per_query_df = pd.DataFrame(per_query_data)

    fallback_count = int(np.sum(should_fallback))
    fallback_rate = float(fallback_count / n)

    comp_r1 = per_query_df["compressed_rank_1_success"].to_numpy()
    orig_r1 = per_query_df["origin_rank_1_success"].to_numpy()
    eff_r1 = per_query_df["effective_rank_1_success"].to_numpy()

    pq_r1_failures = ~comp_r1
    pq_r1_failure_count = int(np.sum(pq_r1_failures))
    fell_back_and_failed_on_pq = should_fallback & pq_r1_failures
    recovered_r1_count = int(np.sum(fell_back_and_failed_on_pq & orig_r1))
    recovery_rate = (
        float(recovered_r1_count / pq_r1_failure_count)
        if pq_r1_failure_count > 0
        else 0.0
    )
    fallback_precision = (
        float(recovered_r1_count / fallback_count)
        if fallback_count > 0
        else 0.0
    )
    unnecessary_fallback_count = int(np.sum(should_fallback & comp_r1))
    regressed_r1_count = int(np.sum(should_fallback & comp_r1 & ~orig_r1))

    summary: dict[str, Any] = {
        "protocol": "tinyface_exact_fallback_rejection_v1",
        "query_count": n,
        "fallback_query_count": fallback_count,
        "fallback_rate": fallback_rate,
        "pq_rank_1_failure_count": pq_r1_failure_count,
        "recovered_rank_1_count": recovered_r1_count,
        "recovery_rate": recovery_rate,
        "fallback_precision": fallback_precision,
        "unnecessary_fallback_count": unnecessary_fallback_count,
        "regressed_rank_1_count": regressed_r1_count,
        "effective_mean_average_precision": float(np.mean(effective_ap)),
        "compressed_mean_average_precision": float(np.mean(comp_ap)),
        "origin_mean_average_precision": float(np.mean(orig_ap)),
        "delta_map": float(np.mean(effective_ap) - np.mean(comp_ap)),
    }

    for r in TINYFACE_RANKS:
        eff_succ = per_query_df[f"effective_rank_{r}_success"].to_numpy()
        comp_succ = per_query_df[f"compressed_rank_{r}_success"].to_numpy()
        orig_succ = per_query_df[f"origin_rank_{r}_success"].to_numpy()
        eff_count = int(np.sum(eff_succ))
        low, high = wilson_score_interval(eff_count, n)
        summary[f"effective_rank_{r}"] = float(eff_count / n)
        summary[f"compressed_rank_{r}"] = float(np.sum(comp_succ) / n)
        summary[f"origin_rank_{r}"] = float(np.sum(orig_succ) / n)
        summary[f"delta_rank_{r}"] = float((eff_count - np.sum(comp_succ)) / n)
        summary[f"effective_rank_{r}_wilson95_low"] = low
        summary[f"effective_rank_{r}_wilson95_high"] = high

    return TinyFaceFallbackResult(per_query=per_query_df, summary=summary)


def simulate_tinyface_fallback_sweep(
    origin_result: TinyFaceIdentificationResult | pd.DataFrame,
    compressed_result: TinyFaceIdentificationResult | pd.DataFrame,
    *,
    budgets: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0),
    fiqa_scores: Mapping[str, float] | pd.Series | None = None,
    score_margins: Mapping[str, float] | pd.Series | None = None,
) -> pd.DataFrame:
    """Sweep fallback budget fractions and evaluate resulting TinyFace metrics."""
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        res = simulate_tinyface_fallback_rejection(
            origin_result,
            compressed_result,
            fiqa_scores=fiqa_scores,
            score_margins=score_margins,
            fallback_budget_fraction=float(budget),
        )
        row = {
            "fallback_budget": float(budget),
            "fallback_rate": res.summary["fallback_rate"],
            "fallback_query_count": res.summary["fallback_query_count"],
            "recovered_rank_1_count": res.summary["recovered_rank_1_count"],
            "recovery_rate": res.summary["recovery_rate"],
            "fallback_precision": res.summary["fallback_precision"],
            "unnecessary_fallback_count": res.summary["unnecessary_fallback_count"],
            "regressed_rank_1_count": res.summary["regressed_rank_1_count"],
            "effective_mean_average_precision": res.summary["effective_mean_average_precision"],
            "delta_map": res.summary["delta_map"],
        }
        for r in TINYFACE_RANKS:
            row[f"effective_rank_{r}"] = res.summary[f"effective_rank_{r}"]
            row[f"delta_rank_{r}"] = res.summary[f"delta_rank_{r}"]
        rows.append(row)
    return pd.DataFrame(rows)

