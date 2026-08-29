from __future__ import annotations

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
            record: dict[str, Any] = {
                "query_index": global_index,
                "query_image_id": str(image_ids[global_index]),
                "identity_id": str(query_ids[global_index]),
                "relevant_gallery_count": int(len(indexes)),
                "first_positive_rank": first_rank,
                "average_precision": _average_precision_from_positive_ranks(ranks),
            }
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
    bootstrap_seed: int = 42,
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
