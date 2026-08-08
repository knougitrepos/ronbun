from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 3
COMPRESSION_METRICS = (
    "reconstruction_mse",
    "angular_error_rad",
    "cosine_to_origin",
)
COMPRESSION_FIXED_COLUMNS = (
    "origin_dimension",
    "search_dimension",
    "reconstruction_available",
    "metric_vector_source",
    "storage_bytes_per_embedding",
    "codebook_bytes",
    "codebook_bytes_source",
    "codec_parameter_bytes",
    "codec_parameter_bytes_source",
    "extraction_uid",
    "dataset_id",
    "origin_embedding_artifact_uid",
)
RETRIEVAL_FIXED_COLUMNS = (
    "top_k",
    "protocol_uid",
    "threshold_source_split",
    "evaluation_split",
    "storage_bytes_per_embedding",
    "codebook_bytes",
    "codebook_bytes_source",
    "codec_parameter_bytes",
    "codec_parameter_bytes_source",
    "extraction_uid",
    "dataset_id",
    "origin_embedding_artifact_uid",
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
    "gallery_template_count",
)
OPTIONAL_RETRIEVAL_FIXED_COLUMNS = (
    "codec_parameter_bytes",
    "codec_parameter_bytes_source",
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
    "gallery_template_count",
)
THRESHOLD_COLUMNS = (
    "decision_threshold",
    "origin_decision_threshold",
    "compressed_decision_threshold",
)
RETRIEVAL_BOOLEAN_COLUMNS = (
    "is_mated",
    "origin_rank1_correct",
    "compressed_rank1_correct",
    "origin_top_k_correct",
    "compressed_top_k_correct",
    "origin_accepted",
    "compressed_accepted",
    "threshold_crossing",
    "origin_decision_correct",
    "compressed_decision_correct",
    "agreement_with_origin",
    "origin_fallback_used",
)
RETRIEVAL_NUMERIC_COLUMNS = (
    "top_k_identity_jaccard",
    "top1_score_drift",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        if series.isna().any():
            raise ValueError("boolean column contains missing values")
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    if normalized.isna().any():
        raise ValueError("boolean column contains missing values")
    invalid = normalized[~normalized.isin(("true", "false")) & normalized.notna()]
    if not invalid.empty:
        raise ValueError(
            f"boolean column contains unexpected values: {invalid.unique()[:5].tolist()}"
        )
    return normalized.eq("true")


def _portable_path(path: Path, *, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _add_fixed_values(
    target: dict[str, set[Any]],
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    for column in columns:
        values = frame[column].drop_duplicates().tolist()
        normalized = {
            None
            if pd.isna(value)
            else value.item()
            if isinstance(value, np.generic)
            else value
            for value in values
        }
        target[column].update(normalized)


def _one(values: set[Any], *, name: str) -> Any:
    if len(values) != 1:
        raise ValueError(f"{name} must be constant within a summary group: {values}")
    return next(iter(values))


def _mean(total: float, count: int) -> float:
    return float(total / count) if count else float("nan")


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def summarize_compression(
    source_path: Path | None,
    *,
    chunksize: int,
    source_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, int]:
    requested_usecols = (
        "compression_family",
        "compression_profile",
        "origin_fallback_used",
        *COMPRESSION_METRICS,
        *COMPRESSION_FIXED_COLUMNS,
    )
    if (source_path is None) == (source_frame is None):
        raise ValueError("provide exactly one of source_path or source_frame")
    source_columns = set(
        source_frame.columns
        if source_frame is not None
        else pd.read_csv(source_path, nrows=0).columns
    )
    usecols = tuple(
        column for column in requested_usecols if column in source_columns
    )
    accumulators: dict[tuple[str, str], dict[str, Any]] = {}
    row_count = 0
    chunks = (
        (source_frame.loc[:, usecols].copy(),)
        if source_frame is not None
        else pd.read_csv(source_path, usecols=usecols, chunksize=chunksize)
    )
    for chunk in chunks:
        row_count += int(len(chunk))
        if "codec_parameter_bytes" not in chunk:
            chunk["codec_parameter_bytes"] = chunk["codebook_bytes"]
        if "codec_parameter_bytes_source" not in chunk:
            chunk["codec_parameter_bytes_source"] = chunk[
                "codebook_bytes_source"
            ]
        for key, group in chunk.groupby(
            ["compression_family", "compression_profile"], sort=False
        ):
            normalized_key = (str(key[0]), str(key[1]))
            acc = accumulators.setdefault(
                normalized_key,
                {
                    "sample_count": 0,
                    "origin_fallback_count": 0,
                    "fixed": defaultdict(set),
                    "metric_sum": defaultdict(float),
                    "metric_count": defaultdict(int),
                    "metric_values": defaultdict(list),
                },
            )
            acc["sample_count"] += int(len(group))
            acc["origin_fallback_count"] += int(
                _as_bool(group["origin_fallback_used"]).sum()
            )
            _add_fixed_values(acc["fixed"], group, COMPRESSION_FIXED_COLUMNS)
            for metric in COMPRESSION_METRICS:
                values = (
                    pd.to_numeric(group[metric], errors="coerce")
                    .dropna()
                    .to_numpy(dtype=np.float64)
                )
                acc["metric_sum"][metric] += float(values.sum())
                acc["metric_count"][metric] += int(values.size)
                if values.size:
                    acc["metric_values"][metric].append(values)

    records: list[dict[str, Any]] = []
    for (family, profile), acc in sorted(accumulators.items()):
        metric_arrays = {
            metric: np.concatenate(acc["metric_values"][metric])
            if acc["metric_values"][metric]
            else np.empty(0, dtype=np.float64)
            for metric in COMPRESSION_METRICS
        }
        fixed = {
            column: _one(acc["fixed"][column], name=f"{family}/{profile}.{column}")
            for column in COMPRESSION_FIXED_COLUMNS
        }
        records.append(
            {
                "dataset": fixed["dataset_id"],
                "model_uid": None,
                "run_id": None,
                "extraction_uid": fixed["extraction_uid"],
                "origin_embedding_artifact_uid": fixed["origin_embedding_artifact_uid"],
                "compression_family": family,
                "compression_profile": profile,
                "sample_count": acc["sample_count"],
                "origin_dimension": fixed["origin_dimension"],
                "search_dimension": fixed["search_dimension"],
                "reconstruction_available": fixed["reconstruction_available"],
                "metric_vector_source": fixed["metric_vector_source"],
                "mean_reconstruction_mse": _mean(
                    acc["metric_sum"]["reconstruction_mse"],
                    acc["metric_count"]["reconstruction_mse"],
                ),
                "p95_reconstruction_mse": float(
                    np.quantile(metric_arrays["reconstruction_mse"], 0.95)
                ),
                "mean_angular_error_rad": _mean(
                    acc["metric_sum"]["angular_error_rad"],
                    acc["metric_count"]["angular_error_rad"],
                ),
                "p95_angular_error_rad": float(
                    np.quantile(metric_arrays["angular_error_rad"], 0.95)
                ),
                "mean_cosine_to_origin": _mean(
                    acc["metric_sum"]["cosine_to_origin"],
                    acc["metric_count"]["cosine_to_origin"],
                ),
                "p05_cosine_to_origin": float(
                    np.quantile(metric_arrays["cosine_to_origin"], 0.05)
                ),
                "origin_fallback_count": acc["origin_fallback_count"],
                "storage_bytes_per_embedding": fixed["storage_bytes_per_embedding"],
                "codebook_bytes": fixed["codebook_bytes"],
                "codebook_bytes_source": fixed["codebook_bytes_source"],
                "codec_parameter_bytes": fixed["codec_parameter_bytes"],
                "codec_parameter_bytes_source": fixed[
                    "codec_parameter_bytes_source"
                ],
            }
        )
    return pd.DataFrame.from_records(records), row_count


def _new_retrieval_accumulator() -> dict[str, Any]:
    return {
        "query_count": 0,
        "mated_count": 0,
        "non_mated_count": 0,
        "origin_rank1_correct_count": 0,
        "compressed_rank1_correct_count": 0,
        "origin_top_k_correct_count": 0,
        "compressed_top_k_correct_count": 0,
        "origin_accepted_count": 0,
        "compressed_accepted_count": 0,
        "origin_dir_rank1_count": 0,
        "compressed_dir_rank1_count": 0,
        "origin_false_accept_count": 0,
        "compressed_false_accept_count": 0,
        "agreement_with_origin_count": 0,
        "threshold_crossing_count": 0,
        "accept_to_reject_count": 0,
        "reject_to_accept_count": 0,
        "origin_decision_correct_count": 0,
        "compressed_decision_correct_count": 0,
        "origin_fallback_count": 0,
        "top_k_identity_jaccard_sum": 0.0,
        "top_k_identity_jaccard_count": 0,
        "top1_score_drift_sum": 0.0,
        "top1_score_drift_abs_sum": 0.0,
        "top1_score_drift_count": 0,
        "fixed": defaultdict(set),
    }


def summarize_retrieval(
    source_path: Path | None,
    *,
    chunksize: int,
    source_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, int]:
    requested_usecols = (
        "compression_family",
        "compression_profile",
        "search_mode",
        "threshold_policy",
        *RETRIEVAL_BOOLEAN_COLUMNS,
        *RETRIEVAL_NUMERIC_COLUMNS,
        *RETRIEVAL_FIXED_COLUMNS,
        *THRESHOLD_COLUMNS,
    )
    if (source_path is None) == (source_frame is None):
        raise ValueError("provide exactly one of source_path or source_frame")
    source_columns = set(
        source_frame.columns
        if source_frame is not None
        else pd.read_csv(source_path, nrows=0).columns
    )
    usecols = tuple(
        column for column in requested_usecols if column in source_columns
    )
    required_source_columns = {
        "compression_family",
        "compression_profile",
        "threshold_policy",
        *RETRIEVAL_BOOLEAN_COLUMNS,
        *RETRIEVAL_NUMERIC_COLUMNS,
        *(
            column
            for column in RETRIEVAL_FIXED_COLUMNS
            if column not in OPTIONAL_RETRIEVAL_FIXED_COLUMNS
        ),
        *THRESHOLD_COLUMNS,
    }
    missing_source_columns = sorted(required_source_columns - source_columns)
    if missing_source_columns:
        raise ValueError(
            "retrieval metrics are missing required columns: "
            f"{missing_source_columns}"
        )
    accumulators: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    row_count = 0

    chunks = (
        (source_frame.loc[:, usecols].copy(),)
        if source_frame is not None
        else pd.read_csv(source_path, usecols=usecols, chunksize=chunksize)
    )
    for chunk in chunks:
        row_count += int(len(chunk))
        if "search_mode" not in chunk:
            chunk["search_mode"] = np.where(
                chunk["compression_family"].astype(str).eq("pca"),
                "pca_direct_cosine",
                "pq_reconstruction_cosine",
            )
        if "origin_score_space" not in chunk:
            chunk["origin_score_space"] = "cosine_similarity"
        if "compressed_score_space" not in chunk:
            chunk["compressed_score_space"] = "cosine_similarity"
        if "score_spaces_comparable" not in chunk:
            chunk["score_spaces_comparable"] = True
        if "frozen_origin_threshold_applicable" not in chunk:
            chunk["frozen_origin_threshold_applicable"] = True
        if "latency_measurement_repeats" not in chunk:
            chunk["latency_measurement_repeats"] = 0
        if "latency_timer" not in chunk:
            chunk["latency_timer"] = "not_measured"
        for latency_column in (
            "compressed_index_build_latency_ms",
            "compressed_gallery_add_latency_ms",
            "compressed_gallery_encode_latency_ms",
            "origin_search_latency_ms_total",
            "compressed_search_latency_ms_total",
            "compressed_search_latency_ms_per_query",
            "compressed_search_queries_per_second",
        ):
            if latency_column not in chunk:
                chunk[latency_column] = np.nan
        if "gallery_template_count" not in chunk:
            chunk["gallery_template_count"] = np.nan
        if "codec_parameter_bytes" not in chunk:
            chunk["codec_parameter_bytes"] = chunk["codebook_bytes"]
        if "codec_parameter_bytes_source" not in chunk:
            chunk["codec_parameter_bytes_source"] = chunk[
                "codebook_bytes_source"
            ]
        for key, group in chunk.groupby(
            [
                "compression_family",
                "compression_profile",
                "search_mode",
                "threshold_policy",
            ],
            sort=False,
        ):
            normalized_key = tuple(str(value) for value in key)
            acc = accumulators.setdefault(normalized_key, _new_retrieval_accumulator())
            boolean = {
                column: _as_bool(group[column]) for column in RETRIEVAL_BOOLEAN_COLUMNS
            }
            mated = boolean["is_mated"]
            non_mated = ~mated
            origin_rank1 = boolean["origin_rank1_correct"]
            compressed_rank1 = boolean["compressed_rank1_correct"]
            origin_top_k = boolean["origin_top_k_correct"]
            compressed_top_k = boolean["compressed_top_k_correct"]
            origin_accepted = boolean["origin_accepted"]
            compressed_accepted = boolean["compressed_accepted"]

            acc["query_count"] += int(len(group))
            acc["mated_count"] += int(mated.sum())
            acc["non_mated_count"] += int(non_mated.sum())
            acc["origin_rank1_correct_count"] += int((origin_rank1 & mated).sum())
            acc["compressed_rank1_correct_count"] += int(
                (compressed_rank1 & mated).sum()
            )
            acc["origin_top_k_correct_count"] += int((origin_top_k & mated).sum())
            acc["compressed_top_k_correct_count"] += int(
                (compressed_top_k & mated).sum()
            )
            acc["origin_accepted_count"] += int(origin_accepted.sum())
            acc["compressed_accepted_count"] += int(compressed_accepted.sum())
            acc["origin_dir_rank1_count"] += int(
                (origin_accepted & origin_rank1 & mated).sum()
            )
            acc["compressed_dir_rank1_count"] += int(
                (compressed_accepted & compressed_rank1 & mated).sum()
            )
            acc["origin_false_accept_count"] += int((origin_accepted & non_mated).sum())
            acc["compressed_false_accept_count"] += int(
                (compressed_accepted & non_mated).sum()
            )
            acc["agreement_with_origin_count"] += int(
                boolean["agreement_with_origin"].sum()
            )
            acc["threshold_crossing_count"] += int(boolean["threshold_crossing"].sum())
            accept_to_reject = boolean["origin_accepted"] & ~boolean[
                "compressed_accepted"
            ]
            reject_to_accept = ~boolean["origin_accepted"] & boolean[
                "compressed_accepted"
            ]
            acc["accept_to_reject_count"] += int(accept_to_reject.sum())
            acc["reject_to_accept_count"] += int(reject_to_accept.sum())
            acc["origin_decision_correct_count"] += int(
                boolean["origin_decision_correct"].sum()
            )
            acc["compressed_decision_correct_count"] += int(
                boolean["compressed_decision_correct"].sum()
            )
            acc["origin_fallback_count"] += int(boolean["origin_fallback_used"].sum())

            jaccard = pd.to_numeric(
                group["top_k_identity_jaccard"], errors="coerce"
            ).dropna()
            drift = pd.to_numeric(group["top1_score_drift"], errors="coerce").dropna()
            acc["top_k_identity_jaccard_sum"] += float(jaccard.sum())
            acc["top_k_identity_jaccard_count"] += int(jaccard.size)
            acc["top1_score_drift_sum"] += float(drift.sum())
            acc["top1_score_drift_abs_sum"] += float(drift.abs().sum())
            acc["top1_score_drift_count"] += int(drift.size)
            _add_fixed_values(
                acc["fixed"],
                group,
                (*RETRIEVAL_FIXED_COLUMNS, *THRESHOLD_COLUMNS),
            )

    records: list[dict[str, Any]] = []
    for (family, profile, search_mode, policy), acc in sorted(
        accumulators.items()
    ):
        fixed = {
            column: _one(
                acc["fixed"][column],
                name=f"{family}/{profile}/{policy}.{column}",
            )
            for column in (*RETRIEVAL_FIXED_COLUMNS, *THRESHOLD_COLUMNS)
        }
        gallery_template_count = fixed["gallery_template_count"]
        has_gallery_count = (
            gallery_template_count is not None
            and np.isfinite(float(gallery_template_count))
            and int(gallery_template_count) > 0
        )
        compressed_gallery_storage_bytes = (
            int(gallery_template_count) * int(fixed["storage_bytes_per_embedding"])
            + int(fixed["codec_parameter_bytes"])
            if has_gallery_count
            else np.nan
        )
        amortized_storage_bytes = (
            float(fixed["storage_bytes_per_embedding"])
            + float(fixed["codec_parameter_bytes"]) / int(gallery_template_count)
            if has_gallery_count
            else np.nan
        )
        records.append(
            {
                "dataset": fixed["dataset_id"],
                "model_uid": None,
                "run_id": None,
                "extraction_uid": fixed["extraction_uid"],
                "origin_embedding_artifact_uid": fixed["origin_embedding_artifact_uid"],
                "protocol_uid": fixed["protocol_uid"],
                "compression_family": family,
                "compression_profile": profile,
                "search_mode": search_mode,
                "origin_score_space": fixed["origin_score_space"],
                "compressed_score_space": fixed["compressed_score_space"],
                "score_spaces_comparable": fixed["score_spaces_comparable"],
                "frozen_origin_threshold_applicable": fixed[
                    "frozen_origin_threshold_applicable"
                ],
                "latency_measurement_repeats": fixed[
                    "latency_measurement_repeats"
                ],
                "latency_timer": fixed["latency_timer"],
                "compressed_index_build_latency_ms": fixed[
                    "compressed_index_build_latency_ms"
                ],
                "compressed_gallery_add_latency_ms": fixed[
                    "compressed_gallery_add_latency_ms"
                ],
                "compressed_gallery_encode_latency_ms": fixed[
                    "compressed_gallery_encode_latency_ms"
                ],
                "origin_search_latency_ms_total": fixed[
                    "origin_search_latency_ms_total"
                ],
                "compressed_search_latency_ms_total": fixed[
                    "compressed_search_latency_ms_total"
                ],
                "compressed_search_latency_ms_per_query": fixed[
                    "compressed_search_latency_ms_per_query"
                ],
                "compressed_search_queries_per_second": fixed[
                    "compressed_search_queries_per_second"
                ],
                "gallery_template_count": gallery_template_count,
                "origin_storage_bytes_per_embedding": 512 * 4,
                "origin_gallery_storage_bytes": (
                    int(gallery_template_count) * 512 * 4
                    if has_gallery_count
                    else np.nan
                ),
                "compressed_gallery_storage_bytes": compressed_gallery_storage_bytes,
                "amortized_storage_bytes_per_gallery_template": (
                    amortized_storage_bytes
                ),
                "threshold_policy": policy,
                "threshold_source_split": fixed["threshold_source_split"],
                "evaluation_split": fixed["evaluation_split"],
                "top_k": fixed["top_k"],
                "query_count": acc["query_count"],
                "mated_count": acc["mated_count"],
                "non_mated_count": acc["non_mated_count"],
                "decision_threshold": fixed["decision_threshold"],
                "origin_decision_threshold": fixed["origin_decision_threshold"],
                "compressed_decision_threshold": fixed["compressed_decision_threshold"],
                "origin_rank1_correct_count": acc["origin_rank1_correct_count"],
                "origin_rank1_rate": _rate(
                    acc["origin_rank1_correct_count"], acc["mated_count"]
                ),
                "compressed_rank1_correct_count": acc["compressed_rank1_correct_count"],
                "compressed_rank1_rate": _rate(
                    acc["compressed_rank1_correct_count"], acc["mated_count"]
                ),
                "origin_top_k_correct_count": acc["origin_top_k_correct_count"],
                "origin_top_k_rate": _rate(
                    acc["origin_top_k_correct_count"], acc["mated_count"]
                ),
                "compressed_top_k_correct_count": acc["compressed_top_k_correct_count"],
                "compressed_top_k_rate": _rate(
                    acc["compressed_top_k_correct_count"], acc["mated_count"]
                ),
                "origin_accepted_count": acc["origin_accepted_count"],
                "compressed_accepted_count": acc["compressed_accepted_count"],
                "origin_dir_rank1_count": acc["origin_dir_rank1_count"],
                "origin_dir_rank1": _rate(
                    acc["origin_dir_rank1_count"], acc["mated_count"]
                ),
                "compressed_dir_rank1_count": acc["compressed_dir_rank1_count"],
                "compressed_dir_rank1": _rate(
                    acc["compressed_dir_rank1_count"], acc["mated_count"]
                ),
                "origin_false_accept_count": acc["origin_false_accept_count"],
                "origin_fpir": _rate(
                    acc["origin_false_accept_count"], acc["non_mated_count"]
                ),
                "compressed_false_accept_count": acc["compressed_false_accept_count"],
                "compressed_fpir": _rate(
                    acc["compressed_false_accept_count"], acc["non_mated_count"]
                ),
                "agreement_with_origin_count": acc["agreement_with_origin_count"],
                "agreement_with_origin_rate": _rate(
                    acc["agreement_with_origin_count"], acc["query_count"]
                ),
                "mean_top_k_identity_jaccard": _mean(
                    acc["top_k_identity_jaccard_sum"],
                    acc["top_k_identity_jaccard_count"],
                ),
                "mean_top1_score_drift": _mean(
                    acc["top1_score_drift_sum"], acc["top1_score_drift_count"]
                ),
                "mean_abs_top1_score_drift": _mean(
                    acc["top1_score_drift_abs_sum"],
                    acc["top1_score_drift_count"],
                ),
                "threshold_crossing_count": acc["threshold_crossing_count"],
                "threshold_crossing_rate": _rate(
                    acc["threshold_crossing_count"], acc["query_count"]
                ),
                "accept_to_reject_count": acc["accept_to_reject_count"],
                "accept_to_reject_rate": _rate(
                    acc["accept_to_reject_count"], acc["query_count"]
                ),
                "reject_to_accept_count": acc["reject_to_accept_count"],
                "reject_to_accept_rate": _rate(
                    acc["reject_to_accept_count"], acc["query_count"]
                ),
                "origin_decision_correct_count": acc["origin_decision_correct_count"],
                "origin_decision_correct_rate": _rate(
                    acc["origin_decision_correct_count"], acc["query_count"]
                ),
                "compressed_decision_correct_count": acc[
                    "compressed_decision_correct_count"
                ],
                "compressed_decision_correct_rate": _rate(
                    acc["compressed_decision_correct_count"], acc["query_count"]
                ),
                "origin_fallback_count": acc["origin_fallback_count"],
                "storage_bytes_per_embedding": fixed["storage_bytes_per_embedding"],
                "codebook_bytes": fixed["codebook_bytes"],
                "codebook_bytes_source": fixed["codebook_bytes_source"],
                "codec_parameter_bytes": fixed["codec_parameter_bytes"],
                "codec_parameter_bytes_source": fixed[
                    "codec_parameter_bytes_source"
                ],
            }
        )
    return pd.DataFrame.from_records(records), row_count


def generate(
    run_dir: Path,
    output_dir: Path,
    *,
    chunksize: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    workflow_dir = run_dir / "artifacts" / "step2_workflow"
    run_manifest_path = run_dir / "run_manifest.json"
    freeze_manifest_path = workflow_dir / "freeze_manifest.json"
    step4_summary_path = workflow_dir / "step4_summary.json"
    paired_path = workflow_dir / "paired_embedding_metrics.csv"
    retrieval_path = workflow_dir / "retrieval_metrics.csv"
    required = (
        run_manifest_path,
        freeze_manifest_path,
        step4_summary_path,
        paired_path,
        retrieval_path,
        run_dir / "COMPLETED",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"run is missing required completed artifacts: {missing}"
        )

    run_manifest = _load_json(run_manifest_path)
    freeze_manifest = _load_json(freeze_manifest_path)
    step4_summary = _load_json(step4_summary_path)
    if run_manifest.get("status") != "completed":
        raise ValueError(f"run status is not completed: {run_manifest.get('status')!r}")
    run_id = str(run_manifest["run_id"])
    model_uid = str(freeze_manifest["model_uid"])
    dataset_id = str(freeze_manifest["dataset_id"])
    if freeze_manifest.get("fallback_free") is not True:
        raise ValueError("freeze manifest does not declare fallback_free=true")
    for label, payload in (
        ("freeze_manifest", freeze_manifest),
        ("step4_summary", step4_summary),
    ):
        if str(payload.get("run_id")) != run_id:
            raise ValueError(f"{label} run_id does not match run_manifest")
        if str(payload.get("dataset_id")) != dataset_id:
            raise ValueError(f"{label} dataset_id does not match freeze_manifest")
        if str(payload.get("model_uid")) != model_uid:
            raise ValueError(f"{label} model_uid does not match freeze_manifest")

    compression, paired_rows = summarize_compression(paired_path, chunksize=chunksize)
    retrieval, retrieval_rows = summarize_retrieval(retrieval_path, chunksize=chunksize)
    expected_paired = int(step4_summary["paired_rows"])
    expected_retrieval = int(step4_summary["retrieval_rows"])
    if paired_rows != expected_paired:
        raise ValueError(f"paired row mismatch: {paired_rows} != {expected_paired}")
    if retrieval_rows != expected_retrieval:
        raise ValueError(
            f"retrieval row mismatch: {retrieval_rows} != {expected_retrieval}"
        )

    for frame in (compression, retrieval):
        frame.loc[:, "dataset"] = dataset_id
        frame.loc[:, "model_uid"] = model_uid
        frame.loc[:, "run_id"] = run_id

    compression_path = output_dir / "compression_summary.csv"
    retrieval_summary_path = output_dir / "retrieval_summary.csv"
    manifest_path = output_dir / "summary_manifest.json"
    existing_outputs = [
        path
        for path in (compression_path, retrieval_summary_path, manifest_path)
        if path.exists()
    ]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            f"summary outputs already exist; pass --overwrite: {existing_outputs}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    compression.to_csv(
        compression_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.12g",
    )
    retrieval.to_csv(
        retrieval_summary_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.12g",
    )

    generator_path = Path(__file__).resolve()
    project_root = generator_path.parent.parent
    source_files = {
        "run_manifest.json": run_manifest_path,
        "freeze_manifest.json": freeze_manifest_path,
        "step4_summary.json": step4_summary_path,
        "paired_embedding_metrics.csv": paired_path,
        "retrieval_metrics.csv": retrieval_path,
    }
    output_files = {
        "compression_summary.csv": compression_path,
        "retrieval_summary.csv": retrieval_summary_path,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "step4_compact_summaries",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "model_uid": model_uid,
        "run_id": run_id,
        "source_run_status": run_manifest["status"],
        "source_run_git": run_manifest["git"],
        "source_ledgers_preserved_immutable": True,
        "generator": {
            "path": _portable_path(generator_path, project_root=project_root),
            "sha256": _sha256_file(generator_path),
        },
        "source_files": {
            name: {
                "path": _portable_path(path, project_root=project_root),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for name, path in source_files.items()
        },
        "validated_counts": {
            "paired_rows": paired_rows,
            "retrieval_rows": retrieval_rows,
            "compression_summary_rows": int(len(compression)),
            "retrieval_summary_rows": int(len(retrieval)),
        },
        "summary_contract": {
            "compression_grain": "one row per compression family and profile",
            "retrieval_grain": (
                "one row per compression family, profile, search mode, and "
                "threshold policy"
            ),
            "score_space_policy": (
                "cross-space score drift is undefined for PQ ADC; frozen-origin "
                "threshold is inapplicable outside cosine score space"
            ),
            "rates": "fractions in [0,1] with explicit numerator and denominator columns",
            "origin_and_compressed_operating_points": "reported separately",
        },
        "output_files": {
            name: {
                "path": _portable_path(path, project_root=project_root),
                "bytes": path.stat().st_size,
                "rows": int(len(compression))
                if name == "compression_summary.csv"
                else int(len(retrieval)),
                "sha256": _sha256_file(path),
            }
            for name, path in output_files.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "output_dir": str(output_dir),
        "compression_rows": int(len(compression)),
        "retrieval_rows": int(len(retrieval)),
        "manifest": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate immutable compact summaries from a completed Step 4 run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.chunksize < 1:
        raise ValueError("chunksize must be positive")
    result = generate(
        args.run_dir,
        args.output_dir,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
