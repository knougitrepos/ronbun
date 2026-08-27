"""Bounded-memory, deterministic Phase 05 retrieval associations."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from research.evaluation.retrieval_ledger import (
    iter_retrieval_source_batches,
    retrieval_source_columns,
)
from research.evaluation.saliency_compression import (
    DEFAULT_PRIMARY_THRESHOLD_EVENT_METRICS,
    DEFAULT_PRIMARY_THRESHOLD_SALIENCY_FEATURES,
    DEFAULT_RETRIEVAL_METRICS,
    DEFAULT_SALIENCY_FEATURES,
    JOIN_KEYS,
    LINEAGE_COLUMNS,
    PROFILE_KEYS,
    RETRIEVAL_DERIVATION_SOURCE_COLUMNS,
    WEIGHTED_RERANK_STRATEGY,
    _require_columns,
    _strict_boolean,
    _strict_false,
    _validate_unique,
    derive_saliency_threshold_metrics,
    saliency_retrieval_associations,
    threshold_instability_associations,
    threshold_policy_event_comparisons,
    threshold_policy_saliency_rho_comparisons,
)
from research.evaluation.saliency_streaming import StreamingJoinResult


PARTITIONED_RETRIEVAL_PROJECTION_SCHEMA_VERSION = 1
PARTITIONED_RETRIEVAL_PROJECTION_ARTIFACT_TYPE = (
    "partitioned_saliency_retrieval_projection"
)
PARTITIONED_ASSOCIATION_ALGORITHM_VERSION = (
    "condition-partitioned-identity-bootstrap-v1"
)

RETRIEVAL_PARTITION_COLUMNS = (
    "dataset_id",
    "model_uid",
    "compression_family",
    "compression_profile",
    "search_mode",
    "protocol_uid",
    "threshold_source_split",
    "evaluation_split",
    "target_fpir",
)

_FEATURE_COLUMNS = (
    *JOIN_KEYS,
    "identity_id",
    "saliency_target_eligible",
    "heatmap_available",
    *DEFAULT_SALIENCY_FEATURES,
)

PartitionProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class PartitionedAssociationResult:
    retrieval_associations: pd.DataFrame
    threshold_instability_associations: pd.DataFrame
    threshold_policy_comparisons: pd.DataFrame
    threshold_policy_saliency_rho_comparisons: pd.DataFrame
    partition_count: int
    worker_count: int
    max_in_flight: int


def _emit(
    progress: PartitionProgressCallback | None,
    message: str,
    **details: object,
) -> None:
    if progress is not None:
        progress(message, details)


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _partition_id(key: Mapping[str, object]) -> str:
    payload = json.dumps(
        key,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _append_csv(path: Path, frame: pd.DataFrame, *, first: bool) -> None:
    frame.to_csv(
        path,
        mode="w" if first else "a",
        header=first,
        index=False,
        lineterminator="\n",
    )


def _merge_saliency(
    source: pd.DataFrame,
    saliency_features: pd.DataFrame,
) -> pd.DataFrame:
    joined = source.merge(
        saliency_features,
        on=list(JOIN_KEYS),
        how="left",
        validate="many_to_one",
        indicator="_saliency_merge",
        suffixes=("", "_saliency"),
    )
    missing = joined["_saliency_merge"] != "both"
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} retrieval rows have no saliency sample row"
        )
    joined = joined.drop(columns="_saliency_merge")
    for column in LINEAGE_COLUMNS:
        saliency_column = f"{column}_saliency"
        if saliency_column not in joined:
            continue
        mismatch = joined[column].astype(str) != joined[saliency_column].astype(str)
        if mismatch.any():
            raise ValueError(
                f"{column} differs between retrieval and saliency artifacts"
            )
        joined = joined.drop(columns=saliency_column)
    return joined


def _parquet_writer(
    writers: dict[str, pq.ParquetWriter],
    path: Path,
    partition_id: str,
    frame: pd.DataFrame,
) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    writer = writers.get(partition_id)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        writers[partition_id] = writer
    elif writer.schema != table.schema:
        table = table.cast(writer.schema)
    writer.write_table(table)


def _validate_saliency_features(frame: pd.DataFrame) -> None:
    _require_columns(
        frame,
        (*_FEATURE_COLUMNS, *LINEAGE_COLUMNS),
        name="saliency_features",
    )
    _validate_unique(frame, JOIN_KEYS, name="saliency_features")


def stream_partition_saliency_retrieval_projection(
    saliency_features: pd.DataFrame,
    retrieval_source_path: str | Path,
    *,
    joined_output_path: str | Path | None,
    projection_manifest_path: str | Path,
    chunksize: int = 100_000,
    expected_rows: int | None = None,
    progress: PartitionProgressCallback | None = None,
) -> StreamingJoinResult:
    """Validate, derive, and shard retrieval rows without repeating saliency."""

    if isinstance(chunksize, bool) or int(chunksize) != chunksize or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    source = Path(retrieval_source_path).resolve()
    manifest_path = Path(projection_manifest_path).resolve()
    projection_root = manifest_path.parent
    joined_path = (
        None if joined_output_path is None else Path(joined_output_path).resolve()
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    if manifest_path.exists() or projection_root.exists():
        raise FileExistsError(manifest_path)
    if joined_path is not None and joined_path.exists():
        raise FileExistsError(joined_path)
    projection_root.mkdir(parents=True, exist_ok=False)
    shard_root = projection_root / "partitions"
    shard_root.mkdir()
    if joined_path is not None:
        joined_path.parent.mkdir(parents=True, exist_ok=True)

    _validate_saliency_features(saliency_features)
    feature_lookup = saliency_features.loc[:, list(_FEATURE_COLUMNS)].copy()
    feature_path = projection_root / "saliency_features.parquet"
    feature_lookup.to_parquet(feature_path, index=False, compression="zstd")

    source_columns = retrieval_source_columns(source)
    normalized_columns = tuple(
        "sample_id"
        if column == "query_id" and "sample_id" not in source_columns
        else column
        for column in source_columns
    )
    required = (
        *JOIN_KEYS,
        *PROFILE_KEYS,
        *LINEAGE_COLUMNS,
        "origin_fallback_used",
        "threshold_policy",
        "is_mated",
        "agreement_with_origin",
        *RETRIEVAL_DERIVATION_SOURCE_COLUMNS,
    )
    missing = [column for column in required if column not in normalized_columns]
    if missing:
        raise ValueError(f"retrieval_sensitivity is missing columns: {missing}")
    optional_groups = tuple(
        column
        for column in (
            "search_mode",
            "protocol_uid",
            "threshold_source_split",
            "evaluation_split",
            "target_fpir",
        )
        if column in normalized_columns
    )
    partition_columns = tuple(
        column for column in RETRIEVAL_PARTITION_COLUMNS if column in normalized_columns
    )
    if not partition_columns:
        raise ValueError("retrieval source has no partition columns")
    unique_keys = (*JOIN_KEYS, *PROFILE_KEYS, *optional_groups, "threshold_policy")
    projection_columns = tuple(
        dict.fromkeys(
            (
                *JOIN_KEYS,
                *PROFILE_KEYS,
                *optional_groups,
                "threshold_policy",
                "is_mated",
                "threshold_metric_derivation_version",
                "origin_fallback_used",
                *DEFAULT_RETRIEVAL_METRICS,
            )
        )
    )

    writers: dict[str, pq.ParquetWriter] = {}
    entries: dict[str, dict[str, object]] = {}
    row_count = 0
    projected_row_count = 0
    chunk_count = 0
    try:
        for raw_chunk in iter_retrieval_source_batches(
            source,
            columns=source_columns,
            chunksize=int(chunksize),
        ):
            chunk = (
                raw_chunk.rename(columns={"query_id": "sample_id"})
                if "query_id" in raw_chunk and "sample_id" not in raw_chunk
                else raw_chunk
            )
            _require_columns(chunk, required, name="retrieval_sensitivity")
            if (
                chunk["threshold_policy"].isna().any()
                or chunk["threshold_policy"].astype(str).str.strip().eq("").any()
            ):
                raise ValueError(
                    "retrieval_sensitivity.threshold_policy must contain "
                    "non-empty values"
                )
            chunk["is_mated"] = _strict_boolean(
                chunk["is_mated"],
                name="retrieval_sensitivity.is_mated",
            )
            chunk = derive_saliency_threshold_metrics(chunk)
            _validate_unique(chunk, unique_keys, name="retrieval_sensitivity")
            _strict_false(
                chunk["origin_fallback_used"],
                name="retrieval_sensitivity.origin_fallback_used",
            )
            joined = _merge_saliency(chunk, saliency_features)
            joined["retrieval_metrics_available"] = True
            chunk_count += 1
            if joined_path is not None:
                _append_csv(joined_path, joined, first=chunk_count == 1)

            has_saliency = joined.loc[
                :, list(DEFAULT_SALIENCY_FEATURES)
            ].notna().any(axis=1)
            has_metric = joined.loc[
                :, list(DEFAULT_RETRIEVAL_METRICS)
            ].notna().any(axis=1)
            projection = joined.loc[
                has_saliency & has_metric,
                list(projection_columns),
            ].copy()
            for column in DEFAULT_RETRIEVAL_METRICS:
                if column in {
                    "agreement_with_origin",
                    "threshold_crossing",
                    "accept_to_reject_crossing",
                    "reject_to_accept_crossing",
                }:
                    projection[column] = _strict_boolean(
                        projection[column],
                        name=f"projection.{column}",
                    )
                else:
                    projection[column] = pd.to_numeric(
                        projection[column],
                        errors="coerce",
                    ).astype(np.float64)

            if not projection.empty:
                grouped = projection.groupby(
                    list(partition_columns),
                    dropna=False,
                    sort=True,
                )
                for raw_key, part in grouped:
                    values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
                    key = {
                        column: _json_value(value)
                        for column, value in zip(partition_columns, values)
                    }
                    partition_id = _partition_id(key)
                    shard_path = shard_root / f"{partition_id}.parquet"
                    _parquet_writer(
                        writers,
                        shard_path,
                        partition_id,
                        part.reset_index(drop=True),
                    )
                    entry = entries.setdefault(
                        partition_id,
                        {
                            "partition_id": partition_id,
                            "key": key,
                            "path": shard_path.relative_to(projection_root).as_posix(),
                            "row_count": 0,
                        },
                    )
                    if entry["key"] != key:
                        raise RuntimeError("retrieval partition hash collision")
                    entry["row_count"] = int(entry["row_count"]) + len(part)

            row_count += len(joined)
            projected_row_count += len(projection)
            details: dict[str, object] = {
                "completed": row_count,
                "chunks": chunk_count,
                "partitions": len(entries),
            }
            if expected_rows is not None:
                details["total"] = int(expected_rows)
            _emit(progress, "retrieval partition join", **details)
    finally:
        for writer in writers.values():
            writer.close()

    if chunk_count == 0:
        raise ValueError("retrieval source contains no rows")
    if expected_rows is not None and row_count != int(expected_rows):
        raise ValueError(
            f"retrieval row count differs from phase 04: {row_count} != {expected_rows}"
        )
    if not entries:
        raise ValueError("retrieval projection contains no eligible rows")
    ordered_entries = sorted(
        entries.values(),
        key=lambda entry: tuple(
            str(entry["key"].get(column, "")) for column in partition_columns
        ),
    )
    manifest = {
        "schema_version": PARTITIONED_RETRIEVAL_PROJECTION_SCHEMA_VERSION,
        "status": "completed",
        "artifact_type": PARTITIONED_RETRIEVAL_PROJECTION_ARTIFACT_TYPE,
        "source_path": str(source),
        "source_row_count": int(row_count),
        "projected_row_count": int(projected_row_count),
        "chunk_count": int(chunk_count),
        "partition_count": len(ordered_entries),
        "partition_columns": list(partition_columns),
        "projection_columns": list(projection_columns),
        "feature_columns": list(_FEATURE_COLUMNS),
        "features_path": feature_path.relative_to(projection_root).as_posix(),
        "feature_row_count": int(len(feature_lookup)),
        "partitions": ordered_entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return StreamingJoinResult(
        joined_path=joined_path,
        association_projection_path=manifest_path,
        row_count=int(row_count),
        projected_row_count=int(projected_row_count),
        chunk_count=int(chunk_count),
    )


def load_partitioned_projection_manifest(
    path: str | Path,
) -> dict[str, object]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        PARTITIONED_RETRIEVAL_PROJECTION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported partitioned retrieval projection schema")
    if payload.get("status") != "completed":
        raise ValueError("partitioned retrieval projection is not completed")
    if payload.get("artifact_type") != (
        PARTITIONED_RETRIEVAL_PROJECTION_ARTIFACT_TYPE
    ):
        raise ValueError("invalid partitioned retrieval projection artifact_type")
    return payload


def _projection_path(root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("partitioned retrieval projection path is invalid")
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("partitioned projection path escapes its root") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def iter_partitioned_projection_batches(
    manifest_path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Rehydrate a partitioned projection one shard at a time."""

    if isinstance(chunksize, bool) or int(chunksize) != chunksize or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    source = Path(manifest_path).resolve()
    payload = load_partitioned_projection_manifest(source)
    root = source.parent
    projection_columns = tuple(str(c) for c in payload["projection_columns"])
    feature_columns = tuple(str(c) for c in payload["feature_columns"])
    available = tuple(dict.fromkeys((*projection_columns, *feature_columns)))
    requested = available if columns is None else tuple(dict.fromkeys(columns))
    missing = [column for column in requested if column not in available]
    if missing:
        raise ValueError(f"partitioned projection is missing columns: {missing}")
    shard_columns = [column for column in requested if column in projection_columns]
    requested_features = [
        column
        for column in requested
        if column in feature_columns and column not in JOIN_KEYS
    ]
    features_path = _projection_path(root, payload["features_path"])
    feature_read_columns = list(dict.fromkeys((*JOIN_KEYS, *requested_features)))
    features = pd.read_parquet(features_path, columns=feature_read_columns)
    _validate_unique(features, JOIN_KEYS, name="partitioned_saliency_features")
    partitions = payload.get("partitions")
    if not isinstance(partitions, list):
        raise ValueError("partitioned projection partitions are invalid")
    for entry in partitions:
        if not isinstance(entry, Mapping):
            raise ValueError("partitioned projection entry is invalid")
        shard_path = _projection_path(root, entry.get("path"))
        shard_read_columns = list(dict.fromkeys((*JOIN_KEYS, *shard_columns)))
        shard = pd.read_parquet(shard_path, columns=shard_read_columns)
        joined = shard.merge(
            features,
            on=list(JOIN_KEYS),
            how="left",
            validate="many_to_one",
        )
        joined = joined.loc[:, list(requested)]
        for start in range(0, len(joined), int(chunksize)):
            yield joined.iloc[start : start + int(chunksize)].reset_index(drop=True)


_WORKER_FEATURES: pd.DataFrame | None = None
_WORKER_THREAD_LIMIT: Any = None


def _initialize_partition_worker(features_path: str) -> None:
    global _WORKER_FEATURES, _WORKER_THREAD_LIMIT
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits

        _WORKER_THREAD_LIMIT = threadpool_limits(limits=1)
    except ImportError:
        _WORKER_THREAD_LIMIT = None
    _WORKER_FEATURES = pd.read_parquet(features_path)
    _validate_unique(_WORKER_FEATURES, JOIN_KEYS, name="worker_saliency_features")


def _association_task(task: Mapping[str, object]) -> dict[str, pd.DataFrame]:
    if _WORKER_FEATURES is None:
        raise RuntimeError("partition worker was not initialized")
    shard = pd.read_parquet(str(task["shard_path"]))
    unique_keys = (
        *JOIN_KEYS,
        *PROFILE_KEYS,
        *tuple(task["optional_groups"]),
        "threshold_policy",
    )
    _validate_unique(shard, unique_keys, name="partitioned_retrieval_projection")
    joined = shard.merge(
        _WORKER_FEATURES,
        on=list(JOIN_KEYS),
        how="left",
        validate="many_to_one",
    )
    if joined["identity_id"].isna().any():
        raise ValueError("partitioned retrieval shard has missing saliency features")
    common = {
        "bootstrap_repeats": int(task["bootstrap_repeats"]),
        "seed": int(task["seed"]),
        "bootstrap_rank_strategy": str(task["bootstrap_rank_strategy"]),
        "bootstrap_batch_size": int(task["bootstrap_batch_size"]),
    }
    retrieval = saliency_retrieval_associations(joined, **common)
    instability = threshold_instability_associations(joined, **common)
    policy = threshold_policy_event_comparisons(
        joined,
        event_metrics=tuple(task["paired_event_metrics"]),
        confidence_level=float(task["paired_confidence_level"]),
        bootstrap_repeats=int(task["bootstrap_repeats"]),
        seed=int(task["seed"]),
    )
    rho = threshold_policy_saliency_rho_comparisons(
        joined,
        saliency_features=tuple(task["paired_saliency_features"]),
        event_metrics=tuple(task["paired_event_metrics"]),
        minimum_event_count=int(task["paired_minimum_event_count"]),
        confidence_level=float(task["paired_confidence_level"]),
        bootstrap_repeats=int(task["bootstrap_repeats"]),
        seed=int(task["seed"]),
        bootstrap_batch_size=int(task["bootstrap_batch_size"]),
    )
    return {
        "retrieval": retrieval,
        "instability": instability,
        "policy": policy,
        "rho": rho,
    }


def _sort_result(frame: pd.DataFrame, preferred: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    columns = [column for column in preferred if column in frame.columns]
    return frame.sort_values(columns, kind="stable").reset_index(drop=True)


def compute_partitioned_retrieval_associations(
    projection_manifest_path: str | Path,
    *,
    bootstrap_repeats: int,
    seed: int,
    bootstrap_rank_strategy: str = WEIGHTED_RERANK_STRATEGY,
    bootstrap_batch_size: int = 4,
    paired_saliency_features: Sequence[str] = (
        DEFAULT_PRIMARY_THRESHOLD_SALIENCY_FEATURES
    ),
    paired_event_metrics: Sequence[str] = (
        DEFAULT_PRIMARY_THRESHOLD_EVENT_METRICS
    ),
    paired_minimum_event_count: int = 5,
    paired_confidence_level: float = 0.95,
    max_workers: int = 4,
    max_in_flight: int | None = None,
    progress: PartitionProgressCallback | None = None,
) -> PartitionedAssociationResult:
    """Compute exact per-group outputs with bounded process-level parallelism."""

    if isinstance(max_workers, bool) or int(max_workers) != max_workers:
        raise ValueError("max_workers must be a positive integer")
    workers = int(max_workers)
    if workers <= 0:
        raise ValueError("max_workers must be a positive integer")
    in_flight = workers * 2 if max_in_flight is None else int(max_in_flight)
    if isinstance(max_in_flight, bool) or in_flight <= 0:
        raise ValueError("max_in_flight must be a positive integer")

    manifest_path = Path(projection_manifest_path).resolve()
    payload = load_partitioned_projection_manifest(manifest_path)
    root = manifest_path.parent
    features_path = _projection_path(root, payload["features_path"])
    partitions = payload.get("partitions")
    if not isinstance(partitions, list) or not partitions:
        raise ValueError("partitioned projection has no partitions")
    optional_groups = tuple(
        column
        for column in (
            "search_mode",
            "protocol_uid",
            "threshold_source_split",
            "evaluation_split",
            "target_fpir",
        )
        if column in payload["projection_columns"]
    )

    tasks: list[dict[str, object]] = []
    for entry in partitions:
        if not isinstance(entry, Mapping):
            raise ValueError("partitioned projection entry is invalid")
        tasks.append(
            {
                "partition_id": str(entry["partition_id"]),
                "shard_path": str(_projection_path(root, entry["path"])),
                "optional_groups": optional_groups,
                "bootstrap_repeats": int(bootstrap_repeats),
                "seed": int(seed),
                "bootstrap_rank_strategy": str(bootstrap_rank_strategy),
                "bootstrap_batch_size": int(bootstrap_batch_size),
                "paired_saliency_features": tuple(paired_saliency_features),
                "paired_event_metrics": tuple(paired_event_metrics),
                "paired_minimum_event_count": int(paired_minimum_event_count),
                "paired_confidence_level": float(paired_confidence_level),
            }
        )

    outputs: dict[str, list[pd.DataFrame]] = {
        "retrieval": [],
        "instability": [],
        "policy": [],
        "rho": [],
    }

    def collect(result: Mapping[str, pd.DataFrame]) -> None:
        for name in outputs:
            frame = result[name]
            if not frame.empty:
                outputs[name].append(frame)

    completed = 0
    if workers == 1:
        global _WORKER_FEATURES
        previous_features = _WORKER_FEATURES
        _WORKER_FEATURES = pd.read_parquet(features_path)
        _validate_unique(
            _WORKER_FEATURES,
            JOIN_KEYS,
            name="worker_saliency_features",
        )
        try:
            try:
                from threadpoolctl import threadpool_limits

                thread_limit = threadpool_limits(limits=1)
            except ImportError:
                thread_limit = nullcontext()
            with thread_limit:
                for task in tasks:
                    collect(_association_task(task))
                    completed += 1
                    _emit(
                        progress,
                        "retrieval association partitions",
                        completed=completed,
                        total=len(tasks),
                        workers=1,
                    )
        finally:
            _WORKER_FEATURES = previous_features
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_partition_worker,
            initargs=(str(features_path),),
        ) as executor:
            task_iter = iter(tasks)
            pending = {}
            for _ in range(min(in_flight, len(tasks))):
                task = next(task_iter)
                pending[executor.submit(_association_task, task)] = task
            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    task = pending.pop(future)
                    try:
                        collect(future.result())
                    except BaseException as error:
                        raise RuntimeError(
                            "partitioned retrieval association failed for "
                            f"{task['partition_id']}"
                        ) from error
                    completed += 1
                    _emit(
                        progress,
                        "retrieval association partitions",
                        completed=completed,
                        total=len(tasks),
                        workers=workers,
                    )
                    try:
                        next_task = next(task_iter)
                    except StopIteration:
                        continue
                    pending[executor.submit(_association_task, next_task)] = (
                        next_task
                    )

    combined = {
        name: (
            pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        )
        for name, parts in outputs.items()
    }
    group_order = (
        "dataset_id",
        "model_uid",
        "compression_family",
        "compression_profile",
        "search_mode",
        "protocol_uid",
        "threshold_source_split",
        "evaluation_split",
        "target_fpir",
        "threshold_policy",
        "is_mated",
    )
    retrieval = _sort_result(
        combined["retrieval"],
        (*group_order, "saliency_feature", "sensitivity_metric"),
    )
    instability = _sort_result(
        combined["instability"],
        (*group_order, "instability_predictor", "sensitivity_metric"),
    )
    policy = _sort_result(
        combined["policy"],
        (*group_order, "event_metric"),
    )
    rho = _sort_result(
        combined["rho"],
        (*group_order, "saliency_feature", "event_metric"),
    )
    return PartitionedAssociationResult(
        retrieval_associations=retrieval,
        threshold_instability_associations=instability,
        threshold_policy_comparisons=policy,
        threshold_policy_saliency_rho_comparisons=rho,
        partition_count=len(tasks),
        worker_count=workers,
        max_in_flight=in_flight,
    )
