from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from research.evaluation.saliency_compression import (
    DEFAULT_GEOMETRY_METRICS,
    DEFAULT_RETRIEVAL_METRICS,
    DEFAULT_SALIENCY_FEATURES,
    JOIN_KEYS,
    LINEAGE_COLUMNS,
    PROFILE_KEYS,
    _require_columns,
    _strict_boolean,
    _strict_false,
    _validate_unique,
)


StreamingProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class StreamingJoinResult:
    joined_path: Path
    association_projection_path: Path
    row_count: int
    projected_row_count: int
    chunk_count: int


def _emit(
    progress: StreamingProgressCallback | None,
    message: str,
    **details: object,
) -> None:
    if progress is not None:
        progress(message, details)


def _source_columns(path: Path) -> tuple[str, ...]:
    return tuple(pd.read_csv(path, nrows=0).columns.astype(str))


def _csv_dtype_overrides(columns: Sequence[str]) -> dict[str, str]:
    string_columns = {
        *JOIN_KEYS,
        *PROFILE_KEYS,
        *LINEAGE_COLUMNS,
        "query_id",
        "protocol_uid",
        "threshold_source_split",
        "evaluation_split",
        "threshold_policy",
    }
    return {
        column: "string"
        for column in columns
        if column in string_columns
    }


def _normalize_retrieval_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    if "query_id" in chunk and "sample_id" not in chunk:
        return chunk.rename(columns={"query_id": "sample_id"})
    return chunk


def _key_hashes(frame: pd.DataFrame, keys: Sequence[str]) -> np.ndarray:
    return pd.util.hash_pandas_object(
        frame.loc[:, list(keys)],
        index=False,
        categorize=False,
    ).to_numpy(dtype=np.uint64, copy=False)


def _duplicate_hash_values(hashes: Sequence[np.ndarray]) -> np.ndarray:
    if not hashes:
        return np.empty(0, dtype=np.uint64)
    combined = np.concatenate(hashes)
    if combined.size < 2:
        return np.empty(0, dtype=np.uint64)
    combined.sort()
    repeated = combined[1:][combined[1:] == combined[:-1]]
    return np.unique(repeated)


def _verify_cross_chunk_uniqueness(
    path: Path,
    *,
    keys: Sequence[str],
    name: str,
    duplicate_hashes: np.ndarray,
    chunksize: int,
    normalize: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> None:
    if duplicate_hashes.size == 0:
        return
    source_columns = _source_columns(path)
    source_keys = [
        "query_id" if key == "sample_id" and "sample_id" not in source_columns else key
        for key in keys
    ]
    candidates: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=source_keys,
        dtype=_csv_dtype_overrides(source_keys),
        chunksize=chunksize,
        low_memory=False,
    ):
        normalized = normalize(chunk) if normalize is not None else chunk
        hashes = _key_hashes(normalized, keys)
        selected = np.isin(hashes, duplicate_hashes)
        if selected.any():
            candidates.append(normalized.loc[selected, list(keys)].copy())
    if not candidates:
        raise RuntimeError("duplicate key hashes could not be reloaded")
    candidate_frame = pd.concat(candidates, ignore_index=True)
    _validate_unique(candidate_frame, keys, name=name)


def _validate_saliency_features(saliency_features: pd.DataFrame) -> None:
    _require_columns(
        saliency_features,
        [
            *JOIN_KEYS,
            *LINEAGE_COLUMNS,
            "identity_id",
            "saliency_spec_uid",
            "saliency_target_eligible",
            "heatmap_available",
            *DEFAULT_SALIENCY_FEATURES,
        ],
        name="saliency_features",
    )
    _validate_unique(saliency_features, JOIN_KEYS, name="saliency_features")


def _merge_saliency(
    source: pd.DataFrame,
    saliency_features: pd.DataFrame,
    *,
    source_name: str,
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
            f"{int(missing.sum())} {source_name} rows have no saliency sample row"
        )
    joined = joined.drop(columns="_saliency_merge")
    for column in LINEAGE_COLUMNS:
        saliency_column = f"{column}_saliency"
        mismatch = (
            joined[column].astype(str)
            != joined[saliency_column].astype(str)
        )
        if mismatch.any():
            raise ValueError(
                f"{column} differs between {source_name} and saliency artifacts"
            )
        joined = joined.drop(columns=saliency_column)
    return joined


def _coerce_projection(
    joined: pd.DataFrame,
    *,
    columns: Sequence[str],
    saliency_features: Sequence[str],
    sensitivity_metrics: Sequence[str],
) -> pd.DataFrame:
    projection = joined.loc[:, list(columns)].copy()
    numeric_columns = tuple(
        dict.fromkeys((*saliency_features, *sensitivity_metrics))
    )
    for column in numeric_columns:
        projection[column] = pd.to_numeric(
            projection[column],
            errors="coerce",
        ).astype(np.float64)
    has_saliency = projection.loc[:, list(saliency_features)].notna().any(axis=1)
    has_metric = projection.loc[:, list(sensitivity_metrics)].notna().any(axis=1)
    return projection.loc[has_saliency & has_metric].reset_index(drop=True)


def _write_parquet_chunk(
    writer: pq.ParquetWriter | None,
    path: Path,
    frame: pd.DataFrame,
) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(
            path,
            table.schema,
            compression="zstd",
        )
    elif table.schema != writer.schema:
        table = table.cast(writer.schema)
    writer.write_table(table)
    return writer


def _append_csv_chunk(
    path: Path,
    frame: pd.DataFrame,
    *,
    first: bool,
) -> None:
    frame.to_csv(
        path,
        mode="w" if first else "a",
        header=first,
        index=False,
        lineterminator="\n",
    )


def stream_join_population_saliency_with_compression(
    saliency_features: pd.DataFrame,
    embedding_distortion_path: str | Path,
    *,
    joined_output_path: str | Path,
    association_projection_path: str | Path,
    chunksize: int = 100_000,
    expected_rows: int | None = None,
    progress: StreamingProgressCallback | None = None,
) -> StreamingJoinResult:
    """Strictly join a large geometry CSV without materializing the full table."""

    if isinstance(chunksize, bool) or int(chunksize) != chunksize or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    source_path = Path(embedding_distortion_path)
    joined_path = Path(joined_output_path)
    projection_path = Path(association_projection_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    for output in (joined_path, projection_path):
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
    _validate_saliency_features(saliency_features)

    source_columns = _source_columns(source_path)
    required = [
        *JOIN_KEYS,
        *PROFILE_KEYS,
        *LINEAGE_COLUMNS,
        "origin_fallback_used",
        *DEFAULT_GEOMETRY_METRICS,
    ]
    missing = [column for column in required if column not in source_columns]
    if missing:
        raise ValueError(f"embedding_distortion is missing columns: {missing}")
    unique_keys = (*JOIN_KEYS, *PROFILE_KEYS)
    projection_columns = tuple(
        dict.fromkeys(
            (
                *JOIN_KEYS,
                *PROFILE_KEYS,
                "identity_id",
                *DEFAULT_SALIENCY_FEATURES,
                *DEFAULT_GEOMETRY_METRICS,
            )
        )
    )

    row_count = 0
    projected_row_count = 0
    chunk_count = 0
    key_hashes: list[np.ndarray] = []
    writer: pq.ParquetWriter | None = None
    try:
        for chunk in pd.read_csv(
            source_path,
            dtype=_csv_dtype_overrides(source_columns),
            chunksize=int(chunksize),
            low_memory=False,
        ):
            _require_columns(chunk, required, name="embedding_distortion")
            _validate_unique(
                chunk,
                unique_keys,
                name="embedding_distortion",
            )
            _strict_false(
                chunk["origin_fallback_used"],
                name="embedding_distortion.origin_fallback_used",
            )
            key_hashes.append(_key_hashes(chunk, unique_keys).copy())
            joined = _merge_saliency(
                chunk,
                saliency_features,
                source_name="compression",
            )
            chunk_count += 1
            _append_csv_chunk(
                joined_path,
                joined,
                first=chunk_count == 1,
            )
            projection = _coerce_projection(
                joined,
                columns=projection_columns,
                saliency_features=DEFAULT_SALIENCY_FEATURES,
                sensitivity_metrics=DEFAULT_GEOMETRY_METRICS,
            )
            if not projection.empty:
                writer = _write_parquet_chunk(writer, projection_path, projection)
            row_count += len(joined)
            projected_row_count += len(projection)
            details: dict[str, object] = {
                "completed": row_count,
                "chunks": chunk_count,
            }
            if expected_rows is not None:
                details["total"] = int(expected_rows)
            _emit(progress, "geometry join", **details)
    finally:
        if writer is not None:
            writer.close()
    if chunk_count == 0:
        raise ValueError("embedding_distortion CSV contains no rows")
    if writer is None:
        empty = pd.DataFrame(columns=projection_columns)
        _write_parquet_chunk(None, projection_path, empty).close()
    if expected_rows is not None and row_count != int(expected_rows):
        raise ValueError(
            f"geometry row count differs from phase 04: {row_count} != {expected_rows}"
        )
    duplicate_hashes = _duplicate_hash_values(key_hashes)
    _verify_cross_chunk_uniqueness(
        source_path,
        keys=unique_keys,
        name="embedding_distortion",
        duplicate_hashes=duplicate_hashes,
        chunksize=int(chunksize),
    )
    return StreamingJoinResult(
        joined_path=joined_path,
        association_projection_path=projection_path,
        row_count=int(row_count),
        projected_row_count=int(projected_row_count),
        chunk_count=int(chunk_count),
    )


def stream_join_population_saliency_with_retrieval(
    saliency_features: pd.DataFrame,
    retrieval_sensitivity_path: str | Path,
    *,
    joined_output_path: str | Path,
    association_projection_path: str | Path,
    chunksize: int = 100_000,
    expected_rows: int | None = None,
    progress: StreamingProgressCallback | None = None,
) -> StreamingJoinResult:
    """Strictly join a large retrieval CSV without materializing the full table."""

    if isinstance(chunksize, bool) or int(chunksize) != chunksize or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    source_path = Path(retrieval_sensitivity_path)
    joined_path = Path(joined_output_path)
    projection_path = Path(association_projection_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    for output in (joined_path, projection_path):
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
    _validate_saliency_features(saliency_features)

    source_columns = _source_columns(source_path)
    normalized_columns = tuple(
        "sample_id" if column == "query_id" and "sample_id" not in source_columns else column
        for column in source_columns
    )
    required = [
        *JOIN_KEYS,
        *PROFILE_KEYS,
        *LINEAGE_COLUMNS,
        "origin_fallback_used",
        "threshold_policy",
        "is_mated",
        *DEFAULT_RETRIEVAL_METRICS,
    ]
    missing = [column for column in required if column not in normalized_columns]
    if missing:
        raise ValueError(f"retrieval_sensitivity is missing columns: {missing}")
    optional_groups = tuple(
        column
        for column in (
            "protocol_uid",
            "threshold_source_split",
            "evaluation_split",
        )
        if column in normalized_columns
    )
    unique_keys = (
        *JOIN_KEYS,
        *PROFILE_KEYS,
        *optional_groups,
        "threshold_policy",
    )
    projection_columns = tuple(
        dict.fromkeys(
            (
                *JOIN_KEYS,
                *PROFILE_KEYS,
                *optional_groups,
                "threshold_policy",
                "is_mated",
                "identity_id",
                *DEFAULT_SALIENCY_FEATURES,
                *DEFAULT_RETRIEVAL_METRICS,
            )
        )
    )

    row_count = 0
    projected_row_count = 0
    chunk_count = 0
    key_hashes: list[np.ndarray] = []
    writer: pq.ParquetWriter | None = None
    try:
        for raw_chunk in pd.read_csv(
            source_path,
            dtype=_csv_dtype_overrides(source_columns),
            chunksize=int(chunksize),
            low_memory=False,
        ):
            chunk = _normalize_retrieval_chunk(raw_chunk)
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
            _validate_unique(
                chunk,
                unique_keys,
                name="retrieval_sensitivity",
            )
            _strict_false(
                chunk["origin_fallback_used"],
                name="retrieval_sensitivity.origin_fallback_used",
            )
            key_hashes.append(_key_hashes(chunk, unique_keys).copy())
            joined = _merge_saliency(
                chunk,
                saliency_features,
                source_name="retrieval",
            )
            joined["retrieval_metrics_available"] = True
            chunk_count += 1
            _append_csv_chunk(
                joined_path,
                joined,
                first=chunk_count == 1,
            )
            projection = _coerce_projection(
                joined,
                columns=projection_columns,
                saliency_features=DEFAULT_SALIENCY_FEATURES,
                sensitivity_metrics=DEFAULT_RETRIEVAL_METRICS,
            )
            if not projection.empty:
                writer = _write_parquet_chunk(writer, projection_path, projection)
            row_count += len(joined)
            projected_row_count += len(projection)
            details: dict[str, object] = {
                "completed": row_count,
                "chunks": chunk_count,
            }
            if expected_rows is not None:
                details["total"] = int(expected_rows)
            _emit(progress, "retrieval join", **details)
    finally:
        if writer is not None:
            writer.close()
    if chunk_count == 0:
        raise ValueError("retrieval_sensitivity CSV contains no rows")
    if writer is None:
        empty = pd.DataFrame(columns=projection_columns)
        _write_parquet_chunk(None, projection_path, empty).close()
    if expected_rows is not None and row_count != int(expected_rows):
        raise ValueError(
            f"retrieval row count differs from phase 04: {row_count} != {expected_rows}"
        )
    duplicate_hashes = _duplicate_hash_values(key_hashes)
    _verify_cross_chunk_uniqueness(
        source_path,
        keys=unique_keys,
        name="retrieval_sensitivity",
        duplicate_hashes=duplicate_hashes,
        chunksize=int(chunksize),
        normalize=_normalize_retrieval_chunk,
    )
    return StreamingJoinResult(
        joined_path=joined_path,
        association_projection_path=projection_path,
        row_count=int(row_count),
        projected_row_count=int(projected_row_count),
        chunk_count=int(chunk_count),
    )
