"""Atomic, normalized storage for large per-query retrieval evaluations.

The Step 4 retrieval evaluator produces the same search result for several
threshold targets and policies.  A flat CSV repeats the expensive search and
top-k payload for every decision row.  This module stores one condition-level
``core`` table, narrow threshold ``decision`` tables, and (for full-retention
runs only) one optional top-k detail table.

The public reader reconstructs the legacy logical row shape in bounded
condition batches so downstream analysis does not need a special scientific
code path.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from research.runtime.hashing import sha256_file


RETRIEVAL_LEDGER_SCHEMA_VERSION = 1
RETRIEVAL_LEDGER_ARTIFACT_TYPE = "normalized_retrieval_ledger"
RETRIEVAL_LEDGER_MANIFEST_NAME = "manifest.json"

TOPK_DETAIL_COLUMNS = (
    "origin_top_k_gallery_ids",
    "compressed_top_k_gallery_ids",
    "origin_top_k_identity_ids",
    "compressed_top_k_identity_ids",
    "origin_top_k_scores",
    "compressed_top_k_scores",
)

DECISION_COLUMNS = (
    "target_fpir",
    "threshold_policy",
    "decision_threshold",
    "origin_decision_threshold",
    "compressed_decision_threshold",
    "origin_accepted",
    "compressed_accepted",
    "origin_tpir_at_rank_k",
    "compressed_tpir_at_rank_k",
    "threshold_crossing",
    "threshold_crossing_direction",
    "origin_decision_correct",
    "compressed_decision_correct",
)

CONDITION_COLUMNS = (
    "dataset",
    "dataset_id",
    "model_uid",
    "extraction_uid",
    "origin_embedding_artifact_uid",
    "compression_family",
    "compression_profile",
    "search_mode",
    "protocol_uid",
    "threshold_source_split",
    "evaluation_split",
)

_ROW_ID = "core_row_id"


def _json_dump(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_entry(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _stable_frame_digest(frame: pd.DataFrame) -> str:
    """Return a process-independent digest for equality checks within a run."""

    hashes = pd.util.hash_pandas_object(
        frame,
        index=False,
        categorize=False,
    ).to_numpy(dtype=np.uint64, copy=False)
    digest = hashlib.sha256()
    digest.update("\x1f".join(str(column) for column in frame.columns).encode())
    digest.update(str(tuple(str(dtype) for dtype in frame.dtypes)).encode())
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def _single_value(frame: pd.DataFrame, column: str) -> object:
    if column not in frame:
        raise ValueError(f"retrieval ledger input is missing {column!r}")
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(
            f"retrieval ledger batch must contain one {column!r} value; "
            f"observed={len(values)}"
        )
    value = values.iloc[0]
    return value.item() if isinstance(value, np.generic) else value


def _condition_payload(frame: pd.DataFrame) -> dict[str, object]:
    return {
        column: _single_value(frame, column)
        for column in CONDITION_COLUMNS
        if column in frame.columns
    }


def _condition_id(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _decision_id(
    condition_id: str,
    *,
    target_fpir: object,
    threshold_policy: object,
) -> str:
    encoded = (
        f"{condition_id}\x1f{float(target_fpir):.17g}\x1f{threshold_policy}"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _portable_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


class RetrievalLedgerWriter:
    """Write normalized retrieval batches and atomically publish a manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        lineage: Mapping[str, object] | None = None,
        include_topk_detail: bool = False,
        overwrite: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        if self.manifest_path.name != RETRIEVAL_LEDGER_MANIFEST_NAME:
            raise ValueError(
                "retrieval ledger path must end in "
                f"{RETRIEVAL_LEDGER_MANIFEST_NAME!r}"
            )
        self.destination = self.manifest_path.parent
        self.lineage = dict(lineage or {})
        self.include_topk_detail = bool(include_topk_detail)
        self.overwrite = bool(overwrite)
        self._staging = self.destination.parent / (
            f".{self.destination.name}.staging.{uuid.uuid4().hex}"
        )
        self._conditions: dict[str, dict[str, object]] = {}
        self._decision_keys: set[tuple[str, float, str]] = set()
        self._logical_columns: tuple[str, ...] | None = None
        self._logical_row_count = 0
        self._closed = False

    def __enter__(self) -> RetrievalLedgerWriter:
        if self.destination.exists() and not self.overwrite:
            raise FileExistsError(self.destination)
        self._staging.mkdir(parents=True, exist_ok=False)
        (self._staging / "core").mkdir()
        (self._staging / "decisions").mkdir()
        if self.include_topk_detail:
            (self._staging / "topk_detail").mkdir()
        return self

    def _annotate_lineage(self, batch: pd.DataFrame) -> pd.DataFrame:
        frame = batch.copy()
        for column, raw_value in self.lineage.items():
            value = _portable_scalar(raw_value)
            if column in frame.columns:
                observed = frame[column].dropna().astype(str).unique().tolist()
                if observed and observed != [str(value)]:
                    raise ValueError(
                        f"retrieval ledger lineage mismatch for {column}: "
                        f"{observed!r} != {value!r}"
                    )
            frame[column] = value
        return frame

    def write(self, batch: pd.DataFrame) -> None:
        """Append one complete condition/target/policy evaluation batch."""

        if self._closed or not self._staging.is_dir():
            raise RuntimeError("retrieval ledger writer is not active")
        if batch.empty:
            raise ValueError("retrieval ledger batch must not be empty")
        frame = self._annotate_lineage(batch).reset_index(drop=True)
        required = {
            "query_id",
            "compression_family",
            "compression_profile",
            "search_mode",
            "target_fpir",
            "threshold_policy",
            "origin_fallback_used",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"retrieval ledger batch is missing columns: {missing}")
        if frame["origin_fallback_used"].astype(bool).any():
            raise RuntimeError("retrieval ledger contains origin fallback rows")
        if frame["query_id"].astype(str).duplicated().any():
            raise ValueError(
                "retrieval ledger batch query_id values must be unique within "
                "one condition/target/policy"
            )

        logical_columns = tuple(str(column) for column in frame.columns)
        if self._logical_columns is None:
            self._logical_columns = logical_columns
        elif logical_columns != self._logical_columns:
            raise ValueError(
                "retrieval ledger batch schema/order drifted between conditions"
            )

        condition = _condition_payload(frame)
        condition_id = _condition_id(condition)
        target_fpir = float(_single_value(frame, "target_fpir"))
        threshold_policy = str(_single_value(frame, "threshold_policy"))
        decision_key = (condition_id, target_fpir, threshold_policy)
        if decision_key in self._decision_keys:
            raise ValueError(
                "duplicate retrieval decision batch for "
                f"condition={condition_id}, target_fpir={target_fpir}, "
                f"threshold_policy={threshold_policy!r}"
            )

        core_columns = tuple(
            column
            for column in logical_columns
            if column not in DECISION_COLUMNS and column not in TOPK_DETAIL_COLUMNS
        )
        core = frame.loc[:, list(core_columns)].copy()
        core.insert(0, _ROW_ID, np.arange(len(core), dtype=np.int64))
        core_digest = _stable_frame_digest(core)
        condition_entry = self._conditions.get(condition_id)
        if condition_entry is None:
            core_path = self._staging / "core" / f"{condition_id}.parquet"
            core.to_parquet(core_path, index=False, compression="zstd")
            condition_entry = {
                "condition_id": condition_id,
                "condition": {
                    key: _portable_scalar(value)
                    for key, value in condition.items()
                },
                "row_count": int(len(frame)),
                "core_digest": core_digest,
                "core_columns": list(core_columns),
                "core": _artifact_entry(core_path, root=self._staging),
                "decisions": [],
            }
            if self.include_topk_detail:
                detail_columns = tuple(
                    column for column in TOPK_DETAIL_COLUMNS if column in frame
                )
                detail = frame.loc[:, list(detail_columns)].copy()
                detail.insert(0, _ROW_ID, np.arange(len(detail), dtype=np.int64))
                detail_path = (
                    self._staging / "topk_detail" / f"{condition_id}.parquet"
                )
                detail.to_parquet(detail_path, index=False, compression="zstd")
                condition_entry["topk_detail_columns"] = list(detail_columns)
                condition_entry["topk_detail"] = _artifact_entry(
                    detail_path,
                    root=self._staging,
                )
            self._conditions[condition_id] = condition_entry
        else:
            if int(condition_entry["row_count"]) != len(frame):
                raise ValueError(
                    "retrieval core row count differs across threshold decisions"
                )
            if str(condition_entry["core_digest"]) != core_digest:
                raise ValueError(
                    "threshold decisions disagree on threshold-invariant "
                    f"retrieval core for condition {condition_id}"
                )

        decision_columns = tuple(
            column for column in DECISION_COLUMNS if column in frame.columns
        )
        decision = frame.loc[:, list(decision_columns)].copy()
        decision.insert(0, _ROW_ID, np.arange(len(decision), dtype=np.int64))
        decision_id = _decision_id(
            condition_id,
            target_fpir=target_fpir,
            threshold_policy=threshold_policy,
        )
        decision_path = (
            self._staging / "decisions" / f"{decision_id}.parquet"
        )
        decision.to_parquet(decision_path, index=False, compression="zstd")
        decision_entry = {
            "decision_id": decision_id,
            "target_fpir": target_fpir,
            "threshold_policy": threshold_policy,
            "row_count": int(len(decision)),
            "columns": list(decision_columns),
            "artifact": _artifact_entry(decision_path, root=self._staging),
        }
        decisions = condition_entry["decisions"]
        if not isinstance(decisions, list):
            raise RuntimeError("invalid in-memory retrieval ledger state")
        decisions.append(decision_entry)
        self._decision_keys.add(decision_key)
        self._logical_row_count += int(len(frame))

    def finalize(self) -> dict[str, object]:
        """Publish the completed ledger and return its manifest payload."""

        if self._closed:
            raise RuntimeError("retrieval ledger writer is already closed")
        if not self._conditions or self._logical_columns is None:
            raise ValueError("retrieval ledger has no batches")
        conditions = sorted(
            self._conditions.values(),
            key=lambda entry: str(entry["condition_id"]),
        )
        for condition in conditions:
            decisions = condition["decisions"]
            if isinstance(decisions, list):
                decisions.sort(
                    key=lambda entry: (
                        float(entry["target_fpir"]),
                        str(entry["threshold_policy"]),
                    )
                )
        topk_columns = [
            column for column in TOPK_DETAIL_COLUMNS if column in self._logical_columns
        ]
        available_columns = [
            column
            for column in self._logical_columns
            if self.include_topk_detail or column not in TOPK_DETAIL_COLUMNS
        ]
        manifest: dict[str, object] = {
            "schema_version": RETRIEVAL_LEDGER_SCHEMA_VERSION,
            "status": "completed",
            "artifact_type": RETRIEVAL_LEDGER_ARTIFACT_TYPE,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "logical_row_count": int(self._logical_row_count),
            "core_row_count": int(
                sum(int(entry["row_count"]) for entry in conditions)
            ),
            "condition_count": len(conditions),
            "decision_partition_count": len(self._decision_keys),
            "logical_columns": list(self._logical_columns),
            "available_columns": available_columns,
            "omitted_columns": (
                [] if self.include_topk_detail else topk_columns
            ),
            "topk_detail_retained": self.include_topk_detail,
            "row_id_column": _ROW_ID,
            "condition_columns": list(CONDITION_COLUMNS),
            "decision_columns": list(DECISION_COLUMNS),
            "topk_detail_columns": list(TOPK_DETAIL_COLUMNS),
            "lineage": {
                key: _portable_scalar(value)
                for key, value in self.lineage.items()
            },
            "conditions": conditions,
        }
        _json_dump(self._staging / RETRIEVAL_LEDGER_MANIFEST_NAME, manifest)
        if self.destination.exists():
            if not self.overwrite:
                raise FileExistsError(self.destination)
            shutil.rmtree(self.destination)
        os.replace(self._staging, self.destination)
        self._closed = True
        return load_retrieval_ledger_manifest(self.manifest_path)

    def abort(self) -> None:
        if not self._closed and self._staging.exists():
            shutil.rmtree(self._staging)
        self._closed = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if exc_type is not None:
            self.abort()
        elif not self._closed:
            try:
                self.finalize()
            except BaseException:
                self.abort()
                raise
        return False


def load_retrieval_ledger_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RETRIEVAL_LEDGER_SCHEMA_VERSION:
        raise ValueError("unsupported retrieval ledger schema_version")
    if payload.get("status") != "completed":
        raise ValueError("retrieval ledger is not completed")
    if payload.get("artifact_type") != RETRIEVAL_LEDGER_ARTIFACT_TYPE:
        raise ValueError("invalid retrieval ledger artifact_type")
    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("retrieval ledger conditions must be a non-empty list")
    return payload


def is_retrieval_ledger(path: str | Path) -> bool:
    source = Path(path)
    if source.name != RETRIEVAL_LEDGER_MANIFEST_NAME or not source.is_file():
        return False
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("artifact_type") == RETRIEVAL_LEDGER_ARTIFACT_TYPE


def retrieval_source_columns(path: str | Path) -> tuple[str, ...]:
    source = Path(path)
    if is_retrieval_ledger(source):
        payload = load_retrieval_ledger_manifest(source)
        raw = payload.get("available_columns")
        if not isinstance(raw, list):
            raise ValueError("retrieval ledger available_columns is invalid")
        return tuple(str(column) for column in raw)
    if source.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return tuple(pq.ParquetFile(source).schema.names)
    return tuple(pd.read_csv(source, nrows=0).columns.astype(str))


def _artifact_path(
    root: Path,
    entry: Mapping[str, object],
    *,
    verified: set[Path] | None = None,
) -> Path:
    raw = entry.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("retrieval ledger artifact path is invalid")
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("retrieval ledger artifact escapes its root") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_bytes = entry.get("bytes")
    if int(expected_bytes) != path.stat().st_size:
        raise ValueError(f"retrieval ledger artifact byte count mismatch: {path}")
    expected_sha256 = entry.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"retrieval ledger artifact SHA-256 is invalid: {path}")
    if verified is None or path not in verified:
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"retrieval ledger artifact SHA-256 mismatch: {path}")
        if verified is not None:
            verified.add(path)
    return path


def iter_retrieval_source_batches(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    chunksize: int = 100_000,
    dtype: Mapping[str, object] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield CSV, Parquet, or normalized-ledger rows in bounded batches."""

    if isinstance(chunksize, bool) or int(chunksize) != chunksize or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    source = Path(path).resolve()
    available = retrieval_source_columns(source)
    requested = available if columns is None else tuple(dict.fromkeys(columns))
    missing = [column for column in requested if column not in available]
    if missing:
        raise ValueError(f"retrieval source is missing columns: {missing}")

    if not is_retrieval_ledger(source):
        if source.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(source)
            for batch in parquet.iter_batches(
                batch_size=int(chunksize),
                columns=list(requested),
            ):
                yield batch.to_pandas()
            return
        yield from pd.read_csv(
            source,
            usecols=list(requested),
            dtype=(
                None
                if dtype is None
                else {
                    column: value
                    for column, value in dtype.items()
                    if column in requested
                }
            ),
            chunksize=int(chunksize),
            low_memory=False,
        )
        return

    payload = load_retrieval_ledger_manifest(source)
    root = source.parent
    verified_artifacts: set[Path] = set()
    conditions = payload["conditions"]
    if not isinstance(conditions, list):
        raise ValueError("retrieval ledger conditions are invalid")
    decision_set = set(DECISION_COLUMNS)
    detail_set = set(TOPK_DETAIL_COLUMNS)
    requested_core = [
        column
        for column in requested
        if column not in decision_set and column not in detail_set
    ]
    requested_decision = [column for column in requested if column in decision_set]
    requested_detail = [column for column in requested if column in detail_set]

    for raw_condition in conditions:
        if not isinstance(raw_condition, Mapping):
            raise ValueError("retrieval ledger condition entry is invalid")
        core_entry = raw_condition.get("core")
        if not isinstance(core_entry, Mapping):
            raise ValueError("retrieval ledger core entry is invalid")
        core_path = _artifact_path(
            root,
            core_entry,
            verified=verified_artifacts,
        )
        core = pd.read_parquet(
            core_path,
            columns=[_ROW_ID, *requested_core],
        )
        expected_rows = int(raw_condition.get("row_count", -1))
        if len(core) != expected_rows:
            raise ValueError("retrieval ledger core row count mismatch")
        if not np.array_equal(
            core[_ROW_ID].to_numpy(dtype=np.int64),
            np.arange(expected_rows, dtype=np.int64),
        ):
            raise ValueError("retrieval ledger core row ids are not contiguous")

        detail: pd.DataFrame | None = None
        if requested_detail:
            detail_entry = raw_condition.get("topk_detail")
            if not isinstance(detail_entry, Mapping):
                raise ValueError("requested top-k detail was not retained")
            detail_path = _artifact_path(
                root,
                detail_entry,
                verified=verified_artifacts,
            )
            detail = pd.read_parquet(
                detail_path,
                columns=[_ROW_ID, *requested_detail],
            )
            if not detail[_ROW_ID].equals(core[_ROW_ID]):
                raise ValueError("retrieval ledger top-k row ids do not match core")

        decisions = raw_condition.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            raise ValueError("retrieval ledger condition has no decisions")
        for raw_decision in decisions:
            if not isinstance(raw_decision, Mapping):
                raise ValueError("retrieval ledger decision entry is invalid")
            artifact = raw_decision.get("artifact")
            if not isinstance(artifact, Mapping):
                raise ValueError("retrieval ledger decision artifact is invalid")
            decision_path = _artifact_path(
                root,
                artifact,
                verified=verified_artifacts,
            )
            decision = pd.read_parquet(
                decision_path,
                columns=[_ROW_ID, *requested_decision],
            )
            if len(decision) != expected_rows or not decision[_ROW_ID].equals(
                core[_ROW_ID]
            ):
                raise ValueError("retrieval ledger decision row ids do not match core")
            joined = core.copy()
            for column in requested_decision:
                joined[column] = decision[column].to_numpy(copy=False)
            if detail is not None:
                for column in requested_detail:
                    joined[column] = detail[column].to_numpy(copy=False)
            joined = joined.loc[:, list(requested)]
            for start in range(0, len(joined), int(chunksize)):
                yield joined.iloc[start : start + int(chunksize)].reset_index(
                    drop=True
                )


def retrieval_source_row_count(path: str | Path) -> int:
    source = Path(path)
    if is_retrieval_ledger(source):
        return int(load_retrieval_ledger_manifest(source)["logical_row_count"])
    if source.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(source).metadata.num_rows)
    return int(sum(len(chunk) for chunk in pd.read_csv(source, chunksize=250_000)))
