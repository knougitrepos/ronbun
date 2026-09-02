"""Lineage-preserving CR-FIQA score extraction for aligned face bundles."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from research.fiqa.cr_fiqa import infer_cr_fiqa_scores
from research.runtime.hashing import canonical_sha256, sha256_file


@dataclass(frozen=True)
class FIQAScoreArtifact:
    root: Path
    scores: pd.DataFrame
    manifest: dict[str, Any]

    @property
    def fiqa_uid(self) -> str:
        return str(self.manifest["fiqa_uid"])


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _verify_aligned_bundle(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"aligned bundle manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_type") != "aligned_face_crops":
        raise ValueError("aligned bundle has an unexpected artifact_type")
    contract = dict(manifest.get("array_contract", {}))
    if contract.get("dtype") != "uint8" or contract.get("layout") != "nhwc":
        raise ValueError("FIQA requires the aligned uint8 NHWC bundle")
    if contract.get("color_order") != "rgb" or contract.get("image_size") != [112, 112]:
        raise ValueError("FIQA requires aligned 112x112 RGB faces")
    outputs = dict(manifest.get("outputs", {}))
    faces_entry = dict(outputs.get("aligned_faces", {}))
    index_entry = dict(outputs.get("aligned_index", {}))
    faces_path = root / str(faces_entry.get("path", ""))
    index_path = root / str(index_entry.get("path", ""))
    for path, entry in ((faces_path, faces_entry), (index_path, index_entry)):
        if not path.is_file():
            raise FileNotFoundError(f"aligned bundle member is missing: {path}")
        if sha256_file(path) != str(entry.get("sha256")):
            raise ValueError(f"aligned bundle member hash mismatch: {path}")
    return faces_path, index_path, manifest


def infer_aligned_bundle_scores(
    aligned_bundle_dir: str | Path,
    *,
    model: Any,
    model_uid: str,
    checkpoint_sha256: str,
    variant: str,
    batch_size: int = 64,
    device: str = "cuda",
    maximum_samples: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score an aligned bundle in stable index order without loading it in RAM."""

    root = Path(aligned_bundle_dir).resolve()
    faces_path, index_path, bundle_manifest = _verify_aligned_bundle(root)
    limit = None if maximum_samples is None else int(maximum_samples)
    if limit is not None and limit <= 0:
        raise ValueError("maximum_samples must be positive or None")
    index = pd.read_csv(index_path, nrows=limit)
    required = {"sample_id", "aligned_face_index", "aligned_content_sha256"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"aligned index is missing columns: {missing}")
    if index["sample_id"].astype(str).duplicated().any():
        raise ValueError("aligned index sample_id values must be unique")
    face_indices = index["aligned_face_index"].to_numpy(dtype=np.int64)
    if np.any(face_indices < 0):
        raise ValueError("aligned_face_index values must be non-negative")
    faces = np.load(faces_path, mmap_mode="r", allow_pickle=False)
    if faces.shape[1:] != (112, 112, 3) or faces.dtype != np.uint8:
        raise ValueError("aligned face array violates the FIQA input contract")
    if len(face_indices) and int(face_indices.max()) >= len(faces):
        raise ValueError("aligned_face_index points outside aligned_faces.npy")

    score_parts: list[np.ndarray] = []
    batch_value = int(batch_size)
    for start in range(0, len(index), batch_value):
        batch_indices = face_indices[start : start + batch_value]
        batch_faces = np.asarray(faces[batch_indices])
        score_parts.append(
            infer_cr_fiqa_scores(
                model,
                batch_faces,
                batch_size=batch_value,
                device=device,
            )
        )
    scores = (
        np.concatenate(score_parts).astype(np.float32, copy=False)
        if score_parts
        else np.empty(0, dtype=np.float32)
    )
    result = index[
        ["sample_id", "aligned_face_index", "aligned_content_sha256"]
    ].copy()
    result["fiqa_score"] = scores
    result["fiqa_model_uid"] = str(model_uid)
    if result.empty or not np.isfinite(result["fiqa_score"]).all():
        raise RuntimeError("FIQA bundle inference produced no finite score rows")

    manifest_base: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "fiqa_score_table",
        "status": "completed",
        "dataset_id": str(bundle_manifest["dataset_id"]),
        "model_family": "cr-fiqa",
        "variant": str(variant).upper(),
        "fiqa_model_uid": str(model_uid),
        "checkpoint_sha256": str(checkpoint_sha256),
        "aligned_bundle_manifest_sha256": sha256_file(root / "bundle_manifest.json"),
        "aligned_faces_sha256": str(
            bundle_manifest["outputs"]["aligned_faces"]["sha256"]
        ),
        "aligned_index_sha256": str(
            bundle_manifest["outputs"]["aligned_index"]["sha256"]
        ),
        "preprocessing": {
            "input": "aligned_uint8_nhwc_rgb_112x112",
            "tensor": "float32_nchw",
            "normalization": "pixel/127.5-1.0",
            "augmentation": "none",
            "score_transform": "none_raw_quality_scalar",
            "higher_is_better": True,
        },
        "execution": {
            "device": str(device),
            "batch_size": batch_value,
            "maximum_samples": limit,
        },
        "row_count": int(len(result)),
        "score_summary": {
            "minimum": float(result["fiqa_score"].min()),
            "median": float(result["fiqa_score"].median()),
            "maximum": float(result["fiqa_score"].max()),
            "mean": float(result["fiqa_score"].mean()),
        },
    }
    manifest_base["fiqa_uid"] = (
        "fiqa-scores-" + canonical_sha256(manifest_base)[:24]
    )
    return result, manifest_base


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize_aligned_bundle_score_artifact(
    aligned_bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    model: Any,
    model_uid: str,
    checkpoint_sha256: str,
    variant: str,
    batch_size: int = 64,
    shard_size: int = 8_192,
    device: str = "cuda",
    overwrite: bool = False,
) -> FIQAScoreArtifact:
    """Resume shard-safe full-bundle inference and atomically publish one table."""

    destination = Path(output_dir).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"FIQA artifact already exists: {destination}")
    root = Path(aligned_bundle_dir).resolve()
    faces_path, index_path, bundle_manifest = _verify_aligned_bundle(root)
    batch_value = int(batch_size)
    shard_value = int(shard_size)
    if batch_value <= 0 or shard_value <= 0:
        raise ValueError("batch_size and shard_size must be positive")
    if shard_value < batch_value:
        raise ValueError("shard_size must be at least batch_size")

    index = pd.read_csv(index_path)
    required = {"sample_id", "aligned_face_index", "aligned_content_sha256"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"aligned index is missing columns: {missing}")
    if index.empty or index["sample_id"].astype(str).duplicated().any():
        raise ValueError("aligned index must contain unique sample IDs")
    face_indices = index["aligned_face_index"].to_numpy(dtype=np.int64)
    faces = np.load(faces_path, mmap_mode="r", allow_pickle=False)
    if faces.shape[1:] != (112, 112, 3) or faces.dtype != np.uint8:
        raise ValueError("aligned face array violates the FIQA input contract")
    if np.any(face_indices < 0) or int(face_indices.max()) >= len(faces):
        raise ValueError("aligned_face_index points outside aligned_faces.npy")

    work_root = destination.with_name(f".{destination.name}.inprogress")
    shard_root = work_root / "shards"
    aligned_manifest_sha256 = sha256_file(root / "bundle_manifest.json")
    job_contract: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "fiqa_score_extraction_work",
        "dataset_id": str(bundle_manifest["dataset_id"]),
        "model_uid": str(model_uid),
        "checkpoint_sha256": str(checkpoint_sha256),
        "variant": str(variant).upper(),
        "aligned_bundle_manifest_sha256": aligned_manifest_sha256,
        "aligned_faces_sha256": str(
            bundle_manifest["outputs"]["aligned_faces"]["sha256"]
        ),
        "aligned_index_sha256": str(
            bundle_manifest["outputs"]["aligned_index"]["sha256"]
        ),
        "row_count": int(len(index)),
        "batch_size": batch_value,
        "shard_size": shard_value,
        "device": str(device),
    }
    job_uid = "fiqa-work-" + canonical_sha256(job_contract)[:24]
    work_manifest_path = work_root / "manifest.json"
    if work_root.exists():
        if not work_manifest_path.is_file() or not shard_root.is_dir():
            raise ValueError(f"incomplete FIQA work directory: {work_root}")
        work_manifest = _read_json(work_manifest_path)
        if work_manifest.get("job_uid") != job_uid:
            raise ValueError(
                "existing FIQA work directory has a different extraction contract"
            )
    else:
        shard_root.mkdir(parents=True, exist_ok=False)
        work_manifest = {
            **job_contract,
            "job_uid": job_uid,
            "status": "in_progress",
            "shards": [],
        }
        _write_json_atomic(work_manifest_path, work_manifest)

    completed = {
        int(item["start"]): dict(item)
        for item in work_manifest.get("shards", [])
    }
    for start in range(0, len(index), shard_value):
        stop = min(start + shard_value, len(index))
        existing = completed.get(start)
        if existing is not None:
            shard_path = shard_root / str(existing["path"])
            if (
                int(existing.get("stop", -1)) != stop
                or not shard_path.is_file()
                or sha256_file(shard_path) != str(existing.get("sha256"))
            ):
                raise ValueError(f"FIQA work shard is invalid: {shard_path}")
            continue

        batch_indices = face_indices[start:stop]
        scores = infer_cr_fiqa_scores(
            model,
            np.asarray(faces[batch_indices]),
            batch_size=batch_value,
            device=device,
        )
        shard = index.iloc[start:stop][
            ["sample_id", "aligned_face_index", "aligned_content_sha256"]
        ].copy()
        shard["fiqa_score"] = scores
        shard["fiqa_model_uid"] = str(model_uid)
        name = f"part-{start:09d}-{stop:09d}.parquet"
        shard_path = shard_root / name
        temporary = shard_path.with_name(f".{name}.tmp-{uuid4().hex}")
        try:
            shard.to_parquet(temporary, index=False)
            os.replace(temporary, shard_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        entry = {
            "start": start,
            "stop": stop,
            "path": name,
            "row_count": int(len(shard)),
            "sha256": sha256_file(shard_path),
            "bytes": shard_path.stat().st_size,
        }
        completed[start] = entry
        work_manifest["shards"] = [
            completed[key] for key in sorted(completed)
        ]
        _write_json_atomic(work_manifest_path, work_manifest)

    parts: list[pd.DataFrame] = []
    expected_start = 0
    for entry in sorted(completed.values(), key=lambda item: int(item["start"])):
        start = int(entry["start"])
        stop = int(entry["stop"])
        if start != expected_start or stop <= start:
            raise ValueError("FIQA work shards are not contiguous")
        shard_path = shard_root / str(entry["path"])
        if not shard_path.is_file() or sha256_file(shard_path) != str(
            entry["sha256"]
        ):
            raise ValueError(f"FIQA work shard hash mismatch: {shard_path}")
        part = pd.read_parquet(shard_path)
        if len(part) != int(entry["row_count"]):
            raise ValueError(f"FIQA work shard row count mismatch: {shard_path}")
        parts.append(part)
        expected_start = stop
    if expected_start != len(index):
        raise ValueError("FIQA work shards do not cover the aligned bundle")
    result = pd.concat(parts, ignore_index=True)
    if not np.array_equal(
        result["sample_id"].astype(str).to_numpy(),
        index["sample_id"].astype(str).to_numpy(),
    ):
        raise ValueError("FIQA shard assembly changed sample order")
    if not np.isfinite(result["fiqa_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("FIQA shard assembly contains non-finite scores")

    manifest_base: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "fiqa_score_table",
        "status": "completed",
        "dataset_id": str(bundle_manifest["dataset_id"]),
        "model_family": "cr-fiqa",
        "variant": str(variant).upper(),
        "fiqa_model_uid": str(model_uid),
        "checkpoint_sha256": str(checkpoint_sha256),
        "aligned_bundle_manifest_sha256": aligned_manifest_sha256,
        "aligned_faces_sha256": str(
            bundle_manifest["outputs"]["aligned_faces"]["sha256"]
        ),
        "aligned_index_sha256": str(
            bundle_manifest["outputs"]["aligned_index"]["sha256"]
        ),
        "preprocessing": {
            "input": "aligned_uint8_nhwc_rgb_112x112",
            "tensor": "float32_nchw",
            "normalization": "pixel/127.5-1.0",
            "augmentation": "none",
            "score_transform": "none_raw_quality_scalar",
            "higher_is_better": True,
        },
        "execution": {
            "device": str(device),
            "batch_size": batch_value,
            "shard_size": shard_value,
            "resumable_shards": True,
            "work_job_uid": job_uid,
        },
        "row_count": int(len(result)),
        "score_summary": {
            "minimum": float(result["fiqa_score"].min()),
            "median": float(result["fiqa_score"].median()),
            "maximum": float(result["fiqa_score"].max()),
            "mean": float(result["fiqa_score"].mean()),
        },
    }
    manifest_base["fiqa_uid"] = (
        "fiqa-scores-" + canonical_sha256(manifest_base)[:24]
    )
    artifact = write_fiqa_score_artifact(
        destination,
        result,
        manifest_base,
        overwrite=overwrite,
    )
    shutil.rmtree(work_root)
    return artifact


def write_fiqa_score_artifact(
    output_dir: str | Path,
    scores: pd.DataFrame,
    manifest_base: dict[str, Any],
    *,
    overwrite: bool = False,
) -> FIQAScoreArtifact:
    """Atomically publish a compact FIQA score CSV and SHA-verified manifest."""

    destination = Path(output_dir).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"FIQA artifact already exists: {destination}")
    staging = destination.with_name(f".{destination.name}.tmp-{uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        score_path = staging / "fiqa_scores.csv"
        scores.to_csv(score_path, index=False, encoding="utf-8")
        manifest = dict(manifest_base)
        manifest["files"] = {
            "fiqa_scores.csv": {
                "sha256": sha256_file(score_path),
                "bytes": score_path.stat().st_size,
                "row_count": int(len(scores)),
                "columns": list(scores.columns),
            }
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            backup = destination.with_name(f".{destination.name}.old-{uuid4().hex}")
            os.replace(destination, backup)
            try:
                os.replace(staging, destination)
            except BaseException:
                os.replace(backup, destination)
                raise
            shutil.rmtree(backup)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_fiqa_score_artifact(destination)


def load_fiqa_score_artifact(directory: str | Path) -> FIQAScoreArtifact:
    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_type") != "fiqa_score_table":
        raise ValueError("unexpected FIQA artifact_type")
    if manifest.get("status") != "completed":
        raise ValueError("FIQA score artifact is not completed")
    entry = dict(manifest.get("files", {}).get("fiqa_scores.csv", {}))
    score_path = root / "fiqa_scores.csv"
    if not score_path.is_file() or sha256_file(score_path) != entry.get("sha256"):
        raise ValueError("FIQA score artifact hash mismatch")
    scores = pd.read_csv(score_path)
    required = {"sample_id", "fiqa_score", "fiqa_model_uid"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"FIQA score table is missing columns: {missing}")
    if len(scores) != int(entry.get("row_count", -1)):
        raise ValueError("FIQA score row count differs from its manifest")
    if scores["sample_id"].astype(str).duplicated().any():
        raise ValueError("FIQA score sample_id values must be unique")
    if not np.isfinite(scores["fiqa_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("FIQA scores must be finite")
    if scores["fiqa_model_uid"].astype(str).nunique() != 1:
        raise ValueError("FIQA score artifact mixes model UIDs")
    if scores["fiqa_model_uid"].astype(str).iloc[0] != str(
        manifest["fiqa_model_uid"]
    ):
        raise ValueError("FIQA score model UID differs from its manifest")
    return FIQAScoreArtifact(root=root, scores=scores, manifest=manifest)
