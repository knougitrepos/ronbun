from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from research.explainability.gradcam.extraction import (
    PopulationSaliencyResult,
    PreparedPopulationInputs,
)
from research.explainability.gradcam.templates import (
    LOO_TARGET_NAME,
    LeaveOneOutTemplateBundle,
)
from research.runtime.hashing import sha256_file


PREPARED_SCHEMA_VERSION = 1
SALIENCY_SCHEMA_VERSION = 1


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value:
        raise ValueError(f"{name} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _new_atomic_directory(destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(
            f"artifact directory already exists and will not be overwritten: "
            f"{destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )


def _publish_atomic_directory(temporary: Path, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(
            f"artifact directory appeared during write: {destination}"
        )
    os.replace(temporary, destination)
    return destination


def _file_entry(path: Path, *, root: Path, **extra: object) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        **extra,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_prepared_population_artifact(
    prepared: PreparedPopulationInputs,
    destination: str | Path,
    *,
    shard_size: int = 2048,
) -> Path:
    """Write immutable, allow_pickle=False embedding/LOO shards."""

    output = Path(destination).resolve()
    rows_per_shard = _positive_integer(shard_size, name="shard_size")
    temporary = _new_atomic_directory(output)
    try:
        sample_path = temporary / "sample_index.parquet"
        prepared.sample_frame().to_parquet(sample_path, index=False)
        shard_dir = temporary / "embedding_shards"
        shard_dir.mkdir()
        shard_entries: list[dict[str, object]] = []
        row_count = len(prepared.sample_ids)
        for shard_index, start in enumerate(range(0, row_count, rows_per_shard)):
            stop = min(start + rows_per_shard, row_count)
            shard_path = shard_dir / f"part-{shard_index:05d}.npz"
            np.savez_compressed(
                shard_path,
                sample_id=prepared.sample_ids[start:stop].astype(str),
                identity_id=prepared.identity_ids[start:stop].astype(str),
                scope_id=prepared.scope_ids[start:stop].astype(str),
                raw_embedding=prepared.raw_embeddings[start:stop].astype(
                    np.float32,
                    copy=False,
                ),
                raw_norm=prepared.raw_norms[start:stop].astype(
                    np.float32,
                    copy=False,
                ),
                normalized_embedding=prepared.normalized_embeddings[start:stop].astype(
                    np.float32, copy=False
                ),
                loo_template=prepared.loo_templates.templates[start:stop].astype(
                    np.float32,
                    copy=False,
                ),
                template_member_count=prepared.loo_templates.template_member_counts[
                    start:stop
                ].astype(np.int32, copy=False),
                saliency_target_eligible=prepared.loo_templates.eligible[
                    start:stop
                ].astype(bool, copy=False),
                saliency_target_status=prepared.loo_templates.exclusion_reasons[
                    start:stop
                ].astype(str),
                identity_template_cosine=prepared.loo_templates.target_scores[
                    start:stop
                ].astype(np.float32, copy=False),
            )
            shard_entries.append(
                _file_entry(
                    shard_path,
                    root=temporary,
                    row_start=start,
                    row_stop=stop,
                    row_count=stop - start,
                )
            )
        manifest = {
            "artifact_type": "prepared_population_saliency_inputs",
            "schema_version": PREPARED_SCHEMA_VERSION,
            "extraction_uid": prepared.extraction_uid,
            "dataset_id": prepared.dataset_id,
            "model_uid": prepared.model_uid,
            "checkpoint_sha256": prepared.checkpoint_sha256,
            "preprocess_hash": prepared.preprocess_hash,
            "origin_embedding_artifact_uid": (prepared.origin_embedding_artifact_uid),
            "target_name": LOO_TARGET_NAME,
            "row_count": row_count,
            "embedding_dimension": int(prepared.raw_embeddings.shape[1]),
            "eligible_count": prepared.loo_templates.eligible_count,
            "sample_index": _file_entry(sample_path, root=temporary),
            "embedding_shards": shard_entries,
        }
        _write_json(temporary / "manifest.json", manifest)
        return _publish_atomic_directory(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verified_manifest(directory: Path, *, artifact_type: str) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"artifact manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != artifact_type:
        raise ValueError(f"unexpected artifact_type={payload.get('artifact_type')!r}")
    return payload


def _verify_entry(directory: Path, entry: dict[str, object]) -> Path:
    path = (directory / str(entry["path"])).resolve()
    root = directory.resolve()
    if root not in path.parents:
        raise ValueError(f"artifact entry escapes directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"artifact entry is missing: {path}")
    actual = sha256_file(path)
    if actual != str(entry["sha256"]):
        raise ValueError(f"artifact entry hash mismatch: {path}")
    return path


def read_prepared_population_artifact(
    directory: str | Path,
) -> PreparedPopulationInputs:
    source = Path(directory).resolve()
    manifest = _verified_manifest(
        source,
        artifact_type="prepared_population_saliency_inputs",
    )
    if int(manifest.get("schema_version", -1)) != PREPARED_SCHEMA_VERSION:
        raise ValueError("unsupported prepared population schema version")
    sample_index_path = _verify_entry(
        source,
        dict(manifest["sample_index"]),
    )
    sample_index = pd.read_parquet(sample_index_path)
    parts: dict[str, list[np.ndarray]] = {}
    expected_start = 0
    for entry in list(manifest["embedding_shards"]):
        item = dict(entry)
        if int(item["row_start"]) != expected_start:
            raise ValueError("embedding shard row ranges are not contiguous")
        shard_path = _verify_entry(source, item)
        with np.load(shard_path, allow_pickle=False) as shard:
            for name in shard.files:
                parts.setdefault(name, []).append(np.asarray(shard[name]))
        expected_start = int(item["row_stop"])
    if expected_start != int(manifest["row_count"]):
        raise ValueError("embedding shard rows do not match manifest row_count")
    combined = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
    if len(sample_index) != int(manifest["row_count"]):
        raise ValueError("sample index row count does not match manifest")
    if "sample_id" not in sample_index:
        raise ValueError("sample index is missing sample_id")
    if not np.array_equal(
        sample_index["sample_id"].astype(str).to_numpy(),
        combined["sample_id"].astype(str),
    ):
        raise ValueError("sample index order differs from embedding shards")
    loo = LeaveOneOutTemplateBundle(
        sample_ids=combined["sample_id"].astype(str),
        identity_ids=combined["identity_id"].astype(str),
        scope_ids=combined["scope_id"].astype(str),
        templates=combined["loo_template"].astype(np.float32),
        template_member_counts=combined["template_member_count"].astype(np.int32),
        eligible=combined["saliency_target_eligible"].astype(bool),
        exclusion_reasons=combined["saliency_target_status"].astype(str),
        target_scores=combined["identity_template_cosine"].astype(np.float32),
        model_uid=str(manifest["model_uid"]),
    )
    return PreparedPopulationInputs(
        extraction_uid=str(manifest["extraction_uid"]),
        dataset_id=str(manifest["dataset_id"]),
        sample_ids=loo.sample_ids,
        identity_ids=loo.identity_ids,
        scope_ids=loo.scope_ids,
        raw_embeddings=combined["raw_embedding"].astype(np.float32),
        raw_norms=combined["raw_norm"].astype(np.float32),
        normalized_embeddings=combined["normalized_embedding"].astype(np.float32),
        loo_templates=loo,
        model_uid=str(manifest["model_uid"]),
        checkpoint_sha256=str(manifest["checkpoint_sha256"]),
        preprocess_hash=str(manifest["preprocess_hash"]),
        origin_embedding_artifact_uid=str(manifest["origin_embedding_artifact_uid"]),
    )


def write_population_saliency_artifact(
    result: PopulationSaliencyResult,
    destination: str | Path,
    *,
    shard_size: int = 2048,
    heatmap_dtype: str = "float16",
) -> Path:
    """Write full-population features and native-resolution heatmap shards."""

    output = Path(destination).resolve()
    rows_per_shard = _positive_integer(shard_size, name="shard_size")
    if heatmap_dtype not in {"float16", "float32"}:
        raise ValueError("heatmap_dtype must be 'float16' or 'float32'")
    map_dtype = np.float16 if heatmap_dtype == "float16" else np.float32
    temporary = _new_atomic_directory(output)
    try:
        feature_path = temporary / "saliency_features.parquet"
        result.features.to_parquet(feature_path, index=False)
        shard_dir = temporary / "heatmap_shards"
        shard_dir.mkdir()
        shard_entries: list[dict[str, object]] = []
        row_count = len(result.heatmap_sample_ids)
        for shard_index, start in enumerate(range(0, row_count, rows_per_shard)):
            stop = min(start + rows_per_shard, row_count)
            shard_path = shard_dir / f"part-{shard_index:05d}.npz"
            arrays: dict[str, np.ndarray] = {
                "sample_id": result.heatmap_sample_ids[start:stop].astype(str),
                "normalized_heatmap": result.normalized_heatmaps[start:stop].astype(
                    map_dtype,
                    copy=False,
                ),
                "raw_cam": result.raw_cams[start:stop].astype(
                    map_dtype,
                    copy=False,
                ),
                "relu_cam": result.relu_cams[start:stop].astype(
                    map_dtype,
                    copy=False,
                ),
                "channel_weights": result.channel_weights[start:stop].astype(
                    np.float32,
                    copy=False,
                ),
                "pass_b_raw_embedding": result.pass_b_raw_embeddings[start:stop].astype(
                    np.float32, copy=False
                ),
                "pass_b_raw_norm": result.pass_b_raw_norms[start:stop].astype(
                    np.float32,
                    copy=False,
                ),
                "pass_b_normalized_embedding": (
                    result.pass_b_normalized_embeddings[start:stop].astype(
                        np.float32,
                        copy=False,
                    )
                ),
            }
            if result.activations is not None:
                arrays["activation"] = result.activations[start:stop].astype(
                    np.float32,
                    copy=False,
                )
            if result.gradients is not None:
                arrays["gradient"] = result.gradients[start:stop].astype(
                    np.float32,
                    copy=False,
                )
            np.savez_compressed(shard_path, **arrays)
            shard_entries.append(
                _file_entry(
                    shard_path,
                    root=temporary,
                    row_start=start,
                    row_stop=stop,
                    row_count=stop - start,
                )
            )
        manifest = {
            "artifact_type": "population_gradcam_saliency",
            "schema_version": SALIENCY_SCHEMA_VERSION,
            "extraction_uid": result.extraction_uid,
            "dataset_id": result.dataset_id,
            "model_uid": result.model_uid,
            "origin_embedding_artifact_uid": result.origin_embedding_artifact_uid,
            "saliency_spec_uid": result.saliency_spec_uid,
            "target_layer": result.target_layer,
            "target_name": LOO_TARGET_NAME,
            "sample_feature_row_count": int(len(result.features)),
            "heatmap_row_count": row_count,
            "heatmap_dtype": heatmap_dtype,
            "intermediate_tensors_persisted": (
                result.activations is not None and result.gradients is not None
            ),
            "saliency_features": _file_entry(feature_path, root=temporary),
            "heatmap_shards": shard_entries,
        }
        _write_json(temporary / "manifest.json", manifest)
        return _publish_atomic_directory(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def read_population_saliency_features(directory: str | Path) -> pd.DataFrame:
    source = Path(directory).resolve()
    manifest = _verified_manifest(
        source,
        artifact_type="population_gradcam_saliency",
    )
    if int(manifest.get("schema_version", -1)) != SALIENCY_SCHEMA_VERSION:
        raise ValueError("unsupported population saliency schema version")
    feature_path = _verify_entry(source, dict(manifest["saliency_features"]))
    frame = pd.read_parquet(feature_path)
    if len(frame) != int(manifest["sample_feature_row_count"]):
        raise ValueError("saliency feature row count does not match manifest")
    return frame


def read_population_heatmaps(
    directory: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    source = Path(directory).resolve()
    manifest = _verified_manifest(
        source,
        artifact_type="population_gradcam_saliency",
    )
    if int(manifest.get("schema_version", -1)) != SALIENCY_SCHEMA_VERSION:
        raise ValueError("unsupported population saliency schema version")
    sample_parts: list[np.ndarray] = []
    heatmap_parts: list[np.ndarray] = []
    expected_start = 0
    for entry in list(manifest["heatmap_shards"]):
        item = dict(entry)
        if int(item["row_start"]) != expected_start:
            raise ValueError("heatmap shard row ranges are not contiguous")
        shard_path = _verify_entry(source, item)
        with np.load(shard_path, allow_pickle=False) as shard:
            sample_parts.append(shard["sample_id"].astype(str))
            heatmap_parts.append(shard["normalized_heatmap"].astype(np.float32))
        expected_start = int(item["row_stop"])
    if expected_start != int(manifest["heatmap_row_count"]):
        raise ValueError("heatmap shard rows do not match manifest row_count")
    return (
        np.concatenate(sample_parts, axis=0),
        np.concatenate(heatmap_parts, axis=0),
    )
