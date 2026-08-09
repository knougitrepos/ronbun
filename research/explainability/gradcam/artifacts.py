from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from collections.abc import Mapping
from uuid import uuid4

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


PREPARED_SCHEMA_VERSION = 3
SALIENCY_SCHEMA_VERSION = 3
SUPPORTED_PREPARED_SCHEMA_VERSIONS = {1, 2, PREPARED_SCHEMA_VERSION}
SUPPORTED_SALIENCY_SCHEMA_VERSIONS = {1, 2, SALIENCY_SCHEMA_VERSION}
_ATOMIC_REPLACE_ATTEMPTS = 8
_ATOMIC_REPLACE_INITIAL_DELAY_SECONDS = 0.05


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value:
        raise ValueError(f"{name} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _new_atomic_directory(destination: Path, *, overwrite: bool) -> Path:
    if destination.exists() and not overwrite:
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


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Replace an artifact path despite short-lived Windows file locks."""

    for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == _ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(_ATOMIC_REPLACE_INITIAL_DELAY_SECONDS * (2**attempt))


def _publish_atomic_directory(
    temporary: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> Path:
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"artifact directory appeared during write: {destination}"
        )
    if not destination.exists():
        _replace_with_retry(temporary, destination)
        return destination

    backup = destination.with_name(f".{destination.name}.backup-{uuid4().hex}")
    _replace_with_retry(destination, backup)
    try:
        _replace_with_retry(temporary, destination)
    except BaseException:
        _replace_with_retry(backup, destination)
        raise
    else:
        shutil.rmtree(backup)
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


def _read_index_frame(path: Path) -> pd.DataFrame:
    """Read the current CSV index while preserving old completed Parquet runs."""

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, low_memory=False)
        # Empty identity strings are meaningful in the LOO eligibility
        # contract; CSV's default NA parsing must not turn them into NaN.
        if "identity_id" in frame:
            frame["identity_id"] = frame["identity_id"].fillna("")
        return frame
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported tabular artifact format: {path}")


def _restore_frame_dtypes(
    frame: pd.DataFrame,
    dtypes: dict[str, object] | None,
) -> pd.DataFrame:
    if not dtypes:
        return frame
    restored = frame.copy()
    for column, dtype in dtypes.items():
        if column not in restored:
            raise ValueError(f"CSV artifact is missing dtype column: {column}")
        restored[column] = restored[column].astype(str(dtype))
    return restored


def write_prepared_population_artifact(
    prepared: PreparedPopulationInputs,
    destination: str | Path,
    *,
    shard_size: int = 2048,
    overwrite: bool = False,
) -> Path:
    """Write one complete allow_pickle=False embedding/LOO shard bundle."""

    output = Path(destination).resolve()
    rows_per_shard = _positive_integer(shard_size, name="shard_size")
    temporary = _new_atomic_directory(output, overwrite=overwrite)
    try:
        sample_path = temporary / "sample_index.csv"
        prepared.sample_frame().to_csv(
            sample_path,
            index=False,
            encoding="utf-8",
        )
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
                saliency_reference_identity_id=(
                    prepared.loo_templates.reference_identity_ids[start:stop].astype(
                        str
                    )
                    if prepared.loo_templates.reference_identity_ids is not None
                    else np.full(stop - start, "not_applicable")
                ),
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
            "target_name": prepared.loo_templates.target_name,
            "row_count": row_count,
            "embedding_dimension": int(prepared.raw_embeddings.shape[1]),
            "eligible_count": prepared.loo_templates.eligible_count,
            "sample_index": _file_entry(sample_path, root=temporary),
            "embedding_shards": shard_entries,
        }
        _write_json(temporary / "manifest.json", manifest)
        return _publish_atomic_directory(
            temporary,
            output,
            overwrite=overwrite,
        )
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
    if int(manifest.get("schema_version", -1)) not in (
        SUPPORTED_PREPARED_SCHEMA_VERSIONS
    ):
        raise ValueError("unsupported prepared population schema version")
    sample_index_path = _verify_entry(
        source,
        dict(manifest["sample_index"]),
    )
    sample_index = _read_index_frame(sample_index_path)
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
        target_name=str(manifest.get("target_name", LOO_TARGET_NAME)),
        reference_identity_ids=(
            combined["saliency_reference_identity_id"].astype(str)
            if "saliency_reference_identity_id" in combined
            else None
        ),
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
    persistence: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write scalar features and only the explicitly retained array payloads."""

    output = Path(destination).resolve()
    rows_per_shard = _positive_integer(shard_size, name="shard_size")
    if heatmap_dtype not in {"float16", "float32"}:
        raise ValueError("heatmap_dtype must be 'float16' or 'float32'")
    map_dtype = np.float16 if heatmap_dtype == "float16" else np.float32
    policy = dict(persistence or {})
    persist_scalar_features = bool(policy.get("persist_scalar_features", True))
    if not persist_scalar_features:
        raise ValueError(
            "population saliency scalar features are required by downstream analysis"
        )
    persist_normalized_heatmap = bool(
        policy.get("persist_normalized_heatmap", True)
    )
    persist_raw_cam = bool(policy.get("persist_raw_cam", True))
    persist_relu_cam = bool(policy.get("persist_relu_cam", True))
    persist_channel_weights = bool(policy.get("persist_channel_weights", True))
    persist_pass_b_embeddings = bool(
        policy.get("persist_pass_b_embeddings", True)
    )
    persisted_array_names = ["sample_id"]
    if persist_normalized_heatmap:
        persisted_array_names.append("normalized_heatmap")
    if persist_raw_cam:
        persisted_array_names.append("raw_cam")
    if persist_relu_cam:
        persisted_array_names.append("relu_cam")
    if persist_channel_weights:
        persisted_array_names.append("channel_weights")
    if persist_pass_b_embeddings:
        persisted_array_names.extend(
            (
                "pass_b_raw_embedding",
                "pass_b_raw_norm",
                "pass_b_normalized_embedding",
            )
        )
    temporary = _new_atomic_directory(output, overwrite=overwrite)
    try:
        feature_path = temporary / "saliency_features.csv"
        result.features.to_csv(
            feature_path,
            index=False,
            encoding="utf-8",
        )
        shard_dir = temporary / "heatmap_shards"
        shard_dir.mkdir()
        shard_entries: list[dict[str, object]] = []
        row_count = len(result.heatmap_sample_ids)
        for shard_index, start in enumerate(range(0, row_count, rows_per_shard)):
            stop = min(start + rows_per_shard, row_count)
            shard_path = shard_dir / f"part-{shard_index:05d}.npz"
            arrays: dict[str, np.ndarray] = {
                "sample_id": result.heatmap_sample_ids[start:stop].astype(str)
            }
            if persist_normalized_heatmap:
                arrays["normalized_heatmap"] = result.normalized_heatmaps[
                    start:stop
                ].astype(map_dtype, copy=False)
            if persist_raw_cam:
                arrays["raw_cam"] = result.raw_cams[start:stop].astype(
                    map_dtype, copy=False
                )
            if persist_relu_cam:
                arrays["relu_cam"] = result.relu_cams[start:stop].astype(
                    map_dtype, copy=False
                )
            if persist_channel_weights:
                arrays["channel_weights"] = result.channel_weights[
                    start:stop
                ].astype(np.float32, copy=False)
            if persist_pass_b_embeddings:
                arrays["pass_b_raw_embedding"] = result.pass_b_raw_embeddings[
                    start:stop
                ].astype(np.float32, copy=False)
                arrays["pass_b_raw_norm"] = result.pass_b_raw_norms[
                    start:stop
                ].astype(np.float32, copy=False)
                arrays["pass_b_normalized_embedding"] = (
                    result.pass_b_normalized_embeddings[start:stop].astype(
                        np.float32, copy=False
                    )
                )
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
        target_names = result.features["saliency_target_name"].astype(str).unique()
        if len(target_names) != 1:
            raise ValueError("population saliency artifact must have one target name")
        manifest = {
            "artifact_type": "population_gradcam_saliency",
            "schema_version": SALIENCY_SCHEMA_VERSION,
            "extraction_uid": result.extraction_uid,
            "dataset_id": result.dataset_id,
            "model_uid": result.model_uid,
            "origin_embedding_artifact_uid": result.origin_embedding_artifact_uid,
            "saliency_spec_uid": result.saliency_spec_uid,
            "target_layer": result.target_layer,
            "target_name": str(target_names[0]),
            "sample_feature_row_count": int(len(result.features)),
            "heatmap_row_count": row_count,
            "heatmap_dtype": heatmap_dtype,
            "persisted_array_names": persisted_array_names,
            "intermediate_tensors_persisted": (
                result.activations is not None and result.gradients is not None
            ),
            "saliency_features": _file_entry(
                feature_path,
                root=temporary,
                dtypes={
                    column: str(dtype)
                    for column, dtype in result.features.dtypes.items()
                },
            ),
            "heatmap_shards": shard_entries,
        }
        _write_json(temporary / "manifest.json", manifest)
        return _publish_atomic_directory(
            temporary,
            output,
            overwrite=overwrite,
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def read_population_saliency_features(directory: str | Path) -> pd.DataFrame:
    source = Path(directory).resolve()
    manifest = _verified_manifest(
        source,
        artifact_type="population_gradcam_saliency",
    )
    if int(manifest.get("schema_version", -1)) not in (
        SUPPORTED_SALIENCY_SCHEMA_VERSIONS
    ):
        raise ValueError("unsupported population saliency schema version")
    feature_entry = dict(manifest["saliency_features"])
    feature_path = _verify_entry(source, feature_entry)
    frame = _read_index_frame(feature_path)
    frame = _restore_frame_dtypes(
        frame,
        dict(feature_entry.get("dtypes", {})),
    )
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


def _validated_unique_row_indices(
    row_indices: np.ndarray | list[int],
    *,
    row_count: int,
    name: str,
) -> np.ndarray:
    indices = np.asarray(row_indices)
    if indices.ndim != 1 or indices.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be a one-dimensional integer vector")
    indices = indices.astype(np.int64, copy=False)
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} must not contain duplicates")
    if np.any(indices < 0) or np.any(indices >= int(row_count)):
        raise IndexError(f"{name} contains an out-of-range index")
    return indices


def read_population_heatmap_subset(
    directory: str | Path,
    heatmap_indices: np.ndarray | list[int],
    *,
    expected_sample_ids: np.ndarray | list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read and verify only selected heatmaps while preserving request order."""

    source = Path(directory).resolve()
    manifest = _verified_manifest(
        source,
        artifact_type="population_gradcam_saliency",
    )
    if int(manifest.get("schema_version", -1)) != SALIENCY_SCHEMA_VERSION:
        raise ValueError("unsupported population saliency schema version")
    indices = _validated_unique_row_indices(
        heatmap_indices,
        row_count=int(manifest["heatmap_row_count"]),
        name="heatmap_indices",
    )
    sample_ids = np.empty(len(indices), dtype=object)
    heatmaps: np.ndarray | None = None
    filled = np.zeros(len(indices), dtype=bool)
    for entry in list(manifest["heatmap_shards"]):
        item = dict(entry)
        start = int(item["row_start"])
        stop = int(item["row_stop"])
        positions = np.flatnonzero((indices >= start) & (indices < stop))
        if len(positions) == 0:
            continue
        shard_path = _verify_entry(source, item)
        local = indices[positions] - start
        with np.load(shard_path, allow_pickle=False) as shard:
            shard_ids = shard["sample_id"].astype(str)
            shard_heatmaps = shard["normalized_heatmap"].astype(np.float32)
        if heatmaps is None:
            heatmaps = np.empty(
                (len(indices), *shard_heatmaps.shape[1:]),
                dtype=np.float32,
            )
        elif heatmaps.shape[1:] != shard_heatmaps.shape[1:]:
            raise ValueError("heatmap shard shapes are inconsistent")
        sample_ids[positions] = shard_ids[local]
        heatmaps[positions] = shard_heatmaps[local]
        filled[positions] = True
    if not bool(filled.all()) or heatmaps is None:
        raise ValueError("selected heatmap rows were not fully resolved")
    sample_ids = sample_ids.astype(str)
    if expected_sample_ids is not None:
        expected = np.asarray(expected_sample_ids).astype(str)
        if expected.shape != sample_ids.shape or not np.array_equal(
            expected,
            sample_ids,
        ):
            raise ValueError("selected heatmap sample IDs differ from expectation")
    return sample_ids, heatmaps


def read_prepared_population_template_subset(
    directory: str | Path,
    row_indices: np.ndarray | list[int],
    *,
    expected_sample_ids: np.ndarray | list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read verified LOO/search targets for selected prepared-population rows."""

    source = Path(directory).resolve()
    manifest = _verified_manifest(
        source,
        artifact_type="prepared_population_saliency_inputs",
    )
    if int(manifest.get("schema_version", -1)) not in (
        SUPPORTED_PREPARED_SCHEMA_VERSIONS
    ):
        raise ValueError("unsupported prepared population schema version")
    indices = _validated_unique_row_indices(
        row_indices,
        row_count=int(manifest["row_count"]),
        name="row_indices",
    )
    sample_ids = np.empty(len(indices), dtype=object)
    templates: np.ndarray | None = None
    scores = np.empty(len(indices), dtype=np.float32)
    filled = np.zeros(len(indices), dtype=bool)
    for entry in list(manifest["embedding_shards"]):
        item = dict(entry)
        start = int(item["row_start"])
        stop = int(item["row_stop"])
        positions = np.flatnonzero((indices >= start) & (indices < stop))
        if len(positions) == 0:
            continue
        shard_path = _verify_entry(source, item)
        local = indices[positions] - start
        with np.load(shard_path, allow_pickle=False) as shard:
            shard_ids = shard["sample_id"].astype(str)
            shard_templates = shard["loo_template"].astype(np.float32)
            shard_scores = shard["identity_template_cosine"].astype(np.float32)
        if templates is None:
            templates = np.empty(
                (len(indices), shard_templates.shape[1]),
                dtype=np.float32,
            )
        elif templates.shape[1] != shard_templates.shape[1]:
            raise ValueError("prepared template dimensions are inconsistent")
        sample_ids[positions] = shard_ids[local]
        templates[positions] = shard_templates[local]
        scores[positions] = shard_scores[local]
        filled[positions] = True
    if not bool(filled.all()) or templates is None:
        raise ValueError("selected prepared-population rows were not fully resolved")
    sample_ids = sample_ids.astype(str)
    if expected_sample_ids is not None:
        expected = np.asarray(expected_sample_ids).astype(str)
        if expected.shape != sample_ids.shape or not np.array_equal(
            expected,
            sample_ids,
        ):
            raise ValueError("selected prepared sample IDs differ from expectation")
    return sample_ids, templates, scores
