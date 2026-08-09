from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from research.datasets.rfw import build_rfw_verification_bundle
from research.datasets.rfw_aligned_bin import iter_rfw_aligned_pair_batches
from research.preprocessing.aligned_crops import (
    ALIGNED_INDEX_COLUMNS,
    FAILED_INDEX_COLUMNS,
)
from research.runtime.hashing import sha256_file


RFW_CUSTOM_ALIGNED_BIN_MODE = "rfw_official_aligned_bin"
RFW_CUSTOM_ALIGNMENT_TEMPLATE_ID = "rfw_official_aligned_bin_112_v1"
RFW_CUSTOM_MATERIALIZER_VERSION = "rfw-custom-aligned-bin-v1"

ProgressCallback = Callable[[str, dict[str, object]], None]


def _emit(
    progress: ProgressCallback | None,
    message: str,
    **details: object,
) -> None:
    if progress is not None:
        progress(message, details)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _output_entry(path: Path, root: Path, **extra: object) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        **extra,
    }


def _publish_directory(
    staging: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    if not destination.exists():
        os.replace(staging, destination)
        return
    if not overwrite:
        raise FileExistsError(
            f"RFW custom aligned bundle already exists: {destination}"
        )
    backup = destination.with_name(f".{destination.name}.backup-{uuid4().hex}")
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    else:
        shutil.rmtree(backup)


def _pair_side_image_lookup(pairs: pd.DataFrame) -> dict[tuple[str, str], str]:
    required = {"pair_id", "left_image_id", "right_image_id"}
    missing = sorted(required.difference(pairs.columns))
    if missing:
        raise ValueError(f"RFW official pairs are missing columns: {missing}")
    lookup: dict[tuple[str, str], str] = {}
    for row in pairs.itertuples(index=False):
        pair_id = str(row.pair_id)
        for side, image_id in (
            ("left", str(row.left_image_id)),
            ("right", str(row.right_image_id)),
        ):
            key = (pair_id, side)
            if key in lookup:
                raise ValueError(f"duplicate RFW pair-side key: {key}")
            lookup[key] = image_id
    return lookup


def materialize_rfw_custom_aligned_bundle(
    source_manifest: pd.DataFrame,
    *,
    project_root: str | Path,
    jpg_archive_path: str | Path,
    aligned_bin_archive_path: str | Path,
    output_dir: str | Path,
    expected_jpg_archive_sha256: str,
    expected_aligned_bin_archive_sha256: str,
    dataset_id: str = "rfw_custom",
    preprocessing_mode: str = RFW_CUSTOM_ALIGNED_BIN_MODE,
    batch_size: int = 128,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Publish unique aligned RFW images for the custom 1:N protocol.

    The official BIN stores pair-side occurrences and may repeat the same
    image. This materializer resolves each occurrence through the validated
    official JPG pair manifest, verifies that repeated occurrences decode to
    identical RGB pixels, and writes exactly one 112x112 row per selected
    custom image. Official pair labels/folds are not used by the custom
    open-set evaluation.
    """

    required = {
        "image_id",
        "identity_id",
        "split",
        "image_path",
        "protocol_uid",
        "official_result_eligible",
    }
    missing = sorted(required.difference(source_manifest.columns))
    if missing:
        raise ValueError(f"RFW custom source manifest is missing columns: {missing}")
    if source_manifest.empty:
        raise ValueError("RFW custom source manifest must not be empty")
    source = source_manifest.copy().reset_index(drop=True)
    source["image_id"] = source["image_id"].astype(str)
    if source["image_id"].duplicated().any():
        raise ValueError("RFW custom image_id values must be unique")
    if source["official_result_eligible"].astype(bool).any():
        raise ValueError("RFW custom aligned inputs must not claim official status")
    protocol_uids = set(source["protocol_uid"].astype(str))
    if len(protocol_uids) != 1 or not next(iter(protocol_uids)).startswith(
        "rfw-custom-"
    ):
        raise ValueError("RFW custom protocol UID is invalid")
    if preprocessing_mode != RFW_CUSTOM_ALIGNED_BIN_MODE:
        raise ValueError(
            f"RFW custom preprocessing_mode must be {RFW_CUSTOM_ALIGNED_BIN_MODE!r}"
        )

    root = Path(project_root).expanduser().resolve()
    jpg_archive = Path(jpg_archive_path).expanduser().resolve()
    aligned_archive = Path(aligned_bin_archive_path).expanduser().resolve()
    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    for path in (root, jpg_archive, aligned_archive):
        if not path.exists():
            raise FileNotFoundError(path)

    expected_jpg = str(expected_jpg_archive_sha256).strip().lower()
    expected_aligned = str(expected_aligned_bin_archive_sha256).strip().lower()
    if len(expected_jpg) != 64 or len(expected_aligned) != 64:
        raise ValueError("RFW archive SHA-256 values must contain 64 hex characters")

    official = build_rfw_verification_bundle(
        jpg_archive,
        root,
        strict_official=True,
    )
    actual_jpg = str(official.summary["source_archive_sha256"]).lower()
    if actual_jpg != expected_jpg:
        raise ValueError(
            f"RFW JPG archive SHA-256 mismatch: {actual_jpg} != {expected_jpg}"
        )
    official_image_ids = set(official.manifest["image_id"].astype(str))
    selected_image_ids = set(source["image_id"])
    absent = sorted(selected_image_ids.difference(official_image_ids))
    if absent:
        raise ValueError(
            f"RFW custom images are absent from the official source: {absent[:3]}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    staging.mkdir(parents=False, exist_ok=False)
    faces_path = staging / "aligned_faces.npy"
    aligned_index_path = staging / "aligned_index.csv"
    failed_path = staging / "failed_samples.csv"
    try:
        ordered = source.reset_index(drop=True)
        output_index = {
            image_id: index for index, image_id in enumerate(ordered["image_id"])
        }
        faces = np.lib.format.open_memmap(
            faces_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(ordered), 112, 112, 3),
        )
        seen = np.zeros(len(ordered), dtype=bool)
        content_sha256: list[str | None] = [None] * len(ordered)
        lookup = _pair_side_image_lookup(official.pairs)
        processed_occurrences = 0
        try:
            for batch in iter_rfw_aligned_pair_batches(
                aligned_archive,
                official.pairs,
                batch_size=batch_size,
                expected_sha256=expected_aligned,
                strict_official=True,
            ):
                if len(batch.faces) != len(batch.occurrences):
                    raise RuntimeError("RFW aligned face/occurrence batch mismatch")
                for face, occurrence in zip(
                    batch.faces,
                    batch.occurrences.itertuples(index=False),
                    strict=True,
                ):
                    processed_occurrences += 1
                    key = (str(occurrence.pair_id), str(occurrence.side))
                    try:
                        image_id = lookup[key]
                    except KeyError as exc:
                        raise ValueError(
                            f"RFW aligned occurrence has no image mapping: {key}"
                        ) from exc
                    index = output_index.get(image_id)
                    if index is None:
                        continue
                    value = np.asarray(face, dtype=np.uint8)
                    if value.shape != (112, 112, 3):
                        raise ValueError(
                            f"RFW aligned face shape is invalid: {value.shape}"
                        )
                    digest = hashlib.sha256(value.tobytes(order="C")).hexdigest()
                    if seen[index]:
                        if content_sha256[index] != digest or not np.array_equal(
                            faces[index], value
                        ):
                            raise ValueError(
                                "RFW repeated pair occurrences decode differently: "
                                f"{image_id}"
                            )
                        continue
                    faces[index] = value
                    seen[index] = True
                    content_sha256[index] = digest
                _emit(
                    progress,
                    "RFW custom aligned BIN materialization",
                    processed_occurrences=processed_occurrences,
                    unique_images=int(seen.sum()),
                    expected_unique_images=len(ordered),
                )
            if not seen.all():
                missing_ids = ordered.loc[~seen, "image_id"].head(5).tolist()
                raise RuntimeError(
                    "RFW aligned BIN did not cover every custom image: "
                    f"{missing_ids}"
                )
            faces.flush()
        finally:
            del faces

        aligned_rows: list[dict[str, object]] = []
        source_locator = _relative_or_absolute(aligned_archive, root)
        for index, row in enumerate(ordered.itertuples(index=False)):
            digest = content_sha256[index]
            if digest is None:
                raise RuntimeError("RFW custom aligned content digest is missing")
            aligned_rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "sample_id": str(row.image_id),
                    "identity_id": str(row.identity_id),
                    "split": str(row.split),
                    "source_image_path": (
                        f"rfw-aligned-bin://{source_locator}#{row.image_id}"
                    ),
                    "source_content_sha256": digest,
                    "source_width": 112,
                    "source_height": 112,
                    "aligned_face_index": index,
                    "aligned_content_sha256": digest,
                    "preprocessing_mode": preprocessing_mode,
                    "face_count": 1,
                    "selected_face_index": 0,
                    "detection_score": 1.0,
                    "bbox_x1": 0.0,
                    "bbox_y1": 0.0,
                    "bbox_x2": 112.0,
                    "bbox_y2": 112.0,
                    "landmark_5points_json": "[]",
                    "detector_name": "not_used_official_aligned_bin",
                    "alignment_template_id": RFW_CUSTOM_ALIGNMENT_TEMPLATE_ID,
                    "materializer_version": RFW_CUSTOM_MATERIALIZER_VERSION,
                }
            )
        aligned_index = pd.DataFrame(aligned_rows, columns=ALIGNED_INDEX_COLUMNS)
        failed = pd.DataFrame(columns=FAILED_INDEX_COLUMNS)
        aligned_index.to_csv(
            aligned_index_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        failed.to_csv(
            failed_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        manifest = {
            "schema_version": 1,
            "artifact_type": "aligned_face_crops",
            "dataset_id": str(dataset_id),
            "materializer_version": RFW_CUSTOM_MATERIALIZER_VERSION,
            "alignment_template_id": RFW_CUSTOM_ALIGNMENT_TEMPLATE_ID,
            "protocol_uid": next(iter(protocol_uids)),
            "official_protocol": False,
            "official_result_eligible": False,
            "checkpoint_overlap_status": "UNKNOWN",
            "preprocessing": {
                "mode": preprocessing_mode,
                "require_full_coverage": True,
                "source_representation": "official_aligned_pair_bin_deduplicated",
            },
            "detector": {
                "enabled": False,
                "name": "not_used_official_aligned_bin",
                "requested_providers": [],
                "providers": [],
            },
            "array_contract": {
                "shape": [len(ordered), 112, 112, 3],
                "dtype": "uint8",
                "layout": "nhwc",
                "color_order": "rgb",
                "image_size": [112, 112],
            },
            "counts": {
                "source": int(len(ordered)),
                "aligned": int(len(ordered)),
                "failed": 0,
                "official_pair_occurrences_scanned": int(processed_occurrences),
            },
            "source_archives": {
                "jpg": {
                    "path": _relative_or_absolute(jpg_archive, root),
                    "sha256": actual_jpg,
                },
                "aligned_bin": {
                    "path": source_locator,
                    "sha256": expected_aligned,
                },
            },
            "outputs": {
                "aligned_faces": _output_entry(
                    faces_path,
                    staging,
                    shape=[len(ordered), 112, 112, 3],
                    dtype="uint8",
                ),
                "aligned_index": _output_entry(
                    aligned_index_path,
                    staging,
                    row_count=len(aligned_index),
                ),
                "failed_samples": _output_entry(
                    failed_path,
                    staging,
                    row_count=0,
                ),
            },
        }
        (staging / "bundle_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (staging / "_SUCCESS").write_text(
            "complete\n",
            encoding="utf-8",
            newline="\n",
        )
        _publish_directory(staging, destination, overwrite=overwrite)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
