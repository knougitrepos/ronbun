"""Materialize one canonical ArcFace-style aligned-crop bundle.

The binary NPY array is accompanied by CSV indexes and a JSON manifest so the
artifact can be inspected without a Parquet reader. Detection failures remain
explicit; the quantitative path never substitutes a center crop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from PIL import Image

from research.runtime.hashing import sha256_file


MATERIALIZER_VERSION = "2.0.0"
ALIGNMENT_TEMPLATE_ID = "insightface_arcface_112_v1"
DETECTOR_NAME = "buffalo_l"
ALIGNED_INDEX_COLUMNS = (
    "dataset_id",
    "sample_id",
    "identity_id",
    "split",
    "source_image_path",
    "source_content_sha256",
    "aligned_face_index",
    "aligned_content_sha256",
    "face_count",
    "selected_face_index",
    "detection_score",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "landmark_5points_json",
    "detector_name",
    "alignment_template_id",
    "materializer_version",
)
FAILED_INDEX_COLUMNS = (
    "dataset_id",
    "sample_id",
    "identity_id",
    "split",
    "source_image_path",
    "source_content_sha256",
    "alignment_failure_reason",
    "face_count",
)


@dataclass(frozen=True)
class AlignmentResult:
    """Published aligned-crop bundle and its in-memory indexes."""

    aligned_faces: np.ndarray
    aligned_index: pd.DataFrame
    failed_index: pd.DataFrame
    bundle_manifest: dict[str, Any]
    output_dir: Path

    @property
    def aligned_manifest(self) -> pd.DataFrame:
        """Compatibility alias for callers using the old proposed name."""

        return self.aligned_index

    @property
    def failed_manifest(self) -> pd.DataFrame:
        """Compatibility alias for callers using the old proposed name."""

        return self.failed_index

    @property
    def bundle_metadata(self) -> dict[str, Any]:
        """Compatibility alias for callers using the old proposed name."""

        return self.bundle_manifest


def _face_value(face: Any, name: str) -> Any:
    if isinstance(face, dict):
        return face[name]
    return getattr(face, name)


def _face_area(face: Any) -> float:
    bbox = np.asarray(_face_value(face, "bbox"), dtype=np.float64)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        return -1.0
    return max(0.0, float(bbox[2] - bbox[0])) * max(
        0.0, float(bbox[3] - bbox[1])
    )


def _select_best_face(faces: Sequence[Any]) -> tuple[int, Any]:
    """Select deterministically by detection score, area, then input order."""

    if not faces:
        raise ValueError("faces must not be empty")
    ranked = []
    for index, face in enumerate(faces):
        score = float(_face_value(face, "det_score"))
        if not np.isfinite(score):
            score = float("-inf")
        ranked.append((score, _face_area(face), -index, index, face))
    _, _, _, index, face = max(ranked, key=lambda item: item[:3])
    return index, face


def _crop_sha256(face: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(face).tobytes()).hexdigest()


def _relative_output_entry(path: Path, root: Path, **extra: Any) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        **extra,
    }


def _publish_single_result(staging: Path, destination: Path, *, overwrite: bool) -> None:
    """Publish a complete directory while keeping at most one canonical result."""

    if not destination.exists():
        os.replace(staging, destination)
        return
    if not overwrite:
        raise FileExistsError(
            f"canonical aligned-crop result already exists: {destination}"
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


def _default_detector(
    detector_name: str,
    providers: tuple[str, ...],
    detection_size: tuple[int, int],
) -> Any:
    from insightface.app import FaceAnalysis

    detector = FaceAnalysis(name=detector_name, providers=list(providers))
    detector.prepare(ctx_id=0, det_size=detection_size)
    return detector


def _default_aligner(image_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    from insightface.utils.face_align import norm_crop

    return np.asarray(norm_crop(image_rgb, landmarks, image_size=112))


def materialize_aligned_crops(
    manifest: pd.DataFrame,
    *,
    project_root: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    detector_name: str = DETECTOR_NAME,
    detection_size: tuple[int, int] = (640, 640),
    providers: tuple[str, ...] = ("CPUExecutionProvider",),
    overwrite: bool = True,
    detector: Any | None = None,
    aligner: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> AlignmentResult:
    """Create the common RGB uint8 NHWC 112x112 crop bundle.

    ``overwrite=True`` replaces the canonical bundle only after a complete
    replacement has been staged. Completed experiment runs that consume this
    bundle remain immutable and retain the recorded input hash.
    """

    required = {"image_id", "identity_id", "split", "image_path"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")
    if manifest.empty:
        raise ValueError("manifest must not be empty")
    if not str(dataset_id).strip():
        raise ValueError("dataset_id must not be empty")
    if manifest["image_id"].astype(str).duplicated().any():
        duplicates = sorted(
            manifest.loc[
                manifest["image_id"].astype(str).duplicated(keep=False),
                "image_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(f"image_id must be unique; duplicates={duplicates[:5]}")
    if len(detection_size) != 2 or min(int(value) for value in detection_size) <= 0:
        raise ValueError("detection_size must contain two positive integers")

    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"canonical aligned-crop result already exists: {destination}"
        )

    active_detector = detector or _default_detector(
        detector_name,
        providers,
        detection_size,
    )
    active_aligner = aligner or _default_aligner
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )

    aligned_faces: list[np.ndarray] = []
    aligned_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    try:
        for row in manifest.itertuples(index=False):
            image_path = Path(str(row.image_path))
            if not image_path.is_absolute():
                image_path = root / image_path
            image_path = image_path.resolve()
            base = {
                "dataset_id": str(dataset_id),
                "sample_id": str(row.image_id),
                "identity_id": str(row.identity_id),
                "split": str(row.split),
                "source_image_path": str(image_path),
                "source_content_sha256": "",
            }
            if not image_path.is_file():
                failed_rows.append(
                    {
                        **base,
                        "alignment_failure_reason": "source_file_not_found",
                        "face_count": 0,
                    }
                )
                continue
            base["source_content_sha256"] = sha256_file(image_path)
            try:
                with Image.open(image_path) as image:
                    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            except Exception as exc:
                failed_rows.append(
                    {
                        **base,
                        "alignment_failure_reason": (
                            f"image_load_error:{type(exc).__name__}"
                        ),
                        "face_count": 0,
                    }
                )
                continue

            try:
                faces = list(active_detector.get(image_rgb[..., ::-1].copy()))
            except Exception as exc:
                failed_rows.append(
                    {
                        **base,
                        "alignment_failure_reason": (
                            f"detection_error:{type(exc).__name__}"
                        ),
                        "face_count": 0,
                    }
                )
                continue
            if not faces:
                failed_rows.append(
                    {
                        **base,
                        "alignment_failure_reason": "no_face_detected",
                        "face_count": 0,
                    }
                )
                continue

            selected_index, selected = _select_best_face(faces)
            landmarks = np.asarray(_face_value(selected, "kps"), dtype=np.float32)
            bbox = np.asarray(_face_value(selected, "bbox"), dtype=np.float32)
            score = float(_face_value(selected, "det_score"))
            if (
                landmarks.shape != (5, 2)
                or bbox.shape != (4,)
                or not np.isfinite(landmarks).all()
                or not np.isfinite(bbox).all()
                or not np.isfinite(score)
            ):
                failed_rows.append(
                    {
                        **base,
                        "alignment_failure_reason": "invalid_detection_geometry",
                        "face_count": len(faces),
                    }
                )
                continue
            try:
                crop = np.asarray(active_aligner(image_rgb, landmarks), dtype=np.uint8)
            except Exception as exc:
                failed_rows.append(
                    {
                        **base,
                        "alignment_failure_reason": (
                            f"alignment_error:{type(exc).__name__}"
                        ),
                        "face_count": len(faces),
                    }
                )
                continue
            if crop.shape != (112, 112, 3):
                failed_rows.append(
                    {
                        **base,
                        "alignment_failure_reason": (
                            f"unexpected_aligned_shape:{tuple(crop.shape)}"
                        ),
                        "face_count": len(faces),
                    }
                )
                continue

            aligned_face_index = len(aligned_faces)
            aligned_faces.append(crop)
            aligned_rows.append(
                {
                    **base,
                    "aligned_face_index": aligned_face_index,
                    "aligned_content_sha256": _crop_sha256(crop),
                    "face_count": len(faces),
                    "selected_face_index": selected_index,
                    "detection_score": score,
                    "bbox_x1": float(bbox[0]),
                    "bbox_y1": float(bbox[1]),
                    "bbox_x2": float(bbox[2]),
                    "bbox_y2": float(bbox[3]),
                    "landmark_5points_json": json.dumps(landmarks.flatten().tolist()),
                    "detector_name": detector_name,
                    "alignment_template_id": ALIGNMENT_TEMPLATE_ID,
                    "materializer_version": MATERIALIZER_VERSION,
                }
            )

        if not aligned_faces:
            raise RuntimeError("no source image could be aligned")

        face_array = np.stack(aligned_faces).astype(np.uint8, copy=False)
        aligned_index = pd.DataFrame(aligned_rows, columns=ALIGNED_INDEX_COLUMNS)
        failed_index = pd.DataFrame(failed_rows, columns=FAILED_INDEX_COLUMNS)
        faces_path = staging / "aligned_faces.npy"
        aligned_path = staging / "aligned_index.csv"
        failed_path = staging / "failed_samples.csv"
        np.save(faces_path, face_array, allow_pickle=False)
        aligned_index.to_csv(aligned_path, index=False, encoding="utf-8")
        failed_index.to_csv(failed_path, index=False, encoding="utf-8")
        bundle_manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "aligned_face_crops",
            "dataset_id": str(dataset_id),
            "materializer_version": MATERIALIZER_VERSION,
            "alignment_template_id": ALIGNMENT_TEMPLATE_ID,
            "detector": {
                "name": detector_name,
                "detection_size": [int(value) for value in detection_size],
                "providers": list(providers),
            },
            "array_contract": {
                "shape": list(face_array.shape),
                "dtype": "uint8",
                "layout": "nhwc",
                "color_order": "rgb",
                "image_size": [112, 112],
            },
            "counts": {
                "source": int(len(manifest)),
                "aligned": int(len(aligned_index)),
                "failed": int(len(failed_index)),
            },
            "outputs": {
                "aligned_faces": _relative_output_entry(
                    faces_path,
                    staging,
                    shape=list(face_array.shape),
                    dtype="uint8",
                ),
                "aligned_index": _relative_output_entry(
                    aligned_path,
                    staging,
                    row_count=len(aligned_index),
                ),
                "failed_samples": _relative_output_entry(
                    failed_path,
                    staging,
                    row_count=len(failed_index),
                ),
            },
        }
        (staging / "bundle_manifest.json").write_text(
            json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "_SUCCESS").write_text("complete\n", encoding="utf-8")
        _publish_single_result(staging, destination, overwrite=overwrite)
        return AlignmentResult(
            aligned_faces=face_array,
            aligned_index=aligned_index,
            failed_index=failed_index,
            bundle_manifest=bundle_manifest,
            output_dir=destination,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
