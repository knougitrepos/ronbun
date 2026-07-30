"""Materialize one canonical ArcFace-style aligned-crop bundle.

The binary NPY array is accompanied by CSV indexes and a JSON manifest so the
artifact can be inspected without a Parquet reader. Ordinary images use
detect-and-align preprocessing. Datasets whose official protocol already
contains face crops use one explicit resize policy for every source image;
detector failure is never used to silently mix preprocessing modes.
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


ProgressCallback = Callable[[str, dict[str, object]], None]


def _emit(progress: ProgressCallback | None, message: str, **details: object) -> None:
    if progress is not None:
        progress(message, details)


MATERIALIZER_VERSION = "2.2.0"
ALIGNMENT_TEMPLATE_ID = "insightface_arcface_112_v1"
OFFICIAL_FACE_CROP_TEMPLATE_ID = "qmul_survface_official_face_crop_resize_112_v1"
DETECTOR_NAME = "buffalo_l"
DETECT_AND_ALIGN = "detect_and_align"
OFFICIAL_FACE_CROP_RESIZE = "official_face_crop_resize"
SUPPORTED_PREPROCESSING_MODES = (
    DETECT_AND_ALIGN,
    OFFICIAL_FACE_CROP_RESIZE,
)
DEFAULT_PROVIDERS = (
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)
DETECTOR_ALLOWED_MODULES = ("detection",)
ALIGNED_INDEX_COLUMNS = (
    "dataset_id",
    "sample_id",
    "identity_id",
    "split",
    "source_image_path",
    "source_content_sha256",
    "source_width",
    "source_height",
    "aligned_face_index",
    "aligned_content_sha256",
    "preprocessing_mode",
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


def validate_aligned_crop_bundle(
    bundle_dir: str | Path,
    *,
    dataset_id: str,
    expected_source_count: int,
    preprocessing_mode: str,
    require_full_coverage: bool,
) -> dict[str, Any]:
    """Validate the lightweight contract without rehashing a multi-GB array."""

    root = Path(bundle_dir).resolve()
    if not (root / "_SUCCESS").is_file():
        raise RuntimeError(f"aligned-crop bundle is incomplete: {root}")
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = manifest.get("counts", {})
    preprocessing = manifest.get("preprocessing", {})
    actual_mode = str(preprocessing.get("mode", DETECT_AND_ALIGN))
    source_count = int(counts.get("source", -1))
    aligned_count = int(counts.get("aligned", -1))
    failed_count = int(counts.get("failed", -1))
    problems: list[str] = []
    if str(manifest.get("dataset_id")) != str(dataset_id):
        problems.append(
            f"dataset_id={manifest.get('dataset_id')!r}, expected={dataset_id!r}"
        )
    if actual_mode != str(preprocessing_mode):
        problems.append(
            f"preprocessing_mode={actual_mode!r}, expected={preprocessing_mode!r}"
        )
    if source_count != int(expected_source_count):
        problems.append(
            f"source_count={source_count}, expected={int(expected_source_count)}"
        )
    if aligned_count < 0 or failed_count < 0 or (
        aligned_count + failed_count != source_count
    ):
        problems.append(
            "aligned/failed counts do not partition the source manifest"
        )
    if require_full_coverage and (
        failed_count != 0 or aligned_count != source_count
    ):
        problems.append(
            f"full coverage required but aligned={aligned_count}, "
            f"failed={failed_count}"
        )
    array_shape = (
        manifest.get("array_contract", {}).get("shape", [])
    )
    if not array_shape or int(array_shape[0]) != aligned_count:
        problems.append(
            f"array row count={array_shape[0] if array_shape else None}, "
            f"aligned={aligned_count}"
        )
    outputs = manifest.get("outputs", {})
    for name in ("aligned_faces", "aligned_index", "failed_samples"):
        entry = outputs.get(name)
        if not isinstance(entry, dict) or not (root / str(entry.get("path"))).is_file():
            problems.append(f"missing output={name}")
    if problems:
        raise RuntimeError(
            "aligned-crop bundle does not satisfy the current contract: "
            + "; ".join(problems)
        )
    return manifest


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
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    available_providers = tuple(ort.get_available_providers())
    primary_provider = providers[0]
    if primary_provider not in available_providers:
        raise RuntimeError(
            f"required primary ONNX Runtime provider is unavailable: "
            f"{primary_provider}; available={available_providers}"
        )

    detector = FaceAnalysis(
        name=detector_name,
        allowed_modules=list(DETECTOR_ALLOWED_MODULES),
        providers=list(providers),
    )
    detection_model = detector.models["detection"]
    active_providers = tuple(detection_model.session.get_providers())
    if not active_providers or active_providers[0] != primary_provider:
        raise RuntimeError(
            "ONNX Runtime did not activate the required primary provider: "
            f"required={primary_provider}, active={active_providers}"
        )
    detector._ronbun_session_providers = active_providers
    detector.prepare(ctx_id=0, det_size=detection_size)
    return detector


def _default_aligner(image_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    from insightface.utils.face_align import norm_crop

    return np.asarray(norm_crop(image_rgb, landmarks, image_size=112))


def _resize_official_face_crop(image_rgb: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    resized = image.resize((112, 112), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _close_memmap(array: np.memmap | None) -> None:
    if array is None:
        return
    array.flush()
    mmap = getattr(array, "_mmap", None)
    if mmap is not None and not mmap.closed:
        mmap.close()


def materialize_aligned_crops(
    manifest: pd.DataFrame,
    *,
    project_root: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    detector_name: str = DETECTOR_NAME,
    detection_size: tuple[int, int] = (640, 640),
    providers: tuple[str, ...] = DEFAULT_PROVIDERS,
    overwrite: bool = True,
    detector: Any | None = None,
    aligner: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    preprocessing_mode: str = DETECT_AND_ALIGN,
    require_full_coverage: bool = False,
    progress: ProgressCallback | None = None,
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
    selected_mode = str(preprocessing_mode).strip()
    if selected_mode not in SUPPORTED_PREPROCESSING_MODES:
        raise ValueError(
            "preprocessing_mode must be one of "
            f"{SUPPORTED_PREPROCESSING_MODES}, got {selected_mode!r}"
        )
    if selected_mode == OFFICIAL_FACE_CROP_RESIZE and (
        detector is not None or aligner is not None
    ):
        raise ValueError(
            "official_face_crop_resize does not accept detector or aligner overrides"
        )
    if selected_mode == OFFICIAL_FACE_CROP_RESIZE and not require_full_coverage:
        raise ValueError(
            "official_face_crop_resize requires require_full_coverage=True"
        )
    requested_providers = tuple(str(provider).strip() for provider in providers)
    if (
        not requested_providers
        or any(not provider for provider in requested_providers)
        or len(set(requested_providers)) != len(requested_providers)
    ):
        raise ValueError("providers must contain unique non-empty provider names")

    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"canonical aligned-crop result already exists: {destination}"
        )

    if selected_mode == DETECT_AND_ALIGN:
        active_detector = detector or _default_detector(
            detector_name,
            requested_providers,
            detection_size,
        )
        if detector is None:
            active_providers = tuple(active_detector._ronbun_session_providers)
        else:
            active_providers = tuple(
                getattr(
                    active_detector,
                    "_ronbun_session_providers",
                    ("injected_detector",),
                )
            )
        active_aligner = aligner or _default_aligner
    else:
        active_detector = None
        active_providers = ()
        active_aligner = None
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )

    total = int(len(manifest))
    faces_path = staging / "aligned_faces.npy"
    streamed_faces: np.memmap | None = None
    aligned_faces: list[np.ndarray] = []
    aligned_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    try:
        if selected_mode == OFFICIAL_FACE_CROP_RESIZE:
            streamed_faces = np.lib.format.open_memmap(
                faces_path,
                mode="w+",
                dtype=np.uint8,
                shape=(total, 112, 112, 3),
            )
        for processed, row in enumerate(manifest.itertuples(index=False), start=1):
            _emit(
                progress,
                "aligned crop materialization",
                processed=processed - 1,
                total=total,
                aligned=len(aligned_rows),
                failed=len(failed_rows),
            )
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

            source_height, source_width = image_rgb.shape[:2]
            if selected_mode == DETECT_AND_ALIGN:
                assert active_detector is not None
                assert active_aligner is not None
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
                landmarks = np.asarray(
                    _face_value(selected, "kps"),
                    dtype=np.float32,
                )
                bbox = np.asarray(
                    _face_value(selected, "bbox"),
                    dtype=np.float32,
                )
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
                            "alignment_failure_reason": (
                                "invalid_detection_geometry"
                            ),
                            "face_count": len(faces),
                        }
                    )
                    continue
                try:
                    crop = np.asarray(
                        active_aligner(image_rgb, landmarks),
                        dtype=np.uint8,
                    )
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
                face_count = len(faces)
                landmark_json = json.dumps(landmarks.flatten().tolist())
                row_detector_name = detector_name
                template_id = ALIGNMENT_TEMPLATE_ID
            else:
                try:
                    crop = _resize_official_face_crop(image_rgb)
                except Exception as exc:
                    failed_rows.append(
                        {
                            **base,
                            "alignment_failure_reason": (
                                f"resize_error:{type(exc).__name__}"
                            ),
                            "face_count": 1,
                        }
                    )
                    continue
                selected_index = 0
                score = float("nan")
                bbox = np.asarray(
                    [0.0, 0.0, float(source_width), float(source_height)],
                    dtype=np.float32,
                )
                face_count = 1
                landmark_json = ""
                row_detector_name = "not_applicable_official_face_crop"
                template_id = OFFICIAL_FACE_CROP_TEMPLATE_ID
            if crop.shape != (112, 112, 3):
                failed_rows.append(
                    {
                        **base,
                        "alignment_failure_reason": (
                            f"unexpected_aligned_shape:{tuple(crop.shape)}"
                        ),
                        "face_count": face_count,
                    }
                )
                continue

            aligned_face_index = len(aligned_rows)
            if streamed_faces is None:
                aligned_faces.append(crop)
            else:
                streamed_faces[aligned_face_index] = crop
            aligned_rows.append(
                {
                    **base,
                    "source_width": int(source_width),
                    "source_height": int(source_height),
                    "aligned_face_index": aligned_face_index,
                    "aligned_content_sha256": _crop_sha256(crop),
                    "preprocessing_mode": selected_mode,
                    "face_count": face_count,
                    "selected_face_index": selected_index,
                    "detection_score": score,
                    "bbox_x1": float(bbox[0]),
                    "bbox_y1": float(bbox[1]),
                    "bbox_x2": float(bbox[2]),
                    "bbox_y2": float(bbox[3]),
                    "landmark_5points_json": landmark_json,
                    "detector_name": row_detector_name,
                    "alignment_template_id": template_id,
                    "materializer_version": MATERIALIZER_VERSION,
                }
            )
        # Emit the terminal count so a milestone reporter always reaches 100%.
        _emit(
            progress,
            "aligned crop materialization",
            processed=total,
            total=total,
            aligned=len(aligned_rows),
            failed=len(failed_rows),
        )

        if require_full_coverage and failed_rows:
            reasons = pd.Series(
                [row["alignment_failure_reason"] for row in failed_rows],
                dtype=str,
            ).value_counts()
            examples = [row["sample_id"] for row in failed_rows[:5]]
            raise RuntimeError(
                f"{dataset_id} preprocessing requires complete source coverage: "
                f"source={total}, prepared={len(aligned_rows)}, "
                f"failed={len(failed_rows)}, reasons={reasons.to_dict()}, "
                f"examples={examples}"
            )
        if not aligned_rows:
            raise RuntimeError("no source image could be aligned")

        _close_memmap(streamed_faces)
        streamed_faces = None
        if selected_mode == OFFICIAL_FACE_CROP_RESIZE:
            face_shape = (len(aligned_rows), 112, 112, 3)
        else:
            face_array = np.stack(aligned_faces).astype(np.uint8, copy=False)
            np.save(faces_path, face_array, allow_pickle=False)
            face_shape = tuple(face_array.shape)
        aligned_index = pd.DataFrame(aligned_rows, columns=ALIGNED_INDEX_COLUMNS)
        failed_index = pd.DataFrame(failed_rows, columns=FAILED_INDEX_COLUMNS)
        aligned_path = staging / "aligned_index.csv"
        failed_path = staging / "failed_samples.csv"
        aligned_index.to_csv(aligned_path, index=False, encoding="utf-8")
        failed_index.to_csv(failed_path, index=False, encoding="utf-8")
        bundle_manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "aligned_face_crops",
            "dataset_id": str(dataset_id),
            "materializer_version": MATERIALIZER_VERSION,
            "alignment_template_id": (
                ALIGNMENT_TEMPLATE_ID
                if selected_mode == DETECT_AND_ALIGN
                else OFFICIAL_FACE_CROP_TEMPLATE_ID
            ),
            "preprocessing": {
                "mode": selected_mode,
                "require_full_coverage": bool(require_full_coverage),
                "resize_interpolation": (
                    None
                    if selected_mode == DETECT_AND_ALIGN
                    else "Pillow.Resampling.BILINEAR"
                ),
            },
            "detector": {
                "enabled": selected_mode == DETECT_AND_ALIGN,
                "name": detector_name,
                "detection_size": [int(value) for value in detection_size],
                "allowed_modules": list(DETECTOR_ALLOWED_MODULES),
                "requested_providers": list(requested_providers),
                "providers": list(active_providers),
            },
            "array_contract": {
                "shape": list(face_shape),
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
                    shape=list(face_shape),
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
        if selected_mode == OFFICIAL_FACE_CROP_RESIZE:
            published_faces = np.load(
                destination / "aligned_faces.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
        else:
            published_faces = face_array
        return AlignmentResult(
            aligned_faces=published_faces,
            aligned_index=aligned_index,
            failed_index=failed_index,
            bundle_manifest=bundle_manifest,
            output_dir=destination,
        )
    except BaseException:
        _close_memmap(streamed_faces)
        shutil.rmtree(staging, ignore_errors=True)
        raise
