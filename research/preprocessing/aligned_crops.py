"""Common aligned 112×112 crop materializer for face recognition experiments.

This module produces the frozen, dataset-agnostic aligned crop arrays that all
Step 2 FR checkpoints (ArcFace, AdaFace, MagFace) consume.  The alignment uses
InsightFace's standard 5-point landmark similarity transformation via
``insightface.utils.face_align.norm_crop``, matching the ArcFace training
pipeline (Deng et al., "ArcFace", CVPR 2019).

Detection failures are recorded in a separate failure manifest and are **never**
silently replaced with center crops.  The smoke-test ``center_fit_112_bilinear``
in ``research.embeddings.smoke_inputs`` is checkpoint-validation-only and must
not be confused with this quantitative materializer.

References
----------
- Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
  Recognition", CVPR 2019.
- Deng et al., "RetinaFace: Single-shot Multi-level Face Localisation in the
  Wild", CVPR 2020.  (buffalo_l detection backbone)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from research.runtime.hashing import sha256_file

logger = logging.getLogger(__name__)

MATERIALIZER_VERSION = "1.0.0"
ALIGNMENT_TEMPLATE_ID = "insightface_arcface_112_v1"
DETECTOR_NAME = "buffalo_l"


@dataclass
class AlignmentResult:
    """Immutable bundle of aligned crop materialisation outputs."""

    aligned_faces: np.ndarray
    aligned_manifest: pd.DataFrame
    failed_manifest: pd.DataFrame
    bundle_metadata: dict[str, Any]
    output_dir: Path


def _compute_face_sha256(face: np.ndarray) -> str:
    """SHA-256 of the raw bytes of a single 112×112×3 uint8 array."""
    return hashlib.sha256(np.ascontiguousarray(face).tobytes()).hexdigest()


def _select_best_face(faces: list) -> tuple[int, Any]:
    """Select the face with highest detection score from a list.

    Returns (index, face_object).
    """
    if len(faces) == 1:
        return 0, faces[0]
    scored = [
        (i, f) for i, f in enumerate(faces) if hasattr(f, "det_score")
    ]
    if not scored:
        return 0, faces[0]
    scored.sort(key=lambda pair: pair[1].det_score, reverse=True)
    return scored[0]


def materialize_aligned_crops(
    manifest: pd.DataFrame,
    *,
    project_root: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    detector_name: str = DETECTOR_NAME,
    detection_size: tuple[int, int] = (640, 640),
    providers: tuple[str, ...] = ("CPUExecutionProvider",),
    overwrite: bool = False,
) -> AlignmentResult:
    """Materialize aligned 112×112 RGB uint8 crops for all manifest rows.

    Parameters
    ----------
    manifest : pd.DataFrame
        Source manifest with columns ``image_id``, ``identity_id``, ``split``,
        ``image_path``.
    project_root : Path
        Project root for resolving relative image paths.
    output_dir : Path
        Directory to write aligned_faces.npy, manifests, and bundle metadata.
    dataset_id : str
        Dataset identifier (e.g. "lfw").
    detector_name : str
        InsightFace model name for FaceAnalysis.
    detection_size : tuple[int, int]
        Detection input resolution.
    providers : tuple[str, ...]
        ONNX Runtime execution providers.
    overwrite : bool
        If False, refuse to overwrite existing outputs.

    Returns
    -------
    AlignmentResult
    """
    # Late imports to avoid hard dependency at module level
    from insightface.app import FaceAnalysis
    from insightface.utils.face_align import norm_crop

    root = Path(project_root).resolve()
    out = Path(output_dir).resolve()

    # --- Validate inputs ---
    required_columns = {"image_id", "identity_id", "split", "image_path"}
    missing_cols = sorted(required_columns - set(manifest.columns))
    if missing_cols:
        raise ValueError(f"manifest에 필수 열이 없습니다: {missing_cols}")
    if manifest.empty:
        raise ValueError("manifest가 비어 있습니다.")
    if not dataset_id or not dataset_id.strip():
        raise ValueError("dataset_id가 비어 있습니다.")

    # --- Check output directory ---
    success_marker = out / "_SUCCESS"
    if not overwrite:
        for name in (
            "aligned_faces.npy",
            "aligned_manifest.parquet",
            "bundle_manifest.json",
        ):
            candidate = out / name
            if candidate.exists():
                raise FileExistsError(
                    f"기존 aligned crop artifact를 덮어쓸 수 없습니다: {candidate}"
                )
    out.mkdir(parents=True, exist_ok=True)

    # --- Initialize detector ---
    logger.info(
        "InsightFace FaceAnalysis 초기화: model=%s, providers=%s, det_size=%s",
        detector_name,
        providers,
        detection_size,
    )
    app = FaceAnalysis(name=detector_name, providers=list(providers))
    app.prepare(ctx_id=0, det_size=detection_size)

    # --- Process each image ---
    aligned_faces_list: list[np.ndarray] = []
    aligned_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    face_index = 0

    total = len(manifest)
    log_interval = max(1, total // 20)

    for row_idx, row in enumerate(manifest.itertuples(index=False)):
        image_id = str(row.image_id)
        identity_id = str(row.identity_id)
        split = str(row.split)
        image_path_raw = str(row.image_path)

        # Resolve path
        image_path = Path(image_path_raw)
        if not image_path.is_absolute():
            image_path = root / image_path
        image_path = image_path.resolve()

        if (row_idx + 1) % log_interval == 0 or row_idx == 0:
            logger.info(
                "처리 중: %d/%d (%.1f%%)",
                row_idx + 1,
                total,
                100.0 * (row_idx + 1) / total,
            )

        # Common metadata
        base_row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "sample_id": image_id,
            "identity_id": identity_id,
            "split": split,
            "source_image_path": str(image_path),
        }

        # Check file exists
        if not image_path.is_file():
            base_row.update(
                {
                    "alignment_status": "failed",
                    "alignment_failure_reason": "source_file_not_found",
                    "source_content_sha256": "",
                }
            )
            failed_rows.append(base_row)
            logger.warning("이미지 파일 없음: %s", image_path)
            continue

        source_sha256 = sha256_file(image_path)
        base_row["source_content_sha256"] = source_sha256

        # Load image as RGB numpy array
        try:
            with Image.open(image_path) as img:
                rgb_image = np.asarray(img.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            base_row.update(
                {
                    "alignment_status": "failed",
                    "alignment_failure_reason": f"image_load_error: {exc}",
                }
            )
            failed_rows.append(base_row)
            logger.warning("이미지 로딩 실패: %s — %s", image_path, exc)
            continue

        # Detect faces (InsightFace expects BGR)
        bgr_image = rgb_image[..., ::-1].copy()
        try:
            faces = app.get(bgr_image)
        except Exception as exc:
            base_row.update(
                {
                    "alignment_status": "failed",
                    "alignment_failure_reason": f"detection_error: {exc}",
                }
            )
            failed_rows.append(base_row)
            logger.warning("얼굴 검출 오류: %s — %s", image_path, exc)
            continue

        if not faces:
            base_row.update(
                {
                    "alignment_status": "failed",
                    "alignment_failure_reason": "no_face_detected",
                    "face_count": 0,
                }
            )
            failed_rows.append(base_row)
            logger.debug("얼굴 미검출: %s", image_path)
            continue

        # Select best face
        selected_idx, selected_face = _select_best_face(faces)
        landmark = selected_face.kps  # shape (5, 2)
        bbox = selected_face.bbox  # shape (4,)
        det_score = float(selected_face.det_score)

        # Perform ArcFace standard alignment via norm_crop
        try:
            aligned = norm_crop(rgb_image, landmark, image_size=112)
        except Exception as exc:
            base_row.update(
                {
                    "alignment_status": "failed",
                    "alignment_failure_reason": f"alignment_error: {exc}",
                    "face_count": len(faces),
                    "selected_face_index": selected_idx,
                    "detection_score": det_score,
                }
            )
            failed_rows.append(base_row)
            logger.warning("정렬 변환 실패: %s — %s", image_path, exc)
            continue

        # Ensure RGB uint8
        aligned_rgb = np.asarray(aligned, dtype=np.uint8)
        if aligned_rgb.shape != (112, 112, 3):
            base_row.update(
                {
                    "alignment_status": "failed",
                    "alignment_failure_reason": (
                        f"unexpected_shape: {aligned_rgb.shape}"
                    ),
                    "face_count": len(faces),
                }
            )
            failed_rows.append(base_row)
            continue

        content_sha256 = _compute_face_sha256(aligned_rgb)

        # Store
        aligned_faces_list.append(aligned_rgb)
        landmark_flat = landmark.flatten().tolist()
        base_row.update(
            {
                "alignment_status": "aligned",
                "alignment_failure_reason": "",
                "face_count": len(faces),
                "selected_face_index": selected_idx,
                "detection_score": det_score,
                "bbox_x1": float(bbox[0]),
                "bbox_y1": float(bbox[1]),
                "bbox_x2": float(bbox[2]),
                "bbox_y2": float(bbox[3]),
                "landmark_5points": json.dumps(landmark_flat),
                "aligned_face_index": face_index,
                "aligned_content_sha256": content_sha256,
                "detector_name": detector_name,
                "alignment_template_id": ALIGNMENT_TEMPLATE_ID,
                "materializer_version": MATERIALIZER_VERSION,
            }
        )
        aligned_rows.append(base_row)
        face_index += 1

    # --- Build outputs ---
    if not aligned_faces_list:
        raise RuntimeError(
            "정렬에 성공한 이미지가 없습니다. 얼굴 검출기 설정을 확인하세요."
        )

    aligned_array = np.stack(aligned_faces_list, axis=0).astype(
        np.uint8, copy=False
    )
    assert aligned_array.shape == (len(aligned_faces_list), 112, 112, 3)

    aligned_df = pd.DataFrame(aligned_rows)
    failed_df = pd.DataFrame(failed_rows) if failed_rows else pd.DataFrame()

    # --- Write outputs ---
    npy_path = out / "aligned_faces.npy"
    manifest_path = out / "aligned_manifest.parquet"
    failed_path = out / "failed_samples.parquet"
    bundle_path = out / "bundle_manifest.json"

    logger.info("NPY 저장: %s (%s)", npy_path, aligned_array.shape)
    np.save(npy_path, aligned_array, allow_pickle=False)

    logger.info("aligned manifest 저장: %s (%d행)", manifest_path, len(aligned_df))
    aligned_df.to_parquet(manifest_path, index=False)

    if not failed_df.empty:
        logger.info("failed manifest 저장: %s (%d행)", failed_path, len(failed_df))
        failed_df.to_parquet(failed_path, index=False)
    else:
        logger.info("실패한 표본이 없습니다.")

    # Bundle metadata
    bundle_metadata: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "materializer_version": MATERIALIZER_VERSION,
        "alignment_template_id": ALIGNMENT_TEMPLATE_ID,
        "detector_name": detector_name,
        "detection_size": list(detection_size),
        "providers": list(providers),
        "image_size": [112, 112],
        "source_color_order": "rgb",
        "dtype": "uint8",
        "layout": "nhwc",
        "total_source_images": total,
        "aligned_count": len(aligned_faces_list),
        "failed_count": len(failed_rows),
        "alignment_success_rate": len(aligned_faces_list) / total if total > 0 else 0.0,
        "outputs": {
            "aligned_faces_npy": {
                "path": str(npy_path),
                "sha256": sha256_file(npy_path),
                "shape": list(aligned_array.shape),
            },
            "aligned_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "row_count": len(aligned_df),
            },
            "failed_manifest": (
                {
                    "path": str(failed_path),
                    "sha256": sha256_file(failed_path),
                    "row_count": len(failed_df),
                }
                if not failed_df.empty
                else None
            ),
        },
    }

    bundle_path.write_text(
        json.dumps(bundle_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("bundle manifest 저장: %s", bundle_path)

    # Success marker
    success_marker.write_text(
        f"aligned_count={len(aligned_faces_list)}\n"
        f"failed_count={len(failed_rows)}\n"
        f"total={total}\n",
        encoding="utf-8",
    )
    logger.info(
        "완료: %d/%d 정렬 성공 (%.1f%%), %d 실패",
        len(aligned_faces_list),
        total,
        100.0 * len(aligned_faces_list) / total if total > 0 else 0.0,
        len(failed_rows),
    )

    return AlignmentResult(
        aligned_faces=aligned_array,
        aligned_manifest=aligned_df,
        failed_manifest=failed_df,
        bundle_metadata=bundle_metadata,
        output_dir=out,
    )
