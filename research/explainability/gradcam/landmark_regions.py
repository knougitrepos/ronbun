from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from research.runtime.hashing import canonical_sha256, sha256_file


ProgressCallback = Callable[[str, dict[str, object]], None]


def _emit(progress: ProgressCallback | None, message: str, **details: object) -> None:
    if progress is not None:
        progress(message, details)


LANDMARK_MODEL_FILENAME = "2d106det.onnx"
LANDMARK_TOPOLOGY_ID = "insightface-2d106-semantic-v1"
LANDMARK_MATERIALIZER_VERSION = "1.0.0"
DEFAULT_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")
ARC_FACE_112_LANDMARKS = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
REGION_NAMES = (
    "left_eye",
    "right_eye",
    "nose",
    "mouth",
    "left_cheek",
    "right_cheek",
    "jaw",
    "face",
    "outside_face",
)

# InsightFace 2d106 output order, recorded as an explicit artifact contract.
# Left/right use image coordinates: "left" has the smaller x coordinate.
LANDMARK_GROUPS: Mapping[str, tuple[int, ...]] = {
    "face_contour": tuple(range(0, 33)),
    "left_eye": tuple(range(33, 43)),
    "left_eyebrow": tuple(range(43, 52)),
    "mouth": tuple(range(52, 72)),
    "nose": tuple(range(72, 87)),
    "right_eye": tuple(range(87, 97)),
    "right_eyebrow": tuple(range(97, 106)),
}


def _publish_directory(staging: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        os.replace(staging, destination)
        return
    if not overwrite:
        raise FileExistsError(f"landmark-region bundle already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.backup-{uuid4().hex}")
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    else:
        shutil.rmtree(backup)


def _output_entry(path: Path, root: Path, **extra: object) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        **extra,
    }


def _validate_landmarks(
    landmarks_5: np.ndarray,
    landmarks_106: np.ndarray,
    *,
    source_image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    five = np.asarray(landmarks_5, dtype=np.float32)
    dense = np.asarray(landmarks_106, dtype=np.float32)
    if five.ndim == 2:
        five = five[np.newaxis, ...]
    if dense.ndim == 2:
        dense = dense[np.newaxis, ...]
    if five.ndim != 3 or five.shape[1:] != (5, 2):
        raise ValueError("landmarks_5 must have shape [N, 5, 2]")
    if dense.ndim != 3 or dense.shape[1:] != (106, 2):
        raise ValueError("landmarks_106 must have shape [N, 106, 2]")
    if len(five) == 0 or len(five) != len(dense):
        raise ValueError("landmark arrays must be non-empty and row aligned")
    if not np.isfinite(five).all() or not np.isfinite(dense).all():
        raise ValueError("landmark arrays must contain only finite values")
    height, width = (int(source_image_size[0]), int(source_image_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("source_image_size values must be positive")
    margin_x = width * 0.25
    margin_y = height * 0.25
    if (
        np.any(dense[..., 0] < -margin_x)
        or np.any(dense[..., 0] > width - 1 + margin_x)
        or np.any(dense[..., 1] < -margin_y)
        or np.any(dense[..., 1] > height - 1 + margin_y)
    ):
        raise ValueError("dense landmarks fall implausibly far outside the crop")
    return five, dense


def _fill_hull(canvas: np.ndarray, points: np.ndarray) -> None:
    import cv2

    polygon = np.rint(points).astype(np.int32)
    hull = cv2.convexHull(polygon)
    cv2.fillConvexPoly(canvas, hull, color=1)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    import cv2

    size = 2 * max(1, int(radius)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def build_insightface_106_region_masks(
    landmarks_5: np.ndarray,
    landmarks_106: np.ndarray,
    *,
    image_size: tuple[int, int] = (112, 112),
    source_image_size: tuple[int, int] = (112, 112),
) -> dict[str, np.ndarray]:
    """Build deterministic face-part masks from the frozen InsightFace topology.

    Dense landmarks define the face outline and feature hulls. The aligned
    five-point anchors define cheek centers and scale. Masks are generated at
    the requested resolution, so Grad-CAM can request native low-resolution
    masks without materializing full 112x112 masks for an entire dataset.
    """

    five, dense = _validate_landmarks(
        landmarks_5,
        landmarks_106,
        source_image_size=source_image_size,
    )
    target_height, target_width = (int(image_size[0]), int(image_size[1]))
    source_height, source_width = (
        int(source_image_size[0]),
        int(source_image_size[1]),
    )
    if target_height <= 0 or target_width <= 0:
        raise ValueError("image_size values must be positive")
    if (target_height, target_width) != (source_height, source_width):
        source_masks = build_insightface_106_region_masks(
            five,
            dense,
            image_size=(source_height, source_width),
            source_image_size=(source_height, source_width),
        )
        row_index = np.minimum(
            np.arange(target_height) * source_height // target_height,
            source_height - 1,
        )
        column_index = np.minimum(
            np.arange(target_width) * source_width // target_width,
            source_width - 1,
        )
        resized = {
            name: values[:, row_index[:, None], column_index[None, :]].copy()
            for name, values in source_masks.items()
        }
        resized["outside_face"] = ~resized["face"]
        for row in range(len(five)):
            face_pixels = np.argwhere(resized["face"][row])
            if len(face_pixels) == 0:
                raise ValueError(f"landmark row {row} lost its face mask when resized")
            for name in REGION_NAMES:
                if resized[name][row].any():
                    continue
                source_pixels = np.argwhere(source_masks[name][row])
                centroid = source_pixels.mean(axis=0)
                target = np.asarray(
                    [
                        centroid[0] * target_height / source_height,
                        centroid[1] * target_width / source_width,
                    ]
                )
                if name == "outside_face":
                    candidates = np.argwhere(~resized["face"][row])
                else:
                    candidates = face_pixels
                nearest = candidates[
                    np.argmin(np.sum((candidates - target[None, :]) ** 2, axis=1))
                ]
                resized[name][row, int(nearest[0]), int(nearest[1])] = True
        return resized
    scale = np.asarray(
        [target_width / source_width, target_height / source_height],
        dtype=np.float32,
    )
    scaled_five = five * scale
    scaled_dense = dense * scale
    scaled_dense[..., 0] = np.clip(scaled_dense[..., 0], 0, target_width - 1)
    scaled_dense[..., 1] = np.clip(scaled_dense[..., 1], 0, target_height - 1)

    result = {
        name: np.zeros((len(five), target_height, target_width), dtype=bool)
        for name in REGION_NAMES
    }
    import cv2

    for row in range(len(five)):
        anchors = scaled_five[row]
        points = scaled_dense[row]
        interocular = float(np.linalg.norm(anchors[1] - anchors[0]))
        if not np.isfinite(interocular) or interocular < 2.0:
            raise ValueError(f"landmark row {row} has an invalid interocular distance")

        face_u8 = np.zeros((target_height, target_width), dtype=np.uint8)
        contour = points[list(LANDMARK_GROUPS["face_contour"])]
        brows = points[
            [
                *LANDMARK_GROUPS["left_eyebrow"],
                *LANDMARK_GROUPS["right_eyebrow"],
            ]
        ]
        forehead = brows.copy()
        forehead[:, 1] -= 0.55 * interocular
        forehead[:, 1] = np.clip(forehead[:, 1], 0, target_height - 1)
        _fill_hull(face_u8, np.concatenate([contour, brows, forehead], axis=0))
        face = face_u8.astype(bool)

        feature_masks: dict[str, np.ndarray] = {}
        for name, radius_fraction in (
            ("left_eye", 0.07),
            ("right_eye", 0.07),
            ("nose", 0.06),
            ("mouth", 0.05),
        ):
            feature_u8 = np.zeros_like(face_u8)
            _fill_hull(feature_u8, points[list(LANDMARK_GROUPS[name])])
            feature = _dilate(
                feature_u8,
                max(1, round(interocular * radius_fraction)),
            )
            feature_masks[name] = feature & face
            result[name][row] = feature_masks[name]

        erode_radius = max(1, round(interocular * 0.13))
        kernel_size = 2 * erode_radius + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        inner_face = cv2.erode(face_u8, kernel).astype(bool)
        row_grid = np.arange(target_height, dtype=np.float32)[:, None]
        nose_y = float(anchors[2, 1])
        mouth_y = float((anchors[3, 1] + anchors[4, 1]) / 2.0)
        lower_boundary = (face & ~inner_face) & (
            row_grid >= nose_y + 0.25 * max(1.0, mouth_y - nose_y)
        )
        chin = face & (row_grid >= mouth_y + 0.22 * interocular)
        jaw = (lower_boundary | chin) & ~feature_masks["mouth"]
        result["jaw"][row] = jaw

        excluded = (
            feature_masks["left_eye"]
            | feature_masks["right_eye"]
            | feature_masks["nose"]
            | feature_masks["mouth"]
            | jaw
        )
        for name, eye_index in (("left_cheek", 0), ("right_cheek", 1)):
            center_x = float((anchors[eye_index, 0] + anchors[2, 0]) / 2.0)
            center_y = float((anchors[2, 1] + mouth_y) / 2.0 + 0.08 * interocular)
            cheek_u8 = np.zeros_like(face_u8)
            cv2.ellipse(
                cheek_u8,
                (round(center_x), round(center_y)),
                (
                    max(1, round(0.27 * interocular)),
                    max(1, round(0.32 * interocular)),
                ),
                0.0,
                0.0,
                360.0,
                color=1,
                thickness=-1,
            )
            cheek = cheek_u8.astype(bool) & face & ~excluded
            if name == "left_cheek":
                cheek &= np.arange(target_width)[None, :] <= anchors[2, 0]
            else:
                cheek &= np.arange(target_width)[None, :] >= anchors[2, 0]
            result[name][row] = cheek

        result["face"][row] = face
        result["outside_face"][row] = ~face

        for name in REGION_NAMES:
            if not result[name][row].any():
                raise ValueError(f"landmark row {row} produced an empty {name} mask")
        face_fraction = float(face.mean())
        if not 0.20 <= face_fraction <= 0.95:
            raise ValueError(
                f"landmark row {row} produced implausible face coverage "
                f"{face_fraction:.4f}"
            )
    return result


@dataclass(frozen=True)
class LandmarkRegionMaskProvider:
    region_mask_uid: str
    sample_ids: np.ndarray
    landmarks_5: np.ndarray
    landmarks_106: np.ndarray
    source_image_size: tuple[int, int] = (112, 112)
    row_map: np.ndarray | None = None

    def __post_init__(self) -> None:
        sample_ids = np.asarray(self.sample_ids).astype(str)
        five, dense = _validate_landmarks(
            self.landmarks_5,
            self.landmarks_106,
            source_image_size=self.source_image_size,
        )
        if len(sample_ids) != len(five):
            raise ValueError("sample_ids must align with landmark rows")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids must be unique")
        if not str(self.region_mask_uid).strip():
            raise ValueError("region_mask_uid must not be empty")
        if self.row_map is not None:
            row_map = np.asarray(self.row_map, dtype=np.int64)
            if row_map.ndim != 1 or len(row_map) == 0:
                raise ValueError("row_map must be a non-empty one-dimensional array")
            if row_map.min() < 0 or row_map.max() >= len(sample_ids):
                raise ValueError("row_map contains an out-of-range landmark row")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "landmarks_5", five)
        object.__setattr__(self, "landmarks_106", dense)

    @property
    def sample_count(self) -> int:
        return len(self.row_map) if self.row_map is not None else len(self.sample_ids)

    def subset(
        self,
        row_indices: Sequence[int] | np.ndarray,
        *,
        expected_sample_ids: Sequence[object] | np.ndarray | None = None,
    ) -> "LandmarkRegionMaskProvider":
        selected = np.asarray(row_indices, dtype=np.int64)
        if selected.ndim != 1 or len(selected) == 0:
            raise ValueError("row_indices must be a non-empty one-dimensional array")
        base = (
            np.arange(len(self.sample_ids), dtype=np.int64)
            if self.row_map is None
            else np.asarray(self.row_map, dtype=np.int64)
        )
        if selected.min() < 0 or selected.max() >= len(base):
            raise ValueError("row_indices contains an out-of-range provider row")
        mapped = base[selected]
        if expected_sample_ids is not None:
            expected = np.asarray(expected_sample_ids).astype(str)
            if not np.array_equal(self.sample_ids[mapped], expected):
                raise ValueError("selected landmark rows do not match expected sample_ids")
        return LandmarkRegionMaskProvider(
            region_mask_uid=self.region_mask_uid,
            sample_ids=self.sample_ids,
            landmarks_5=self.landmarks_5,
            landmarks_106=self.landmarks_106,
            source_image_size=self.source_image_size,
            row_map=mapped,
        )

    def build_region_masks(
        self,
        row_indices: Sequence[int] | np.ndarray,
        *,
        image_size: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        requested = np.asarray(row_indices, dtype=np.int64)
        base = (
            np.arange(len(self.sample_ids), dtype=np.int64)
            if self.row_map is None
            else np.asarray(self.row_map, dtype=np.int64)
        )
        if requested.ndim != 1 or len(requested) == 0:
            raise ValueError("row_indices must be a non-empty one-dimensional array")
        if requested.min() < 0 or requested.max() >= len(base):
            raise ValueError("row_indices contains an out-of-range provider row")
        mapped = base[requested]
        return build_insightface_106_region_masks(
            self.landmarks_5[mapped],
            self.landmarks_106[mapped],
            image_size=image_size,
            source_image_size=self.source_image_size,
        )


def _default_landmark_model(
    model_path: Path,
    providers: tuple[str, ...],
) -> Any:
    import onnxruntime as ort
    from insightface.model_zoo import get_model

    available = tuple(ort.get_available_providers())
    primary = providers[0]
    if primary not in available:
        raise RuntimeError(
            f"required primary ONNX Runtime provider is unavailable: "
            f"{primary}; available={available}"
        )
    model = get_model(str(model_path), providers=list(providers))
    active = tuple(model.session.get_providers())
    if not active or active[0] != primary:
        raise RuntimeError(
            "landmark model did not activate the required primary provider: "
            f"required={primary}, active={active}"
        )
    model.prepare(ctx_id=0)
    model._ronbun_session_providers = active
    return model


def _aligned_face_bbox(landmarks_5: np.ndarray) -> np.ndarray:
    interocular = float(np.linalg.norm(landmarks_5[1] - landmarks_5[0]))
    mouth_y = float((landmarks_5[3, 1] + landmarks_5[4, 1]) / 2.0)
    return np.asarray(
        [
            min(landmarks_5[0, 0], landmarks_5[3, 0]) - 0.75 * interocular,
            min(landmarks_5[0, 1], landmarks_5[1, 1]) - 1.05 * interocular,
            max(landmarks_5[1, 0], landmarks_5[4, 0]) + 0.75 * interocular,
            mouth_y + 0.75 * interocular,
        ],
        dtype=np.float32,
    )


def materialize_landmark_region_bundle(
    aligned_bundle_dir: str | Path,
    *,
    output_dir: str | Path,
    dataset_id: str,
    model_path: str | Path | None = None,
    providers: tuple[str, ...] = DEFAULT_PROVIDERS,
    overwrite: bool = False,
    landmark_model: Any | None = None,
    progress: ProgressCallback | None = None,
) -> LandmarkRegionMaskProvider:
    """Extract 106-point landmarks and publish a deterministic mask provider."""

    aligned_root = Path(aligned_bundle_dir).resolve()
    if not (aligned_root / "_SUCCESS").is_file():
        raise RuntimeError(f"aligned-crop bundle is incomplete: {aligned_root}")
    faces_path = aligned_root / "aligned_faces.npy"
    index_path = aligned_root / "aligned_index.csv"
    manifest_path = aligned_root / "bundle_manifest.json"
    for path in (faces_path, index_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    aligned_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(aligned_manifest.get("dataset_id")) != str(dataset_id):
        raise ValueError("aligned-crop bundle dataset_id does not match requested dataset")
    faces = np.load(faces_path, mmap_mode="r", allow_pickle=False)
    aligned_index = pd.read_csv(index_path)
    if (
        faces.ndim != 4
        or faces.shape[1:] != (112, 112, 3)
        or faces.dtype != np.uint8
    ):
        raise ValueError("aligned faces must be RGB uint8 NHWC 112x112")
    required_index = {"sample_id", "identity_id", "split", "aligned_face_index"}
    missing = sorted(required_index.difference(aligned_index.columns))
    if missing:
        raise ValueError(f"aligned index is missing columns: {missing}")
    expected_indices = np.arange(len(aligned_index), dtype=np.int64)
    if len(faces) != len(aligned_index) or not np.array_equal(
        aligned_index["aligned_face_index"].to_numpy(dtype=np.int64),
        expected_indices,
    ):
        raise ValueError("aligned faces and aligned index row order differ")
    sample_ids = aligned_index["sample_id"].astype(str).to_numpy()
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("aligned index sample_id must be unique")

    requested_providers = tuple(str(value).strip() for value in providers)
    if (
        not requested_providers
        or any(not value for value in requested_providers)
        or len(set(requested_providers)) != len(requested_providers)
    ):
        raise ValueError("providers must contain unique non-empty provider names")
    resolved_model = (
        Path(model_path).expanduser().resolve()
        if model_path is not None
        else (
            Path.home()
            / ".insightface"
            / "models"
            / "buffalo_l"
            / LANDMARK_MODEL_FILENAME
        ).resolve()
    )
    if landmark_model is None and not resolved_model.is_file():
        raise FileNotFoundError(
            f"InsightFace 2d106 landmark model not found: {resolved_model}"
        )
    active_model = landmark_model or _default_landmark_model(
        resolved_model,
        requested_providers,
    )
    active_providers = tuple(
        getattr(active_model, "_ronbun_session_providers", ("injected_model",))
    )

    destination = Path(output_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"landmark-region bundle already exists: {destination}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        five_path = staging / "aligned_landmarks_5.npy"
        dense_path = staging / "aligned_landmarks_106.npy"
        five_output = np.lib.format.open_memmap(
            five_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(faces), 5, 2),
        )
        dense_output = np.lib.format.open_memmap(
            dense_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(faces), 106, 2),
        )
        coverage = {
            f"{name}_pixel_count": np.zeros(len(faces), dtype=np.int32)
            for name in REGION_NAMES
        }
        from insightface.app.common import Face

        for row in range(len(faces)):
            aligned_five = ARC_FACE_112_LANDMARKS.copy()
            face = Face(
                bbox=_aligned_face_bbox(aligned_five),
                kps=aligned_five.copy(),
            )
            predicted = np.asarray(
                active_model.get(faces[row][..., ::-1].copy(), face),
                dtype=np.float32,
            )
            if predicted.shape != (106, 2) or not np.isfinite(predicted).all():
                raise RuntimeError(
                    f"invalid 2d106 landmark output for sample {sample_ids[row]}"
                )
            five_output[row] = aligned_five
            dense_output[row] = predicted
            masks = build_insightface_106_region_masks(
                aligned_five,
                predicted,
            )
            for name, values in masks.items():
                coverage[f"{name}_pixel_count"][row] = int(values[0].sum())
            _emit(
                progress,
                "106-point landmark materialization",
                processed=row + 1,
                total=len(faces),
            )
        five_output.flush()
        dense_output.flush()
        del five_output
        del dense_output

        model_sha256 = (
            sha256_file(resolved_model)
            if resolved_model.is_file()
            else str(getattr(active_model, "_ronbun_model_sha256", "injected"))
        )
        uid_payload = {
            "artifact_type": "landmark_region_mask_provider",
            "dataset_id": str(dataset_id),
            "topology_id": LANDMARK_TOPOLOGY_ID,
            "materializer_version": LANDMARK_MATERIALIZER_VERSION,
            "model_sha256": model_sha256,
            "aligned_faces_sha256": sha256_file(faces_path),
            "aligned_index_sha256": sha256_file(index_path),
            "landmarks_5_sha256": sha256_file(five_path),
            "landmarks_106_sha256": sha256_file(dense_path),
        }
        region_mask_uid = (
            f"landmark-regions-{canonical_sha256(uid_payload)[:24]}"
        )
        mask_index = aligned_index[
            ["sample_id", "identity_id", "split", "aligned_face_index"]
        ].copy()
        mask_index.insert(0, "dataset_id", str(dataset_id))
        mask_index["region_mask_uid"] = region_mask_uid
        for column, values in coverage.items():
            mask_index[column] = values
        mask_index_path = staging / "mask_index.csv"
        mask_index.to_csv(mask_index_path, index=False, encoding="utf-8")

        bundle_manifest = {
            "schema_version": 1,
            **uid_payload,
            "region_mask_uid": region_mask_uid,
            "coordinate_convention": {
                "source_layout": "aligned_crop_xy",
                "left_right": "image_coordinates",
                "source_image_size": [112, 112],
            },
            "landmark_groups": {
                name: list(indices) for name, indices in LANDMARK_GROUPS.items()
            },
            "regions": list(REGION_NAMES),
            "provider": {
                "requested": list(requested_providers),
                "active": list(active_providers),
            },
            "row_count": int(len(faces)),
            "outputs": {
                "aligned_landmarks_5": _output_entry(
                    five_path,
                    staging,
                    shape=[len(faces), 5, 2],
                    dtype="float32",
                ),
                "aligned_landmarks_106": _output_entry(
                    dense_path,
                    staging,
                    shape=[len(faces), 106, 2],
                    dtype="float32",
                ),
                "mask_index": _output_entry(
                    mask_index_path,
                    staging,
                    row_count=len(mask_index),
                ),
            },
        }
        bundle_manifest_path = staging / "bundle_manifest.json"
        bundle_manifest_path.write_text(
            json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "_SUCCESS").write_text("complete\n", encoding="utf-8")
        _publish_directory(staging, destination, overwrite=overwrite)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return read_landmark_region_bundle(destination)


def read_landmark_region_bundle(
    bundle_dir: str | Path,
) -> LandmarkRegionMaskProvider:
    root = Path(bundle_dir).resolve()
    if not (root / "_SUCCESS").is_file():
        raise RuntimeError(f"landmark-region bundle is incomplete: {root}")
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("topology_id") != LANDMARK_TOPOLOGY_ID:
        raise ValueError("landmark-region topology_id is unsupported")
    outputs = manifest.get("outputs", {})
    for entry in outputs.values():
        path = root / str(entry["path"])
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"landmark-region output checksum mismatch: {path}")
    index = pd.read_csv(root / str(outputs["mask_index"]["path"]))
    if len(index) != int(manifest["row_count"]):
        raise ValueError("landmark-region index row count differs from manifest")
    five = np.load(
        root / str(outputs["aligned_landmarks_5"]["path"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    dense = np.load(
        root / str(outputs["aligned_landmarks_106"]["path"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    return LandmarkRegionMaskProvider(
        region_mask_uid=str(manifest["region_mask_uid"]),
        sample_ids=index["sample_id"].astype(str).to_numpy(),
        landmarks_5=five,
        landmarks_106=dense,
        source_image_size=tuple(
            manifest["coordinate_convention"]["source_image_size"]
        ),
    )
