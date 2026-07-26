"""Deterministic smoke-test inputs for FR checkpoint validation.

These inputs validate checkpoint loading, preprocessing, output shape, raw
feature norm, and target-layer resolution. They are not quantitative research
artifacts and must not be reused as the Step 2 aligned-crop population.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from research.embeddings.base import ColorOrder
from research.runtime.hashing import sha256_file


@dataclass(frozen=True)
class SmokeInputBatch:
    aligned_faces: np.ndarray
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        faces = np.asarray(self.aligned_faces)
        if (
            faces.ndim != 4
            or faces.shape[0] == 0
            or faces.shape[1:] != (112, 112, 3)
            or faces.dtype != np.uint8
        ):
            raise ValueError(
                "smoke faces must be non-empty uint8 [N,112,112,3]"
            )
        object.__setattr__(self, "aligned_faces", faces)


def _bounded_faces(faces: np.ndarray, max_images: int) -> np.ndarray:
    array = np.asarray(faces)
    if (
        array.ndim != 4
        or array.shape[0] == 0
        or array.shape[1:] != (112, 112, 3)
        or array.dtype != np.uint8
    ):
        raise ValueError(
            "smoke input must contain non-empty uint8 [N,112,112,3]"
        )
    return np.asarray(array[:max_images], dtype=np.uint8)


def _load_explicit_bundle(path: Path, max_images: int) -> SmokeInputBatch:
    if not path.is_file():
        raise FileNotFoundError(f"smoke input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        faces = np.load(path, mmap_mode="r", allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as bundle:
            if "aligned_faces" not in bundle.files:
                raise ValueError(
                    "smoke .npz must contain an 'aligned_faces' array"
                )
            faces = np.asarray(bundle["aligned_faces"])
    else:
        raise ValueError("explicit smoke input must be .npy or .npz")
    selected = _bounded_faces(faces, max_images)
    return SmokeInputBatch(
        aligned_faces=selected,
        metadata={
            "source_type": "explicit_aligned_array",
            "source_path": str(path),
            "source_sha256": sha256_file(path),
            "sample_count": int(len(selected)),
            "smoke_only": True,
            "quantitative_experiment_input": False,
        },
    )


def _manifest_rows(
    project_root: Path,
    manifest_path: Path,
) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "LFW smoke fallback manifest does not exist. Run "
            "notebooks/lfw/00_data_preparation/00_data_preparation.ipynb first or set "
            f"SMOKE_INPUT_PATH explicitly: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "identity_id", "image_path"}
    if not rows:
        raise ValueError(f"smoke fallback manifest is empty: {manifest_path}")
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"smoke fallback manifest missing columns: {missing}")

    resolved: list[dict[str, str]] = []
    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = project_root / image_path
        image_path = image_path.resolve()
        if image_path.is_file():
            resolved.append(
                {
                    "image_id": str(row["image_id"]),
                    "identity_id": str(row["identity_id"]),
                    "image_path": str(image_path),
                }
            )
    if not resolved:
        raise FileNotFoundError(
            "no image referenced by the LFW manifest exists locally"
        )
    return resolved


def _stable_diverse_rows(
    rows: list[dict[str, str]],
    *,
    max_images: int,
    seed: int,
) -> list[dict[str, str]]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['image_id']}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[dict[str, str]] = []
    seen_identities: set[str] = set()
    for row in ordered:
        if row["identity_id"] in seen_identities:
            continue
        selected.append(row)
        seen_identities.add(row["identity_id"])
        if len(selected) == max_images:
            return selected
    for row in ordered:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) == max_images:
            break
    return selected


def _load_lfw_smoke_faces(
    project_root: Path,
    manifest_path: Path,
    *,
    max_images: int,
    seed: int,
    source_color_order: ColorOrder,
) -> SmokeInputBatch:
    rows = _stable_diverse_rows(
        _manifest_rows(project_root, manifest_path),
        max_images=max_images,
        seed=seed,
    )
    faces: list[np.ndarray] = []
    source_files: list[dict[str, str]] = []
    for row in rows:
        image_path = Path(row["image_path"])
        with Image.open(image_path) as image:
            rgb = ImageOps.fit(
                image.convert("RGB"),
                (112, 112),
                method=Image.Resampling.BILINEAR,
                centering=(0.5, 0.5),
            )
            values = np.asarray(rgb, dtype=np.uint8)
        if source_color_order == "bgr":
            values = values[..., ::-1].copy()
        faces.append(values)
        source_files.append(
            {
                "image_id": row["image_id"],
                "identity_id": row["identity_id"],
                "path": str(image_path),
                "sha256": sha256_file(image_path),
            }
        )
    return SmokeInputBatch(
        aligned_faces=np.stack(faces).astype(np.uint8, copy=False),
        metadata={
            "source_type": "lfw_deepfunneled_manifest_fallback",
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "source_color_order": source_color_order,
            "resize_policy": "center_fit_112_bilinear",
            "selection_policy": "seeded_identity_diverse",
            "seed": int(seed),
            "sample_count": int(len(faces)),
            "source_files": source_files,
            "smoke_only": True,
            "quantitative_experiment_input": False,
            "warning": (
                "This deterministic LFW fallback is only for model smoke "
                "validation, not for Step 2 quantitative embeddings or Grad-CAM."
            ),
        },
    )


def resolve_smoke_input_batch(
    project_root: str | Path,
    *,
    source_color_order: ColorOrder,
    explicit_path: str | Path | None = None,
    lfw_manifest_path: str | Path | None = None,
    max_images: int = 8,
    seed: int = 42,
) -> SmokeInputBatch:
    """Resolve explicit aligned arrays or build an automatic LFW smoke batch."""

    if max_images <= 0:
        raise ValueError("max_images must be positive")
    if source_color_order not in {"rgb", "bgr"}:
        raise ValueError("source_color_order must be 'rgb' or 'bgr'")
    root = Path(project_root).expanduser().resolve()
    if explicit_path is not None:
        return _load_explicit_bundle(
            Path(explicit_path).expanduser().resolve(),
            max_images,
        )
    manifest = (
        root / "data/interim/lfw/face_manifest.csv"
        if lfw_manifest_path is None
        else Path(lfw_manifest_path).expanduser()
    )
    if not manifest.is_absolute():
        manifest = root / manifest
    return _load_lfw_smoke_faces(
        root,
        manifest.resolve(),
        max_images=max_images,
        seed=seed,
        source_color_order=source_color_order,
    )
