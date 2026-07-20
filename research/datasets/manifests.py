from __future__ import annotations

import json
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from research.protocols import validate_identity_disjoint_splits

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MANIFEST_COLUMNS = ("image_id", "identity_id", "split", "image_path")


@dataclass(frozen=True)
class ManifestBundle:
    """A generic manifest and the identity lists needed by notebook 00."""

    manifest: pd.DataFrame
    gallery_identities: tuple[str, ...]
    unknown_unknown_identities: tuple[str, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class SurvFaceOfficialBundle:
    """QMUL-SurvFace official open-set identification rows and roles."""

    manifest: pd.DataFrame
    gallery_identities: tuple[str, ...]
    unknown_unknown_identities: tuple[str, ...]
    summary: dict[str, Any]

    def _rows_for(self, role: str) -> pd.DataFrame:
        return self.manifest.loc[self.manifest["protocol_role"] == role].reset_index(
            drop=True
        )

    @property
    def gallery(self) -> pd.DataFrame:
        return self._rows_for("gallery")

    @property
    def registered_probes(self) -> pd.DataFrame:
        return self._rows_for("registered_probe")

    @property
    def unknown_unknown_probes(self) -> pd.DataFrame:
        return self._rows_for("unknown_unknown_probe")


@dataclass(frozen=True)
class SurvFaceTrainingBundle:
    """Identity-disjoint development/calibration rows from SurvFace training_set."""

    manifest: pd.DataFrame
    summary: dict[str, Any]


def _resolved_directory(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {resolved}")
    return resolved


def _relative_posix(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"dataset file must be inside project_root: {path.resolve()} not under {project_root}"
        ) from exc


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _validate_manifest(manifest: pd.DataFrame) -> None:
    missing_columns = set(MANIFEST_COLUMNS).difference(manifest.columns)
    if missing_columns:
        raise ValueError(f"missing manifest columns: {sorted(missing_columns)}")
    if manifest.empty:
        raise ValueError("manifest must not be empty")
    if manifest[list(MANIFEST_COLUMNS)].isna().any().any():
        raise ValueError("required manifest columns must not contain missing values")
    if manifest["image_id"].duplicated().any():
        raise ValueError("image_id values must be unique")
    if manifest["image_path"].duplicated().any():
        raise ValueError("image_path values must be unique")


def _split_counts(total: int, development_fraction: float, calibration_fraction: float) -> tuple[int, int]:
    if total < 3:
        raise ValueError("at least three identities are required")
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between 0 and 1")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    if development_fraction + calibration_fraction >= 1.0:
        raise ValueError("development and calibration fractions must leave a test split")

    development_count = max(1, int(total * development_fraction))
    calibration_count = max(1, int(total * calibration_fraction))
    if development_count + calibration_count >= total:
        raise ValueError("split fractions leave no test identities")
    return development_count, calibration_count


def build_lfw_manifest(
    lfw_root: str | Path,
    project_root: str | Path,
    *,
    seed: int = 42,
    development_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
    gallery_size: int = 50,
    min_gallery_images: int = 6,
    unknown_unknown_fraction: float = 0.50,
) -> ManifestBundle:
    """Build an identity-disjoint LFW manifest without writing any files.

    Gallery identities are selected only from the test split and must contain
    enough images to leave registered probes after enrollment. Test identities
    not selected for the gallery are deterministically divided into known and
    unknown unknown groups.
    """

    root = _resolved_directory(lfw_root, "LFW root")
    project = _resolved_directory(project_root, "project root")
    if gallery_size < 1:
        raise ValueError("gallery_size must be positive")
    if min_gallery_images < 2:
        raise ValueError("min_gallery_images must be at least 2")
    if not 0.0 < unknown_unknown_fraction < 1.0:
        raise ValueError("unknown_unknown_fraction must be between 0 and 1")

    identity_images: dict[str, list[Path]] = {}
    for identity_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        images = _image_files(identity_dir)
        if images:
            identity_images[identity_dir.name] = images
    if not identity_images:
        raise ValueError(f"no identity image directories found under {root}")

    identities = sorted(identity_images)
    random.Random(seed).shuffle(identities)
    development_count, calibration_count = _split_counts(
        len(identities), development_fraction, calibration_fraction
    )
    development_ids = identities[:development_count]
    calibration_end = development_count + calibration_count
    calibration_ids = identities[development_count:calibration_end]
    test_ids = identities[calibration_end:]

    split_by_identity = {
        **{identity: "development" for identity in development_ids},
        **{identity: "calibration" for identity in calibration_ids},
        **{identity: "test" for identity in test_ids},
    }

    eligible_gallery_ids = sorted(
        identity
        for identity in test_ids
        if len(identity_images[identity]) >= min_gallery_images
    )
    if len(eligible_gallery_ids) < gallery_size:
        raise ValueError(
            "not enough test identities satisfy min_gallery_images: "
            f"required={gallery_size}, available={len(eligible_gallery_ids)}, "
            f"min_gallery_images={min_gallery_images}"
        )
    gallery_rng = random.Random(seed + 1)
    gallery_ids = tuple(sorted(gallery_rng.sample(eligible_gallery_ids, gallery_size)))

    non_gallery_test_ids = sorted(set(test_ids).difference(gallery_ids))
    if len(non_gallery_test_ids) < 2:
        raise ValueError("at least two non-gallery test identities are required")
    unknown_count = round(len(non_gallery_test_ids) * unknown_unknown_fraction)
    unknown_count = min(max(1, unknown_count), len(non_gallery_test_ids) - 1)
    unknown_rng = random.Random(seed + 2)
    unknown_ids = tuple(sorted(unknown_rng.sample(non_gallery_test_ids, unknown_count)))
    known_unknown_ids = sorted(set(non_gallery_test_ids).difference(unknown_ids))

    rows: list[dict[str, str]] = []
    for identity in sorted(identity_images):
        identity_id = f"lfw:{identity}"
        for image_path in identity_images[identity]:
            rows.append(
                {
                    "image_id": f"lfw:{identity}:{image_path.stem}",
                    "identity_id": identity_id,
                    "split": split_by_identity[identity],
                    "image_path": _relative_posix(image_path, project),
                }
            )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values(
        ["split", "identity_id", "image_id"]
    ).reset_index(drop=True)
    _validate_manifest(manifest)
    validate_identity_disjoint_splits(manifest)

    prefixed_gallery_ids = tuple(f"lfw:{identity}" for identity in gallery_ids)
    prefixed_unknown_ids = tuple(f"lfw:{identity}" for identity in unknown_ids)
    summary = {
        "dataset": "lfw-deepfunneled",
        "seed": int(seed),
        "image_count": int(len(manifest)),
        "identity_count": int(manifest["identity_id"].nunique()),
        "development_identity_count": int(len(development_ids)),
        "calibration_identity_count": int(len(calibration_ids)),
        "test_identity_count": int(len(test_ids)),
        "gallery_identity_count": int(len(prefixed_gallery_ids)),
        "known_unknown_identity_count": int(len(known_unknown_ids)),
        "unknown_unknown_identity_count": int(len(prefixed_unknown_ids)),
        "min_gallery_images": int(min_gallery_images),
    }
    return ManifestBundle(
        manifest=manifest,
        gallery_identities=prefixed_gallery_ids,
        unknown_unknown_identities=prefixed_unknown_ids,
        summary=summary,
    )


def _mat_string(value: Any) -> str:
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    text = str(value).strip()
    if not text:
        raise ValueError("empty image filename in MAT file")
    return text


def _load_survface_pairs(
    mat_path: Path,
    *,
    names_key: str,
    ids_key: str,
) -> tuple[list[str], list[int]]:
    if not mat_path.is_file():
        raise FileNotFoundError(f"SurvFace protocol MAT file not found: {mat_path}")
    data = loadmat(mat_path)
    if names_key not in data or ids_key not in data:
        raise ValueError(
            f"MAT file {mat_path.name} must contain {names_key!r} and {ids_key!r}"
        )
    names = [_mat_string(value) for value in np.asarray(data[names_key]).reshape(-1)]
    ids = [int(value) for value in np.asarray(data[ids_key]).reshape(-1)]
    if len(names) != len(ids):
        raise ValueError(f"filename/identity length mismatch in {mat_path.name}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate filenames in {mat_path.name}")
    unsafe_names = [name for name in names if Path(name).name != name]
    if unsafe_names:
        raise ValueError(f"nested or unsafe filenames in {mat_path.name}: {unsafe_names[:3]}")
    return names, ids


def _require_exact_file_set(directory: Path, expected_names: list[str], label: str) -> None:
    actual_names = {path.name for path in _image_files(directory)}
    expected_set = set(expected_names)
    missing = sorted(expected_set.difference(actual_names))
    extra = sorted(actual_names.difference(expected_set))
    if missing or extra:
        raise ValueError(
            f"{label} files do not match the official protocol: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def _survface_frame(
    *,
    names: list[str],
    ids: list[int] | None,
    directory: Path,
    project_root: Path,
    role: str,
) -> pd.DataFrame:
    relative_directory = _relative_posix(directory, project_root)
    if ids is None:
        identity_ids = [f"survface:unmated:{Path(name).stem}" for name in names]
        official_ids = pd.array([pd.NA] * len(names), dtype="Int64")
    else:
        identity_ids = [f"survface:registered:{official_id:04d}" for official_id in ids]
        official_ids = pd.array(ids, dtype="Int64")
    return pd.DataFrame(
        {
            "image_id": [f"survface:{role}:{Path(name).stem}" for name in names],
            "identity_id": identity_ids,
            "split": "test",
            "image_path": [f"{relative_directory}/{name}" for name in names],
            "dataset": "qmul-survface-v1",
            "protocol_role": role,
            "probe_type": {
                "gallery": "not_applicable",
                "registered_probe": "registered",
                "unknown_unknown_probe": "unknown_unknown",
            }[role],
            "official_identity_id": official_ids,
            "protocol_index": np.arange(len(names), dtype=np.int64),
        }
    )


def build_survface_training_manifest(
    survface_root: str | Path,
    project_root: str | Path,
    *,
    seed: int = 42,
    development_fraction: float = 0.80,
) -> SurvFaceTrainingBundle:
    """Build a leakage-safe manifest from the official SurvFace training_set.

    The official identification test gallery and probes are not read here.
    Identities, rather than image rows, are deterministically divided between
    development and calibration so PCA/PQ fitting never observes calibration
    or official-test identities through this helper.
    """

    root = _resolved_directory(survface_root, "QMUL-SurvFace root")
    project = _resolved_directory(project_root, "project root")
    training_dir = _resolved_directory(
        root / "training_set", "SurvFace training set"
    )
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between 0 and 1")

    identity_images: dict[str, list[Path]] = {}
    empty_identity_count = 0
    for identity_dir in sorted(path for path in training_dir.iterdir() if path.is_dir()):
        images = _image_files(identity_dir)
        if images:
            identity_images[identity_dir.name] = images
        else:
            empty_identity_count += 1
    if len(identity_images) < 2:
        raise ValueError("SurvFace training_set requires at least two non-empty identities")

    shuffled_identities = sorted(identity_images)
    random.Random(seed).shuffle(shuffled_identities)
    development_count = max(
        1,
        min(
            len(shuffled_identities) - 1,
            int(len(shuffled_identities) * development_fraction),
        ),
    )
    development_ids = set(shuffled_identities[:development_count])
    split_by_identity = {
        identity: "development" if identity in development_ids else "calibration"
        for identity in identity_images
    }

    rows: list[dict[str, Any]] = []
    for identity in sorted(identity_images):
        stable_identity = f"survface:train:{identity}"
        for image in identity_images[identity]:
            rows.append(
                {
                    "image_id": f"{stable_identity}:{image.stem}",
                    "identity_id": stable_identity,
                    "split": split_by_identity[identity],
                    "image_path": _relative_posix(image, project),
                    "dataset": "qmul-survface-v1",
                    "protocol_role": "training",
                    "probe_type": "not_applicable",
                }
            )

    manifest = pd.DataFrame(rows)
    _validate_manifest(manifest)
    validate_identity_disjoint_splits(manifest)
    split_identity_counts = (
        manifest.groupby("split")["identity_id"].nunique().astype(int).to_dict()
    )
    split_image_counts = manifest["split"].value_counts().astype(int).to_dict()
    summary = {
        "dataset": "qmul-survface-v1",
        "protocol": "official-training-identity-disjoint",
        "seed": int(seed),
        "development_fraction": float(development_fraction),
        "image_count": int(len(manifest)),
        "identity_count": int(manifest["identity_id"].nunique()),
        "development_identity_count": int(split_identity_counts["development"]),
        "calibration_identity_count": int(split_identity_counts["calibration"]),
        "development_image_count": int(split_image_counts["development"]),
        "calibration_image_count": int(split_image_counts["calibration"]),
        "empty_identity_directory_count": int(empty_identity_count),
        "official_test_included": False,
    }
    return SurvFaceTrainingBundle(manifest=manifest, summary=summary)


def build_survface_official_manifest(
    survface_root: str | Path,
    project_root: str | Path,
) -> SurvFaceOfficialBundle:
    """Build the official QMUL-SurvFace open-set identification manifest.

    The provided gallery/mated MAT order is retained. Unmated probes have no
    ground-truth identity in the official protocol, so each gets an opaque,
    unique identity ID and must be evaluated only as an unknown probe.
    """

    root = _resolved_directory(survface_root, "QMUL-SurvFace root")
    project = _resolved_directory(project_root, "project root")
    evaluation_dir = _resolved_directory(
        root / "Face_Identification_Evaluation", "SurvFace identification evaluation"
    )
    test_dir = _resolved_directory(
        root / "Face_Identification_Test_Set", "SurvFace identification test set"
    )
    gallery_dir = _resolved_directory(test_dir / "gallery", "SurvFace gallery")
    mated_dir = _resolved_directory(test_dir / "mated_probe", "SurvFace mated probes")
    unmated_dir = _resolved_directory(test_dir / "unmated_probe", "SurvFace unmated probes")

    gallery_names, gallery_ids = _load_survface_pairs(
        evaluation_dir / "gallery_img_ID_pairs.mat",
        names_key="gallery_set",
        ids_key="gallery_ids",
    )
    mated_names, mated_ids = _load_survface_pairs(
        evaluation_dir / "mated_probe_img_ID_pairs.mat",
        names_key="mated_probe_set",
        ids_key="mated_probe_ids",
    )
    unmated_names = [path.name for path in _image_files(unmated_dir)]
    if not unmated_names:
        raise ValueError(f"no unmated probe images found under {unmated_dir}")

    _require_exact_file_set(gallery_dir, gallery_names, "gallery")
    _require_exact_file_set(mated_dir, mated_names, "mated probe")
    if set(gallery_ids) != set(mated_ids):
        raise ValueError("official gallery and mated probe identity sets differ")

    gallery = _survface_frame(
        names=gallery_names,
        ids=gallery_ids,
        directory=gallery_dir,
        project_root=project,
        role="gallery",
    )
    registered_probes = _survface_frame(
        names=mated_names,
        ids=mated_ids,
        directory=mated_dir,
        project_root=project,
        role="registered_probe",
    )
    unknown_unknown_probes = _survface_frame(
        names=unmated_names,
        ids=None,
        directory=unmated_dir,
        project_root=project,
        role="unknown_unknown_probe",
    )
    manifest = pd.concat(
        [gallery, registered_probes, unknown_unknown_probes], ignore_index=True
    )
    _validate_manifest(manifest)
    validate_identity_disjoint_splits(manifest)

    gallery_identity_ids = tuple(
        f"survface:registered:{official_id:04d}" for official_id in sorted(set(gallery_ids))
    )
    unknown_identity_ids = tuple(unknown_unknown_probes["identity_id"].tolist())
    if set(gallery_identity_ids).intersection(unknown_identity_ids):
        raise ValueError("registered and unmated synthetic identity IDs overlap")

    summary = {
        "dataset": "qmul-survface-v1",
        "protocol": "official-open-set-identification",
        "image_count": int(len(manifest)),
        "gallery_image_count": int(len(gallery)),
        "registered_probe_image_count": int(len(registered_probes)),
        "unknown_unknown_probe_image_count": int(len(unknown_unknown_probes)),
        "registered_identity_count": int(len(gallery_identity_ids)),
        "known_unknown_identity_count": 0,
        # The official bundle does not publish identity labels for unmated
        # probes.  The opaque per-image keys above are implementation keys,
        # not evidence that every image belongs to a different person.
        "unknown_unknown_identity_count": None,
        "unknown_unknown_identity_labels_available": False,
        "unmated_probe_count": int(len(unknown_unknown_probes)),
        "training_set_included": False,
        "protocol_note": (
            "Official gallery roles and MAT ordering are fixed; do not resample the gallery. "
            "The official protocol has no known-unknown probe group and does not "
            "publish identity labels for unmated probes."
        ),
    }
    return SurvFaceOfficialBundle(
        manifest=manifest,
        gallery_identities=gallery_identity_ids,
        unknown_unknown_identities=unknown_identity_ids,
        summary=summary,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_targets(output_dir: str | Path, names: tuple[str, ...], overwrite: bool) -> dict[str, Path]:
    directory = Path(output_dir).expanduser().resolve()
    targets = {name: directory / name for name in names}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing manifest outputs: "
            + ", ".join(str(path) for path in existing)
        )
    directory.mkdir(parents=True, exist_ok=True)
    return targets


def _identity_text(identities: tuple[str, ...]) -> str:
    return "".join(f"{identity}\n" for identity in identities)


def write_lfw_manifest_bundle(
    bundle: ManifestBundle,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write LFW manifest inputs expected by notebook 00."""

    targets = _prepare_targets(
        output_dir,
        (
            "face_manifest.csv",
            "gallery_identities.txt",
            "unknown_unknown_identities.txt",
            "summary.json",
        ),
        overwrite,
    )
    _atomic_write_csv(targets["face_manifest.csv"], bundle.manifest)
    _atomic_write_text(
        targets["gallery_identities.txt"], _identity_text(bundle.gallery_identities)
    )
    _atomic_write_text(
        targets["unknown_unknown_identities.txt"],
        _identity_text(bundle.unknown_unknown_identities),
    )
    _atomic_write_text(
        targets["summary.json"],
        json.dumps(bundle.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return targets


def write_survface_official_bundle(
    bundle: SurvFaceOfficialBundle,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write the official SurvFace role-preserving manifest files."""

    targets = _prepare_targets(
        output_dir,
        (
            "official_manifest.csv",
            "gallery.csv",
            "registered_probes.csv",
            "unknown_unknown_probes.csv",
            "gallery_identities.txt",
            "unknown_unknown_identities.txt",
            "summary.json",
        ),
        overwrite,
    )
    _atomic_write_csv(targets["official_manifest.csv"], bundle.manifest)
    _atomic_write_csv(targets["gallery.csv"], bundle.gallery)
    _atomic_write_csv(targets["registered_probes.csv"], bundle.registered_probes)
    _atomic_write_csv(
        targets["unknown_unknown_probes.csv"], bundle.unknown_unknown_probes
    )
    _atomic_write_text(
        targets["gallery_identities.txt"], _identity_text(bundle.gallery_identities)
    )
    _atomic_write_text(
        targets["unknown_unknown_identities.txt"],
        _identity_text(bundle.unknown_unknown_identities),
    )
    _atomic_write_text(
        targets["summary.json"],
        json.dumps(bundle.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return targets


def write_survface_training_bundle(
    bundle: SurvFaceTrainingBundle,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write SurvFace development/calibration manifest files."""

    targets = _prepare_targets(
        output_dir,
        ("training_manifest.csv", "training_summary.json"),
        overwrite,
    )
    _atomic_write_csv(targets["training_manifest.csv"], bundle.manifest)
    _atomic_write_text(
        targets["training_summary.json"],
        json.dumps(bundle.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return targets
