from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from research.runtime.hashing import canonical_sha256, sha256_file


TINYFACE_DATASET_ID = "tinyface"
TINYFACE_PROTOCOL_ID = "tinyface_official_closed_set_v1"
TINYFACE_ARTIFACT_TYPE = "tinyface_official_identification_protocol_v1"
TINYFACE_OFFICIAL_COUNTS = {
    "development_pool": 7_804,
    "gallery_match": 4_443,
    "gallery_distractor": 153_428,
    "registered_probe": 3_728,
}


@dataclass(frozen=True)
class TinyFaceOfficialBundle:
    """Validated TinyFace closed-set 1:N identification protocol.

    TinyFace has registered probes and a large distractor gallery, but no
    non-mated probe split.  The bundle is therefore deliberately not adapted
    to the project's open-set FPIR/TPIR protocol.
    """

    manifest: pd.DataFrame
    summary: dict[str, Any]
    protocol_uid: str


def _stable_key(*, seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(
        f"{namespace}\x1f{int(seed)}\x1f{value}".encode("utf-8")
    ).hexdigest()


def _resolve_root(root: str | Path) -> Path:
    source = Path(root).expanduser().resolve()
    required = (
        source / "Training_Set",
        source / "Testing_Set" / "Gallery_Match",
        source / "Testing_Set" / "Gallery_Distractor",
        source / "Testing_Set" / "Probe",
        source / "Testing_Set" / "gallery_match_img_ID_pairs.mat",
        source / "Testing_Set" / "probe_img_ID_pairs.mat",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"TinyFace protocol files are missing: {missing}")
    return source


def _mat_pairs(path: Path, *, names_key: str, ids_key: str) -> list[tuple[str, int]]:
    payload = loadmat(path, squeeze_me=True)
    if names_key not in payload or ids_key not in payload:
        raise ValueError(f"TinyFace MAT file is missing {names_key}/{ids_key}: {path}")
    names = np.atleast_1d(payload[names_key]).reshape(-1)
    ids = np.atleast_1d(payload[ids_key]).reshape(-1)
    if len(names) != len(ids) or len(names) == 0:
        raise ValueError(f"TinyFace MAT name/ID arrays are invalid: {path}")
    pairs: list[tuple[str, int]] = []
    for raw_name, raw_id in zip(names.tolist(), ids.tolist(), strict=True):
        name = str(raw_name).strip()
        identity = int(raw_id)
        if not name or identity < 1:
            raise ValueError(f"TinyFace MAT contains an invalid name or ID: {path}")
        pairs.append((name, identity))
    if len({name for name, _ in pairs}) != len(pairs):
        raise ValueError(f"TinyFace MAT contains duplicate filenames: {path}")
    return pairs


def _jpgs(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".jpg"),
        key=lambda path: path.relative_to(directory).as_posix(),
    )


def _row(
    *,
    root: Path,
    image_path: Path,
    identity_id: str,
    source_identity_id: str,
    split: str,
    protocol_role: str,
    protocol_index: int,
    is_distractor: bool,
) -> dict[str, Any]:
    relative = image_path.relative_to(root).as_posix()
    return {
        "image_id": f"tinyface:{relative}",
        "identity_id": identity_id,
        "source_identity_id": source_identity_id,
        "split": split,
        "image_path": str(image_path),
        "source_relative_path": relative,
        "dataset": TINYFACE_DATASET_ID,
        "dataset_role": (
            "compressor_fit_development"
            if split == "development"
            else "official_identification_test"
        ),
        "protocol_role": protocol_role,
        "protocol_index": int(protocol_index),
        "is_distractor_gallery": bool(is_distractor),
        "is_mated": (pd.NA if protocol_role.startswith("gallery") or split == "development" else True),
        "protocol_kind": "official_closed_set_1_to_n_with_distractors",
        "protocol_family_uid": TINYFACE_PROTOCOL_ID,
        "artifact_type": TINYFACE_ARTIFACT_TYPE,
        "official_result_eligible": True,
        "preprocessing_mode": "official_face_crop_resize",
    }


def build_tinyface_official_bundle(
    root: str | Path,
    *,
    strict_official: bool = True,
) -> TinyFaceOfficialBundle:
    """Build and validate the official TinyFace test and development roles."""

    source = _resolve_root(root)
    testing = source / "Testing_Set"
    gallery_mat = testing / "gallery_match_img_ID_pairs.mat"
    probe_mat = testing / "probe_img_ID_pairs.mat"
    gallery_pairs = _mat_pairs(
        gallery_mat,
        names_key="gallery_set",
        ids_key="gallery_ids",
    )
    probe_pairs = _mat_pairs(
        probe_mat,
        names_key="probe_set",
        ids_key="probe_ids",
    )

    rows: list[dict[str, Any]] = []
    training_paths = _jpgs(source / "Training_Set")
    for index, path in enumerate(training_paths):
        local_identity = path.parent.relative_to(source / "Training_Set").as_posix()
        rows.append(
            _row(
                root=source,
                image_path=path,
                identity_id=f"tinyface:train:{local_identity}",
                source_identity_id=local_identity,
                split="development",
                protocol_role="development_pool",
                protocol_index=index,
                is_distractor=False,
            )
        )

    gallery_dir = testing / "Gallery_Match"
    for index, (name, identity) in enumerate(gallery_pairs):
        path = gallery_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"TinyFace match-gallery image is missing: {path}")
        rows.append(
            _row(
                root=source,
                image_path=path,
                identity_id=f"tinyface:test:{identity:04d}",
                source_identity_id=str(identity),
                split="test",
                protocol_role="gallery_match",
                protocol_index=index,
                is_distractor=False,
            )
        )

    distractor_dir = testing / "Gallery_Distractor"
    distractor_paths = _jpgs(distractor_dir)
    for index, path in enumerate(distractor_paths):
        relative = path.relative_to(distractor_dir).as_posix()
        rows.append(
            _row(
                root=source,
                image_path=path,
                identity_id=f"tinyface:distractor:{relative}",
                source_identity_id=relative,
                split="test",
                protocol_role="gallery_distractor",
                protocol_index=index,
                is_distractor=True,
            )
        )

    probe_dir = testing / "Probe"
    for index, (name, identity) in enumerate(probe_pairs):
        path = probe_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"TinyFace probe image is missing: {path}")
        rows.append(
            _row(
                root=source,
                image_path=path,
                identity_id=f"tinyface:test:{identity:04d}",
                source_identity_id=str(identity),
                split="test",
                protocol_role="registered_probe",
                protocol_index=index,
                is_distractor=False,
            )
        )

    manifest = pd.DataFrame.from_records(rows)
    if manifest["image_id"].duplicated().any() or manifest["image_path"].duplicated().any():
        raise ValueError("TinyFace images must be unique across protocol roles")
    role_counts = {
        str(role): int(count)
        for role, count in manifest["protocol_role"].value_counts().items()
    }
    if strict_official and role_counts != TINYFACE_OFFICIAL_COUNTS:
        raise ValueError(
            "TinyFace installed counts do not match the official release: "
            f"expected={TINYFACE_OFFICIAL_COUNTS}, actual={role_counts}"
        )
    gallery_identities = set(
        manifest.loc[manifest["protocol_role"].eq("gallery_match"), "identity_id"]
    )
    probe_identities = set(
        manifest.loc[manifest["protocol_role"].eq("registered_probe"), "identity_id"]
    )
    if not probe_identities.issubset(gallery_identities):
        raise ValueError("TinyFace official probes must all have match-gallery identities")
    if strict_official and len(gallery_identities) != 2_569:
        raise ValueError("TinyFace official test identity count must be 2,569")

    protocol_payload = {
        "protocol_family_uid": TINYFACE_PROTOCOL_ID,
        "gallery_pairs_sha256": sha256_file(gallery_mat),
        "probe_pairs_sha256": sha256_file(probe_mat),
        "role_counts": role_counts,
        "ordered_rows": manifest[
            ["source_relative_path", "identity_id", "protocol_role", "protocol_index"]
        ].to_dict(orient="records"),
    }
    protocol_uid = f"{TINYFACE_PROTOCOL_ID}-{canonical_sha256(protocol_payload)[:16]}"
    manifest["protocol_uid"] = protocol_uid
    summary = {
        "artifact_type": TINYFACE_ARTIFACT_TYPE,
        "dataset_id": TINYFACE_DATASET_ID,
        "protocol_kind": "official_closed_set_1_to_n_with_distractors",
        "protocol_uid": protocol_uid,
        "strict_official": bool(strict_official),
        "official_result_eligible": bool(strict_official),
        "open_set_protocol": False,
        "non_mated_probe_count": 0,
        "fpir_tpir_metrics_applicable": False,
        "identity_count": int(len(gallery_identities)),
        "probe_identity_count": int(len(probe_identities)),
        "role_counts": role_counts,
        "total_image_count": int(len(manifest)),
        "preprocessing_mode": "official_face_crop_resize",
        "compression_fit_role": "development_pool",
        "gallery_roles": ["gallery_match", "gallery_distractor"],
        "query_role": "registered_probe",
        "official_metrics": ["mean_average_precision", "rank_1", "rank_5", "rank_10", "rank_20"],
        "source_root": str(source),
        "source_protocol_files": {
            gallery_mat.name: sha256_file(gallery_mat),
            probe_mat.name: sha256_file(probe_mat),
        },
    }
    return TinyFaceOfficialBundle(
        manifest=manifest,
        summary=summary,
        protocol_uid=protocol_uid,
    )


def select_tinyface_protocol_fraction(
    manifest: pd.DataFrame,
    *,
    data_fraction: float,
    seed: int = 42,
    minimum_development_samples: int = 4_096,
) -> pd.DataFrame:
    """Select a deterministic, identity-preserving non-paper quick scope."""

    try:
        fraction = float(data_fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("data_fraction must be in (0, 1]") from exc
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("data_fraction must be in (0, 1]")
    required = {
        "image_id", "identity_id", "protocol_role", "protocol_index",
        "official_result_eligible", "protocol_uid",
    }
    missing = sorted(required - set(manifest.columns))
    if missing or manifest.empty:
        raise ValueError(f"TinyFace manifest is empty or missing columns: {missing}")
    if fraction == 1.0:
        selected = manifest.copy().reset_index(drop=True)
        selected["scope_data_fraction"] = 1.0
        selected["scope_seed"] = int(seed)
        selected["scope_is_full"] = True
        return selected

    frame = manifest.copy()
    probes = frame.loc[frame["protocol_role"].eq("registered_probe")]
    identities = sorted(set(probes["identity_id"].astype(str)))
    identity_count = max(1, int(math.ceil(len(identities) * fraction)))
    selected_identities = set(
        sorted(
            identities,
            key=lambda value: _stable_key(seed=seed, namespace="test_identity", value=value),
        )[:identity_count]
    )
    distractors = frame.loc[frame["protocol_role"].eq("gallery_distractor")]
    distractor_count = max(1, int(math.ceil(len(distractors) * fraction)))
    selected_distractor_ids = set(
        sorted(
            distractors["image_id"].astype(str),
            key=lambda value: _stable_key(seed=seed, namespace="distractor", value=value),
        )[:distractor_count]
    )
    development = frame.loc[frame["protocol_role"].eq("development_pool")]
    requested_development = max(
        int(minimum_development_samples),
        int(math.ceil(len(development) * fraction)),
    )
    requested_development = min(len(development), requested_development)
    selected_development_ids = set(
        sorted(
            development["image_id"].astype(str),
            key=lambda value: _stable_key(seed=seed, namespace="development", value=value),
        )[:requested_development]
    )
    keep = (
        frame["identity_id"].astype(str).isin(selected_identities)
        | frame["image_id"].astype(str).isin(selected_distractor_ids)
        | frame["image_id"].astype(str).isin(selected_development_ids)
    )
    selected = frame.loc[keep].copy()
    selected["source_protocol_uid"] = selected["protocol_uid"]
    quick_payload = {
        "source_protocol_uid": str(selected["protocol_uid"].iloc[0]),
        "data_fraction": fraction,
        "seed": int(seed),
        "image_ids": sorted(selected["image_id"].astype(str)),
    }
    selected["protocol_uid"] = (
        f"{TINYFACE_PROTOCOL_ID}-quick-{canonical_sha256(quick_payload)[:16]}"
    )
    selected["official_result_eligible"] = False
    selected["scope_data_fraction"] = fraction
    selected["scope_seed"] = int(seed)
    selected["scope_is_full"] = False
    role_order = {
        "development_pool": 0,
        "gallery_match": 1,
        "gallery_distractor": 2,
        "registered_probe": 3,
    }
    selected["_role_order"] = selected["protocol_role"].map(role_order)
    selected = selected.sort_values(
        ["_role_order", "protocol_index", "image_id"], kind="stable"
    ).drop(columns="_role_order").reset_index(drop=True)
    for role, indexes in selected.groupby("protocol_role", sort=False).groups.items():
        selected.loc[list(indexes), "protocol_index"] = np.arange(len(indexes))
    selected["protocol_index"] = selected["protocol_index"].astype("int64")
    return selected
