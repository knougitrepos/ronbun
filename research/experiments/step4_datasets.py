from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from research.protocols import (
    build_survface_official_protocol,
    validate_identity_disjoint_splits,
)
from research.datasets.rfw import build_rfw_verification_bundle
from research.datasets.rfw_custom import build_rfw_custom_open_set_bundle


SUPPORTED_STEP4_DATASETS = ("lfw", "survface", "rfw_custom")
SURVFACE_OFFICIAL_ROLES = {
    "gallery",
    "registered_probe",
    "unknown_unknown_probe",
}


@dataclass(frozen=True)
class Step4DatasetSpec:
    project_root: Path
    dataset_id: str
    protocol_adapter: str
    manifest_paths: tuple[Path, ...]
    aligned_bundle_dir: Path
    landmark_region_bundle_dir: Path
    preprocessing_mode: str
    require_full_coverage: bool
    source_archive_sha256: str | None = None
    aligned_bin_archive_path: Path | None = None
    aligned_bin_archive_sha256: str | None = None
    protocol_options: dict[str, Any] | None = None


def resolve_step4_dataset_spec(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    dataset_id: str | None = None,
) -> Step4DatasetSpec:
    root = Path(project_root).resolve()
    selected = str(dataset_id or config["run"]["dataset_id"]).strip().lower()
    if selected not in SUPPORTED_STEP4_DATASETS:
        raise ValueError(
            f"dataset_id must be one of {SUPPORTED_STEP4_DATASETS}, got {selected!r}"
        )
    dataset = config["datasets"][selected]
    if selected == "survface":
        manifest_values = (
            dataset["training_manifest_path"],
            dataset["manifest_path"],
        )
    else:
        manifest_values = (dataset["manifest_path"],)
    manifest_paths = tuple((root / str(value)).resolve() for value in manifest_values)
    for path in manifest_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    aligned_bin_path: Path | None = None
    if selected == "rfw_custom":
        aligned_bin_value = dataset.get("aligned_bin_archive_path")
        if aligned_bin_value is None:
            raise ValueError("RFW custom aligned_bin_archive_path is required")
        aligned_bin_path = (root / str(aligned_bin_value)).resolve()
        if not aligned_bin_path.is_file():
            raise FileNotFoundError(aligned_bin_path)
    aligned_value = dataset.get(
        "aligned_bundle_dir",
        f"data/interim/step4/{selected}/aligned_112",
    )
    landmark_value = dataset.get(
        "landmark_region_bundle_dir",
        f"data/interim/step4/{selected}/landmark_regions_106",
    )
    return Step4DatasetSpec(
        project_root=root,
        dataset_id=selected,
        protocol_adapter=str(dataset["protocol_adapter"]),
        manifest_paths=manifest_paths,
        aligned_bundle_dir=(root / str(aligned_value)).resolve(),
        landmark_region_bundle_dir=(root / str(landmark_value)).resolve(),
        preprocessing_mode=str(
            dataset.get("preprocessing_mode", "detect_and_align")
        ),
        require_full_coverage=bool(
            dataset.get("require_full_coverage", False)
        ),
        source_archive_sha256=(
            str(dataset["source_archive_sha256"]).lower()
            if dataset.get("source_archive_sha256") is not None
            else None
        ),
        aligned_bin_archive_path=(
            aligned_bin_path
            if aligned_bin_path is not None
            else None
        ),
        aligned_bin_archive_sha256=(
            str(dataset["aligned_bin_archive_sha256"]).lower()
            if dataset.get("aligned_bin_archive_sha256") is not None
            else None
        ),
        protocol_options=(
            dict(dataset["protocol"])
            if isinstance(dataset.get("protocol"), dict)
            else None
        ),
    )


@lru_cache(maxsize=4)
def _load_rfw_custom_manifest_cached(
    source_archive_path: str,
    project_root: str,
    source_archive_sha256: str,
    protocol_options_json: str,
) -> pd.DataFrame:
    options = json.loads(protocol_options_json)
    official = build_rfw_verification_bundle(
        source_archive_path,
        project_root,
        strict_official=True,
    )
    actual_sha = str(official.summary["source_archive_sha256"]).lower()
    if actual_sha != str(source_archive_sha256).lower():
        raise ValueError(
            "RFW custom source archive SHA-256 mismatch: "
            f"{actual_sha} != {str(source_archive_sha256).lower()}"
        )
    bundle = build_rfw_custom_open_set_bundle(
        official.manifest,
        source_archive_sha256=actual_sha,
        gallery_identity_count_per_group=options[
            "gallery_identity_count_per_group"
        ],
        enrollment_count=int(options.get("enrollment_count", 1)),
        seed=int(options.get("seed", 42)),
        development_fraction=float(options.get("development_fraction", 0.40)),
        calibration_fraction=float(options.get("calibration_fraction", 0.20)),
        unknown_unknown_fraction=float(
            options.get("unknown_unknown_fraction", 0.50)
        ),
    )
    return bundle.manifest


def load_step4_source_manifest(spec: Step4DatasetSpec) -> pd.DataFrame:
    if spec.dataset_id == "rfw_custom":
        if len(spec.manifest_paths) != 1:
            raise ValueError("RFW custom requires exactly one source archive")
        if not spec.source_archive_sha256 or not spec.protocol_options:
            raise ValueError(
                "RFW custom requires source_archive_sha256 and protocol options"
            )
        manifest = _load_rfw_custom_manifest_cached(
            str(spec.manifest_paths[0]),
            str(spec.project_root),
            spec.source_archive_sha256,
            json.dumps(spec.protocol_options, sort_keys=True, separators=(",", ":")),
        ).copy()
    else:
        parts = [pd.read_csv(path) for path in spec.manifest_paths]
        manifest = pd.concat(parts, ignore_index=True, sort=False)
    if "image_id" not in manifest and "sample_id" in manifest:
        manifest = manifest.rename(columns={"sample_id": "image_id"})
    required = {"image_id", "identity_id", "split", "image_path"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Step 4 source manifest is missing columns: {missing}")
    for column in ("image_id", "identity_id", "split", "image_path"):
        if manifest[column].isna().any():
            raise ValueError(f"Step 4 source manifest {column} contains missing values")
        manifest[column] = manifest[column].astype(str)
    if manifest["image_id"].duplicated().any():
        duplicates = (
            manifest.loc[manifest["image_id"].duplicated(keep=False), "image_id"]
            .head(5)
            .tolist()
        )
        raise ValueError(f"Step 4 image_id must be unique: {duplicates}")
    validate_identity_disjoint_splits(
        manifest.loc[:, ["image_id", "identity_id", "split", "image_path"]]
    )

    if spec.dataset_id == "survface":
        official_mask = manifest["protocol_role"].astype(str).isin(
            SURVFACE_OFFICIAL_ROLES
        )
        official = manifest.loc[official_mask].copy()
        if official.empty:
            raise ValueError("SurvFace official protocol rows are missing")
        protocol = build_survface_official_protocol(official)
        if not protocol.known_unknown_probes.empty:
            raise ValueError("SurvFace official protocol contains known-unknown rows")
        training = manifest.loc[~official_mask]
        if training.empty:
            raise ValueError("SurvFace training development/calibration rows are missing")
        if set(training["split"]) != {"development", "calibration"}:
            raise ValueError(
                "SurvFace training rows must contain development and calibration"
            )
    elif spec.dataset_id == "rfw_custom":
        required_roles = {
            "development_pool",
            "calibration_pool",
            "gallery",
            "registered_probe",
            "known_unknown_probe",
            "unknown_unknown_probe",
        }
        if set(manifest["protocol_role"].astype(str)) != required_roles:
            raise ValueError("RFW custom protocol roles are incomplete")
        if manifest["official_result_eligible"].astype(bool).any():
            raise ValueError("RFW custom rows must not claim official eligibility")
        if set(manifest["checkpoint_overlap_status"].astype(str)) != {"UNKNOWN"}:
            raise ValueError("RFW custom checkpoint overlap must remain UNKNOWN")
    return manifest.reset_index(drop=True)


def select_step4_saliency_sample_mask(
    manifest: pd.DataFrame,
    eligible: np.ndarray,
    *,
    dataset_id: str,
    maximum_samples: int | None,
    seed: int,
) -> np.ndarray:
    """Select a deterministic, role-balanced saliency subset.

    Compression and open-set evaluation still use the full selected dataset.
    Only the expensive backward Grad-CAM pass is capped. SurvFace preserves
    protocol roles; RFW-Custom preserves demographic-group × protocol-role
    strata.
    """

    eligibility = np.asarray(eligible)
    if (
        eligibility.ndim != 1
        or len(eligibility) != len(manifest)
        or eligibility.dtype.kind != "b"
    ):
        raise ValueError("eligible must be a boolean vector aligned with manifest")
    if "sample_id" not in manifest:
        raise ValueError("manifest must contain sample_id")
    if maximum_samples is None:
        return eligibility.astype(bool, copy=True)
    limit = int(maximum_samples)
    if isinstance(maximum_samples, bool) or limit != maximum_samples or limit <= 0:
        raise ValueError("maximum_samples must be a positive integer or null")
    eligible_indices = np.flatnonzero(eligibility)
    if len(eligible_indices) <= limit:
        return eligibility.astype(bool, copy=True)

    rows = manifest.reset_index(drop=True)
    selected_dataset = str(dataset_id).lower()
    if selected_dataset == "survface":
        if "protocol_role" not in rows:
            raise ValueError("SurvFace manifest must contain protocol_role")
        strata = rows["protocol_role"].astype(str)
    elif selected_dataset == "rfw_custom":
        required = {"rfw_group", "protocol_role"}
        missing = sorted(required.difference(rows.columns))
        if missing:
            raise ValueError(
                f"RFW custom manifest is missing saliency strata: {missing}"
            )
        strata = (
            rows["rfw_group"].astype(str)
            + "|"
            + rows["protocol_role"].astype(str)
        )
    else:
        strata = rows["split"].astype(str)
    candidates: dict[str, list[tuple[str, int]]] = {}
    for index in eligible_indices:
        sample_id = str(rows.at[int(index), "sample_id"])
        digest = hashlib.sha256(
            f"{int(seed)}\x1f{sample_id}".encode("utf-8")
        ).hexdigest()
        candidates.setdefault(str(strata.iloc[int(index)]), []).append(
            (digest, int(index))
        )
    for values in candidates.values():
        values.sort()

    selected: list[int] = []
    positions = {name: 0 for name in candidates}
    names = sorted(candidates)
    while len(selected) < limit:
        progressed = False
        for name in names:
            position = positions[name]
            values = candidates[name]
            if position >= len(values):
                continue
            selected.append(values[position][1])
            positions[name] += 1
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    mask = np.zeros(len(rows), dtype=bool)
    mask[np.asarray(selected, dtype=np.int64)] = True
    return mask
