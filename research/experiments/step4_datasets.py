from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from research.protocols import (
    build_survface_official_protocol,
    validate_identity_disjoint_splits,
)


SUPPORTED_STEP4_DATASETS = ("lfw", "survface")
SURVFACE_OFFICIAL_ROLES = {
    "gallery",
    "registered_probe",
    "unknown_unknown_probe",
}


@dataclass(frozen=True)
class Step4DatasetSpec:
    dataset_id: str
    protocol_adapter: str
    manifest_paths: tuple[Path, ...]
    aligned_bundle_dir: Path
    landmark_region_bundle_dir: Path


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
    aligned_value = dataset.get(
        "aligned_bundle_dir",
        f"data/interim/step4/{selected}/aligned_112",
    )
    landmark_value = dataset.get(
        "landmark_region_bundle_dir",
        f"data/interim/step4/{selected}/landmark_regions_106",
    )
    return Step4DatasetSpec(
        dataset_id=selected,
        protocol_adapter=str(dataset["protocol_adapter"]),
        manifest_paths=manifest_paths,
        aligned_bundle_dir=(root / str(aligned_value)).resolve(),
        landmark_region_bundle_dir=(root / str(landmark_value)).resolve(),
    )


def load_step4_source_manifest(spec: Step4DatasetSpec) -> pd.DataFrame:
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
    Only the expensive backward Grad-CAM pass is capped for SurvFace.
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
    if str(dataset_id).lower() == "survface":
        if "protocol_role" not in rows:
            raise ValueError("SurvFace manifest must contain protocol_role")
        strata = rows["protocol_role"].astype(str)
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
