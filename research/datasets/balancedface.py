from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import pandas as pd

from research.datasets.sources import DatasetIntegrityError
from research.datasets.sources import inspect_tar_archive, require_valid_tar


BALANCEDFACE_RACE_LABELS = {
    0: "Caucasian",
    1: "Indian",
    2: "Asian",
    3: "African",
}
BALANCEDFACE_EXPECTED_IMAGE_COUNT = 1_251_416
BALANCEDFACE_EXPECTED_IDENTITY_COUNT = 27_999
BALANCEDFACE_EXPECTED_RACE_IMAGE_COUNTS = {
    "Caucasian": 326_484,
    "Indian": 275_063,
    "Asian": 325_493,
    "African": 324_376,
}
BALANCEDFACE_EXPECTED_RACE_IDENTITY_COUNTS = {
    "Caucasian": 7_000,
    "Indian": 6_999,
    "Asian": 7_000,
    "African": 7_000,
}
_WINDOWS_RESERVED_NAME = re.compile(
    r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class BalancedFaceIndexBundle:
    """Leakage-audited metadata for the aligned Equalizedface RecordIO."""

    manifest: pd.DataFrame
    excluded_identities: pd.DataFrame
    summary: dict[str, Any]


def _relative_posix(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"dataset file must be inside project_root: {path.resolve()}"
        ) from exc


def _validate_list_path(value: str) -> str:
    text = str(value).strip()
    raw_parts = text.split("/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or len(raw_parts) != 2
        or any(part in {"", ".", ".."} for part in raw_parts)
        or any(
            ":" in part
            or part.endswith((" ", "."))
            or _WINDOWS_RESERVED_NAME.fullmatch(part) is not None
            for part in raw_parts
        )
    ):
        raise DatasetIntegrityError(
            f"unsafe or unexpected BalancedFace list path: {text!r}"
        )
    return text


def _stable_split_key(seed: int, group: str, identity: str) -> str:
    payload = f"{seed}\0{group}\0{identity}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identity_splits(
    identity_groups: pd.DataFrame,
    *,
    seed: int,
    development_fraction: float,
) -> dict[str, str]:
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between 0 and 1")
    groups = set(identity_groups["dataset_group"].astype(str))
    expected_groups = set(BALANCEDFACE_RACE_LABELS.values())
    if groups != expected_groups:
        raise DatasetIntegrityError(
            "BalancedFace overlap exclusion must leave all four groups: "
            f"missing={sorted(expected_groups.difference(groups))}, "
            f"extra={sorted(groups.difference(expected_groups))}"
        )
    split_by_identity: dict[str, str] = {}
    for group, group_rows in identity_groups.groupby(
        "dataset_group", sort=True, observed=True
    ):
        identities = group_rows["source_identity_id"].astype(str).tolist()
        identities.sort(key=lambda identity: _stable_split_key(seed, str(group), identity))
        if len(identities) < 2:
            raise DatasetIntegrityError(
                f"BalancedFace group {group!r} needs at least two identities"
            )
        development_count = max(
            1,
            min(len(identities) - 1, int(len(identities) * development_fraction)),
        )
        split_by_identity.update(
            {
                identity: (
                    "development"
                    if index < development_count
                    else "calibration"
                )
                for index, identity in enumerate(identities)
            }
        )
    return split_by_identity


def _validate_official_index(frame: pd.DataFrame) -> None:
    if len(frame) != BALANCEDFACE_EXPECTED_IMAGE_COUNT:
        raise DatasetIntegrityError(
            "BalancedFace list image count mismatch: "
            f"expected={BALANCEDFACE_EXPECTED_IMAGE_COUNT}, actual={len(frame)}"
        )
    if frame["source_identity_id"].nunique() != BALANCEDFACE_EXPECTED_IDENTITY_COUNT:
        raise DatasetIntegrityError(
            "BalancedFace identity count mismatch: "
            f"expected={BALANCEDFACE_EXPECTED_IDENTITY_COUNT}, "
            f"actual={frame['source_identity_id'].nunique()}"
        )
    labels = set(frame["identity_label"].astype(int))
    expected_labels = set(range(BALANCEDFACE_EXPECTED_IDENTITY_COUNT))
    if labels != expected_labels:
        missing = sorted(expected_labels.difference(labels))
        extra = sorted(labels.difference(expected_labels))
        raise DatasetIntegrityError(
            "BalancedFace numeric identity labels are not contiguous 0..27998: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    image_counts = frame["dataset_group"].value_counts().astype(int).to_dict()
    if image_counts != BALANCEDFACE_EXPECTED_RACE_IMAGE_COUNTS:
        raise DatasetIntegrityError(
            "BalancedFace group image counts mismatch: "
            f"expected={BALANCEDFACE_EXPECTED_RACE_IMAGE_COUNTS}, actual={image_counts}"
        )
    identity_counts = (
        frame.groupby("dataset_group", observed=True)["source_identity_id"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    if identity_counts != BALANCEDFACE_EXPECTED_RACE_IDENTITY_COUNTS:
        raise DatasetIntegrityError(
            "BalancedFace group identity counts mismatch: "
            f"expected={BALANCEDFACE_EXPECTED_RACE_IDENTITY_COUNTS}, "
            f"actual={identity_counts}"
        )


def build_balancedface_index_bundle(
    list_path: str | Path,
    recordio_archive_path: str | Path,
    project_root: str | Path,
    *,
    rfw_source_identity_ids: Iterable[str],
    seed: int = 42,
    development_fraction: float = 0.80,
    strict_official: bool = True,
    verify_recordio_archive: bool = True,
) -> BalancedFaceIndexBundle:
    """Build development/calibration metadata and remove RFW identity overlap.

    The returned rows index the aligned RecordIO source; they are not decoded
    image files. A later materialization stage must produce an image/crop
    manifest before an FR encoder can consume them.
    """

    source = Path(list_path).expanduser().resolve()
    recordio_archive = Path(recordio_archive_path).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"BalancedFace list not found: {source}")
    if not recordio_archive.is_file():
        raise FileNotFoundError(
            f"BalancedFace RecordIO archive not found: {recordio_archive}"
        )
    if not project.is_dir():
        raise FileNotFoundError(f"project root not found: {project}")
    rfw_ids = {str(identity).strip() for identity in rfw_source_identity_ids}
    if not rfw_ids:
        raise DatasetIntegrityError(
            "RFW source identities are required for the BalancedFace overlap audit"
        )
    if "" in rfw_ids:
        raise ValueError("rfw_source_identity_ids must not contain empty values")

    recordio_report = None
    recordio_members: dict[str, int] = {}
    recordio_property: str | None = None
    if verify_recordio_archive:
        recordio_report = inspect_tar_archive(
            recordio_archive,
            expected_prefixes=("Equalizedface",),
        )
        require_valid_tar(recordio_report)
        with tarfile.open(recordio_archive, mode="r:*") as handle:
            regular_members = {
                member.name: int(member.size)
                for member in handle.getmembers()
                if member.isfile()
            }
            expected_members = {
                "Equalizedface/property",
                "Equalizedface/train.idx",
                "Equalizedface/train.rec",
            }
            if set(regular_members) != expected_members:
                raise DatasetIntegrityError(
                    "BalancedFace RecordIO archive members mismatch: "
                    f"expected={sorted(expected_members)}, "
                    f"actual={sorted(regular_members)}"
                )
            property_source = handle.extractfile("Equalizedface/property")
            if property_source is None:
                raise DatasetIntegrityError(
                    "BalancedFace RecordIO property member is unreadable"
                )
            with property_source:
                recordio_property = property_source.read().decode("ascii").strip()
            if recordio_property != "27999,112,112":
                raise DatasetIntegrityError(
                    "BalancedFace RecordIO property mismatch: "
                    f"expected='27999,112,112', actual={recordio_property!r}"
                )
            recordio_members = regular_members

    frame = pd.read_csv(
        source,
        sep="\t",
        header=None,
        names=["record_relative_path", "identity_label", "race_label"],
        dtype={
            "record_relative_path": "string",
            "identity_label": "int64",
            "race_label": "int64",
        },
        keep_default_na=False,
    )
    if frame.empty:
        raise DatasetIntegrityError("BalancedFace list is empty")
    if frame.isna().any().any():
        raise DatasetIntegrityError("BalancedFace list contains missing values")
    frame["record_relative_path"] = frame["record_relative_path"].map(
        _validate_list_path
    )
    if frame["record_relative_path"].duplicated().any():
        raise DatasetIntegrityError("BalancedFace list contains duplicate image paths")
    unknown_labels = sorted(set(frame["race_label"]).difference(BALANCEDFACE_RACE_LABELS))
    if unknown_labels:
        raise DatasetIntegrityError(
            f"unknown BalancedFace race labels: {unknown_labels}"
        )

    frame.insert(0, "source_row_index", range(len(frame)))
    frame["source_identity_id"] = frame["record_relative_path"].str.split(
        "/", n=1
    ).str[0]
    frame["dataset_group"] = frame["race_label"].map(BALANCEDFACE_RACE_LABELS)

    identity_consistency = frame.groupby("source_identity_id").agg(
        identity_label_count=("identity_label", "nunique"),
        group_count=("dataset_group", "nunique"),
    )
    inconsistent = identity_consistency.loc[
        (identity_consistency["identity_label_count"] != 1)
        | (identity_consistency["group_count"] != 1)
    ]
    if not inconsistent.empty:
        raise DatasetIntegrityError(
            "BalancedFace source identity maps to multiple labels/groups: "
            f"{inconsistent.index[:3].tolist()}"
        )
    label_identity_counts = frame.groupby("identity_label")[
        "source_identity_id"
    ].nunique()
    if (label_identity_counts != 1).any():
        raise DatasetIntegrityError(
            "BalancedFace numeric identity label maps to multiple source identities"
        )

    if strict_official:
        _validate_official_index(frame)

    overlap_ids = sorted(set(frame["source_identity_id"]).intersection(rfw_ids))
    excluded = (
        frame.loc[frame["source_identity_id"].isin(overlap_ids)]
        .groupby(
            [
                "source_identity_id",
                "dataset_group",
                "identity_label",
                "race_label",
            ],
            as_index=False,
            observed=True,
        )
        .agg(excluded_image_count=("source_row_index", "size"))
        .sort_values(["dataset_group", "source_identity_id"])
        .reset_index(drop=True)
    )
    filtered = frame.loc[~frame["source_identity_id"].isin(overlap_ids)].copy()

    identity_groups = filtered[
        ["source_identity_id", "dataset_group"]
    ].drop_duplicates()
    split_by_identity = _identity_splits(
        identity_groups,
        seed=seed,
        development_fraction=development_fraction,
    )
    filtered["split"] = filtered["source_identity_id"].map(split_by_identity)
    filtered["protocol_role"] = filtered["split"].map(
        {
            "development": "compression_fit",
            "calibration": "threshold_calibration",
        }
    )
    filtered["identity_id"] = (
        "balancedface:" + filtered["source_identity_id"].astype(str)
    )
    filtered["image_id"] = filtered["record_relative_path"].map(
        lambda value: "balancedface:"
        + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    )
    filtered["dataset"] = "bupt-balancedface-equalizedface"
    filtered["dataset_role"] = "development_and_calibration_only"
    filtered["dataset_group_source"] = "dataset_provided"
    filtered["storage_kind"] = "mxnet_recordio_pending_decoder"
    filtered["source_list_path"] = _relative_posix(source, project)
    filtered["source_archive_path"] = _relative_posix(recordio_archive, project)

    manifest_columns = [
        "image_id",
        "identity_id",
        "source_identity_id",
        "split",
        "dataset",
        "dataset_role",
        "protocol_role",
        "dataset_group",
        "dataset_group_source",
        "identity_label",
        "race_label",
        "source_row_index",
        "record_relative_path",
        "storage_kind",
        "source_list_path",
        "source_archive_path",
    ]
    manifest = filtered[manifest_columns].sort_values(
        ["split", "dataset_group", "identity_id", "source_row_index"]
    ).reset_index(drop=True)
    if manifest["image_id"].duplicated().any():
        raise DatasetIntegrityError("BalancedFace image IDs are not unique")
    if manifest.groupby("identity_id")["split"].nunique().max() != 1:
        raise DatasetIntegrityError(
            "BalancedFace identities must be disjoint across development/calibration"
        )

    split_identity_counts = (
        manifest.groupby("split")["identity_id"].nunique().astype(int).to_dict()
    )
    split_image_counts = manifest["split"].value_counts().astype(int).to_dict()
    summary = {
        "dataset": "bupt-balancedface-equalizedface",
        "role": "development_and_calibration_only",
        "final_evaluation_allowed": False,
        "source_representation": "mxnet_recordio_aligned",
        "recordio_decoder_implemented": False,
        "recordio_archive_integrity_verified": bool(verify_recordio_archive),
        "recordio_archive_readable_to_eof": (
            bool(recordio_report.readable_to_eof)
            if recordio_report is not None
            else False
        ),
        "recordio_property": recordio_property,
        "recordio_member_bytes": recordio_members,
        "source_row_index_is_recordio_key": False,
        "materialized_image_manifest_ready": False,
        "strict_official_index_validated": bool(strict_official),
        "full_source_index": True,
        "seed": int(seed),
        "development_fraction": float(development_fraction),
        "source_image_count": int(len(frame)),
        "source_identity_count": int(frame["source_identity_id"].nunique()),
        "image_count_after_rfw_overlap_exclusion": int(len(manifest)),
        "identity_count_after_rfw_overlap_exclusion": int(
            manifest["identity_id"].nunique()
        ),
        "excluded_rfw_overlap_identity_count": int(len(overlap_ids)),
        "excluded_rfw_overlap_image_count": int(
            excluded["excluded_image_count"].sum() if not excluded.empty else 0
        ),
        "development_identity_count": int(
            split_identity_counts.get("development", 0)
        ),
        "calibration_identity_count": int(
            split_identity_counts.get("calibration", 0)
        ),
        "development_image_count": int(split_image_counts.get("development", 0)),
        "calibration_image_count": int(split_image_counts.get("calibration", 0)),
        "group_image_counts": {
            str(key): int(value)
            for key, value in manifest["dataset_group"].value_counts().to_dict().items()
        },
        "group_identity_counts": {
            str(key): int(value)
            for key, value in manifest.groupby("dataset_group", observed=True)[
                "identity_id"
            ]
            .nunique()
            .to_dict()
            .items()
        },
        "race_label_mapping": {
            str(key): value for key, value in BALANCEDFACE_RACE_LABELS.items()
        },
        "race_label_mapping_evidence": (
            "labels 0/1/2 were cross-checked against named folders in the local "
            "JPG archive; label 3 is the remaining African category"
        ),
        "protocol_note": (
            "Use development rows only for PCA/PQ fitting and calibration rows "
            "only for threshold calibration. This dataset has no final test split. "
            "The list row index must not be treated as a RecordIO key; a decoder "
            "must validate train.idx/train.rec linkage before materialization."
        ),
    }
    return BalancedFaceIndexBundle(
        manifest=manifest,
        excluded_identities=excluded,
        summary=summary,
    )


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_balancedface_index_bundle(
    bundle: BalancedFaceIndexBundle,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write leakage-audited metadata; no images or raw archives are changed."""

    directory = Path(output_dir).expanduser().resolve()
    targets = {
        "source_index_manifest.csv": directory / "source_index_manifest.csv",
        "excluded_rfw_overlap_identities.csv": (
            directory / "excluded_rfw_overlap_identities.csv"
        ),
        "summary.json": directory / "summary.json",
        "_SUCCESS": directory / "_SUCCESS",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite BalancedFace outputs: "
            + ", ".join(str(path) for path in existing)
        )
    directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        targets["_SUCCESS"].unlink(missing_ok=True)
    _atomic_write_csv(targets["source_index_manifest.csv"], bundle.manifest)
    _atomic_write_csv(
        targets["excluded_rfw_overlap_identities.csv"],
        bundle.excluded_identities,
    )
    _atomic_write_json(targets["summary.json"], bundle.summary)
    _atomic_write_json(
        targets["_SUCCESS"],
        {
            "dataset": "bupt-balancedface-equalizedface",
            "rfw_overlap_audit_identity_count": bundle.summary.get(
                "excluded_rfw_overlap_identity_count"
            ),
            "recordio_archive_integrity_verified": bundle.summary.get(
                "recordio_archive_integrity_verified"
            ),
        },
    )
    return targets


def select_balancedface_index_scope(
    bundle: BalancedFaceIndexBundle,
    *,
    mode: str,
    data_fraction: float,
    seed: int,
) -> BalancedFaceIndexBundle:
    """Select group/split-stratified identities for a non-paper dev run."""

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"dev", "real"}:
        raise ValueError("mode must be 'dev' or 'real'")
    if not 0.0 < data_fraction <= 1.0:
        raise ValueError("data_fraction must be in (0, 1]")
    if normalized_mode == "real" and data_fraction != 1.0:
        raise ValueError("BalancedFace real mode requires data_fraction=1.0")

    if normalized_mode == "real":
        expected_image_count = int(
            bundle.summary["image_count_after_rfw_overlap_exclusion"]
        )
        expected_identity_count = int(
            bundle.summary["identity_count_after_rfw_overlap_exclusion"]
        )
        if (
            not bundle.summary.get("strict_official_index_validated", False)
            or not bundle.summary.get("full_source_index", False)
            or len(bundle.manifest) != expected_image_count
            or bundle.manifest["identity_id"].nunique() != expected_identity_count
        ):
            raise DatasetIntegrityError(
                "BalancedFace real mode requires the full strict source index"
            )
        return BalancedFaceIndexBundle(
            manifest=bundle.manifest.copy(),
            excluded_identities=bundle.excluded_identities.copy(),
            summary={
                **bundle.summary,
                "scope_mode": "real",
                "data_fraction": 1.0,
                "scope_seed": int(seed),
                "full_source_index": True,
            },
        )

    identity_rows = bundle.manifest[
        ["source_identity_id", "dataset_group", "split"]
    ].drop_duplicates()
    selected_identities: set[str] = set()
    for (group, split), stratum in identity_rows.groupby(
        ["dataset_group", "split"],
        sort=True,
        observed=True,
    ):
        identities = stratum["source_identity_id"].astype(str).tolist()
        identities.sort(
            key=lambda identity: _stable_split_key(
                seed,
                f"{group}:{split}",
                identity,
            )
        )
        selected_count = max(1, int(len(identities) * data_fraction))
        selected_identities.update(identities[:selected_count])
    manifest = bundle.manifest.loc[
        bundle.manifest["source_identity_id"].isin(selected_identities)
    ].reset_index(drop=True)
    split_identity_counts = (
        manifest.groupby("split")["identity_id"].nunique().astype(int).to_dict()
    )
    split_image_counts = manifest["split"].value_counts().astype(int).to_dict()
    return BalancedFaceIndexBundle(
        manifest=manifest,
        excluded_identities=bundle.excluded_identities.copy(),
        summary={
            **bundle.summary,
            "source_strict_official_index_validated": bool(
                bundle.summary.get("strict_official_index_validated", False)
            ),
            "strict_official_index_validated": False,
            "scope_mode": "dev",
            "data_fraction": float(data_fraction),
            "scope_seed": int(seed),
            "full_source_index": False,
            "selected_identity_count": int(manifest["identity_id"].nunique()),
            "selected_image_count": int(len(manifest)),
            "image_count_after_rfw_overlap_exclusion": int(len(manifest)),
            "identity_count_after_rfw_overlap_exclusion": int(
                manifest["identity_id"].nunique()
            ),
            "development_identity_count": int(
                split_identity_counts.get("development", 0)
            ),
            "calibration_identity_count": int(
                split_identity_counts.get("calibration", 0)
            ),
            "development_image_count": int(
                split_image_counts.get("development", 0)
            ),
            "calibration_image_count": int(
                split_image_counts.get("calibration", 0)
            ),
            "group_image_counts": {
                str(key): int(value)
                for key, value in manifest["dataset_group"]
                .value_counts()
                .to_dict()
                .items()
            },
            "group_identity_counts": {
                str(key): int(value)
                for key, value in manifest.groupby(
                    "dataset_group", observed=True
                )["identity_id"]
                .nunique()
                .to_dict()
                .items()
            },
            "selection_strata": ["dataset_group", "split"],
        },
    )
