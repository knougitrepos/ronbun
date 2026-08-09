from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd

from research.datasets.rfw import RFW_GROUPS
from research.protocols.open_set import (
    OpenSetProtocol,
    validate_identity_disjoint_splits,
)


RFW_CUSTOM_DATASET_ID = "rfw-v1"
RFW_CUSTOM_PROTOCOL_FAMILY_UID = "rfw-custom-identity-balanced-open-set-v1"
RFW_CUSTOM_ARTIFACT_TYPE = "rfw_custom_open_set_protocol_bundle_v1"
RFW_OFFICIAL_PROTOCOL_UID = "rfw_official_groupwise_10fold_verification"

_SOURCE_REQUIRED_COLUMNS = {
    "image_id",
    "identity_id",
    "source_identity_id",
    "split",
    "image_path",
    "dataset",
    "dataset_role",
    "protocol_role",
    "rfw_group",
    "group_label_source",
    "source_label",
    "face_index",
    "protocol_index",
    "source_archive_path",
    "archive_member",
}
_CUSTOM_ROLES = (
    "development_pool",
    "calibration_pool",
    "gallery",
    "registered_probe",
    "known_unknown_probe",
    "unknown_unknown_probe",
)
_CUSTOM_REQUIRED_COLUMNS = _SOURCE_REQUIRED_COLUMNS.union(
    {
        "source_dataset_role",
        "source_split",
        "source_protocol_role",
        "source_protocol_index",
        "demographic_group",
        "demographic_group_source",
        "probe_type",
        "is_mated",
        "protocol_kind",
        "protocol_uid",
        "protocol_family_uid",
        "artifact_type",
        "official_pair_protocol_used",
        "official_result_eligible",
        "checkpoint_overlap_status",
        "strict_unseen_identity_evidence",
    }
)


@dataclass(frozen=True)
class RFWCustomOpenSetBundle:
    """Derived RFW 1:N protocol, explicitly separate from official pairs."""

    manifest: pd.DataFrame
    protocol: OpenSetProtocol
    gallery_identities: tuple[str, ...]
    known_unknown_identities: tuple[str, ...]
    unknown_unknown_identities: tuple[str, ...]
    non_mated_identities: tuple[str, ...]
    summary: dict[str, Any]


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    try:
        finite = math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not finite or integer != value or integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _group_counts(
    value: int | Mapping[str, int],
) -> dict[str, int]:
    if isinstance(value, Mapping):
        keys = {str(key) for key in value}
        expected = set(RFW_GROUPS)
        if keys != expected:
            raise ValueError(
                "gallery_identity_count_per_group keys must be exactly "
                f"{sorted(expected)}, got {sorted(keys)}"
            )
        return {
            group: _positive_integer(
                value[group],
                name=f"gallery_identity_count_per_group[{group!r}]",
            )
            for group in RFW_GROUPS
        }
    count = _positive_integer(
        value,
        name="gallery_identity_count_per_group",
    )
    return {group: count for group in RFW_GROUPS}


def _open_unit_fraction(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be between zero and one")
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be between zero and one") from exc
    if not math.isfinite(resolved) or not 0.0 < resolved < 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return resolved


def _sha256_text(value: object, *, name: str) -> str:
    resolved = str(value).strip().upper()
    if len(resolved) != 64 or any(
        character not in "0123456789ABCDEF" for character in resolved
    ):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return resolved


def _stable_key(*, seed: int, namespace: str, value: str) -> str:
    payload = f"{namespace}\x1f{seed}\x1f{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _logical_frame_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(columns).encode("utf-8"))
    digest.update(b"\x1e")
    ordered = frame.sort_values(columns, kind="stable").loc[:, columns]
    for row in ordered.itertuples(index=False, name=None):
        digest.update("\x1f".join(str(value) for value in row).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _normalized_source_manifest(source_manifest: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(source_manifest, pd.DataFrame):
        raise TypeError("source_manifest must be a pandas DataFrame")
    missing = sorted(_SOURCE_REQUIRED_COLUMNS.difference(source_manifest.columns))
    if missing:
        raise ValueError(f"RFW source manifest is missing columns: {missing}")
    if source_manifest.empty:
        raise ValueError("RFW source manifest must not be empty")

    source = source_manifest.copy()
    required = sorted(_SOURCE_REQUIRED_COLUMNS)
    if source[required].isna().any().any():
        raise ValueError("RFW source manifest required columns contain missing values")
    text_columns = (
        "image_id",
        "identity_id",
        "source_identity_id",
        "split",
        "image_path",
        "dataset",
        "dataset_role",
        "protocol_role",
        "rfw_group",
        "group_label_source",
        "source_archive_path",
        "archive_member",
    )
    for column in text_columns:
        source[column] = source[column].astype(str)
        if source[column].str.strip().eq("").any():
            raise ValueError(f"RFW source manifest {column} contains empty values")

    if source["image_id"].duplicated().any():
        raise ValueError("RFW source manifest image_id values must be unique")
    if source["image_path"].duplicated().any():
        raise ValueError("RFW source manifest image_path values must be unique")
    if set(source["rfw_group"]) != set(RFW_GROUPS):
        raise ValueError(
            f"RFW source manifest demographic groups must be exactly {list(RFW_GROUPS)}"
        )
    expected_values = {
        "split": {"test"},
        "dataset": {RFW_CUSTOM_DATASET_ID},
        "dataset_role": {"evaluation_test_only"},
        "protocol_role": {"verification_image"},
        "group_label_source": {"dataset_provided"},
    }
    for column, expected in expected_values.items():
        actual = set(source[column])
        if actual != expected:
            raise ValueError(
                f"RFW source manifest {column} must be {sorted(expected)}, "
                f"got {sorted(actual)}"
            )

    indexes = pd.to_numeric(source["protocol_index"], errors="coerce")
    if (
        indexes.isna().any()
        or not indexes.map(lambda value: float(value).is_integer()).all()
    ):
        raise ValueError("RFW source protocol_index must contain integers")
    source["protocol_index"] = indexes.astype("int64")
    for group, rows in source.groupby("rfw_group", sort=True):
        actual = sorted(rows["protocol_index"].tolist())
        if actual != list(range(len(rows))):
            raise ValueError(
                f"RFW source protocol_index for {group} must be unique and "
                "contiguous from zero"
            )

    identity_group_counts = source.groupby("identity_id")["rfw_group"].nunique()
    if (identity_group_counts != 1).any():
        raise ValueError("RFW source identity_id maps to multiple demographic groups")
    identity_source_counts = source.groupby("identity_id")[
        "source_identity_id"
    ].nunique()
    if (identity_source_counts != 1).any():
        raise ValueError("RFW source identity_id maps to multiple source identities")
    local_identity_counts = source.groupby(["rfw_group", "source_identity_id"])[
        "identity_id"
    ].nunique()
    if (local_identity_counts != 1).any():
        raise ValueError(
            "RFW source group/source_identity_id maps to multiple identity_id values"
        )
    label_counts = source.groupby("identity_id")["source_label"].nunique()
    if (label_counts != 1).any():
        raise ValueError("RFW source identity_id maps to multiple source labels")
    return source


def _role_frame(
    rows: pd.DataFrame,
    *,
    role: str,
    protocol_uid: str,
) -> pd.DataFrame:
    if role not in _CUSTOM_ROLES:
        raise ValueError(f"unknown RFW custom role: {role}")
    frame = rows.copy()
    frame["source_dataset_role"] = frame["dataset_role"]
    frame["source_split"] = frame["split"]
    frame["source_protocol_role"] = frame["protocol_role"]
    frame["source_protocol_index"] = frame["protocol_index"].astype("int64")
    split_by_role = {
        "development_pool": "development",
        "calibration_pool": "calibration",
        "gallery": "test",
        "registered_probe": "test",
        "known_unknown_probe": "test",
        "unknown_unknown_probe": "test",
    }
    dataset_role_by_role = {
        "development_pool": "compressor_fit_development",
        "calibration_pool": "threshold_calibration",
        "gallery": "evaluation_test_custom_protocol",
        "registered_probe": "evaluation_test_custom_protocol",
        "known_unknown_probe": "evaluation_test_custom_protocol",
        "unknown_unknown_probe": "evaluation_test_custom_protocol",
    }
    frame["split"] = split_by_role[role]
    frame["dataset_role"] = dataset_role_by_role[role]
    frame["protocol_role"] = role
    frame["demographic_group"] = frame["rfw_group"]
    frame["demographic_group_source"] = "dataset_provided"
    frame["protocol_kind"] = "custom_1_to_n_open_set_identification"
    frame["protocol_uid"] = protocol_uid
    frame["protocol_family_uid"] = RFW_CUSTOM_PROTOCOL_FAMILY_UID
    frame["artifact_type"] = RFW_CUSTOM_ARTIFACT_TYPE
    frame["official_pair_protocol_used"] = False
    frame["official_result_eligible"] = False
    frame["checkpoint_overlap_status"] = "UNKNOWN"
    frame["strict_unseen_identity_evidence"] = False
    if role in {"development_pool", "calibration_pool", "gallery"}:
        frame["probe_type"] = "not_applicable"
        frame["is_mated"] = pd.array([pd.NA] * len(frame), dtype="boolean")
    elif role == "registered_probe":
        frame["probe_type"] = "registered"
        frame["is_mated"] = pd.array([True] * len(frame), dtype="boolean")
    elif role == "known_unknown_probe":
        frame["probe_type"] = "known_unknown"
        frame["is_mated"] = pd.array([False] * len(frame), dtype="boolean")
    else:
        frame["probe_type"] = "unknown_unknown"
        frame["is_mated"] = pd.array([False] * len(frame), dtype="boolean")
    frame = frame.sort_values(
        ["rfw_group", "identity_id", "image_id"], kind="stable"
    ).reset_index(drop=True)
    frame["protocol_index"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _coerce_boolean(value: object, *, column: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    raise ValueError(f"RFW custom {column} must contain booleans")


def adapt_rfw_custom_manifest_to_open_set_protocol(
    manifest: pd.DataFrame,
    *,
    expected_protocol_uid: str | None = None,
) -> OpenSetProtocol:
    """Validate a persisted custom manifest and recover its open-set roles."""

    if not isinstance(manifest, pd.DataFrame):
        raise TypeError("manifest must be a pandas DataFrame")
    missing = sorted(_CUSTOM_REQUIRED_COLUMNS.difference(manifest.columns))
    if missing:
        raise ValueError(f"RFW custom manifest is missing columns: {missing}")
    if manifest.empty:
        raise ValueError("RFW custom manifest must not be empty")
    frame = manifest.copy()
    non_nullable = sorted(_CUSTOM_REQUIRED_COLUMNS.difference({"is_mated"}))
    if frame[non_nullable].isna().any().any():
        raise ValueError("RFW custom required columns contain missing values")
    for column in (
        "image_id",
        "identity_id",
        "image_path",
        "dataset",
        "split",
        "rfw_group",
        "demographic_group",
        "protocol_role",
        "probe_type",
        "protocol_kind",
        "protocol_uid",
        "protocol_family_uid",
        "artifact_type",
        "source_protocol_role",
        "source_split",
        "checkpoint_overlap_status",
    ):
        frame[column] = frame[column].astype(str)

    if frame["image_id"].duplicated().any():
        raise ValueError("RFW custom image_id values must be unique")
    if frame["image_path"].duplicated().any():
        raise ValueError("RFW custom image_path values must be unique")
    singleton_contract = {
        "dataset": RFW_CUSTOM_DATASET_ID,
        "protocol_kind": "custom_1_to_n_open_set_identification",
        "protocol_family_uid": RFW_CUSTOM_PROTOCOL_FAMILY_UID,
        "artifact_type": RFW_CUSTOM_ARTIFACT_TYPE,
        "source_protocol_role": "verification_image",
        "source_split": "test",
        "checkpoint_overlap_status": "UNKNOWN",
    }
    for column, expected in singleton_contract.items():
        actual = set(frame[column])
        if actual != {expected}:
            raise ValueError(
                f"RFW custom {column} must be {expected!r}, got {sorted(actual)}"
            )
    protocol_uids = set(frame["protocol_uid"])
    if len(protocol_uids) != 1:
        raise ValueError("RFW custom manifest must contain exactly one protocol_uid")
    protocol_uid = next(iter(protocol_uids))
    if not protocol_uid.startswith(f"{RFW_CUSTOM_PROTOCOL_FAMILY_UID}-"):
        raise ValueError("RFW custom protocol_uid is outside the custom namespace")
    if protocol_uid == RFW_OFFICIAL_PROTOCOL_UID:
        raise ValueError("RFW custom protocol_uid collides with the official protocol")
    if expected_protocol_uid is not None and protocol_uid != str(expected_protocol_uid):
        raise ValueError(
            "RFW custom protocol_uid mismatch: "
            f"expected={expected_protocol_uid!r}, actual={protocol_uid!r}"
        )
    if set(frame["protocol_role"]) != set(_CUSTOM_ROLES):
        raise ValueError(f"RFW custom roles must be exactly {list(_CUSTOM_ROLES)}")
    if set(frame["rfw_group"]) != set(RFW_GROUPS):
        raise ValueError("RFW custom manifest is missing demographic groups")
    if not frame["rfw_group"].eq(frame["demographic_group"]).all():
        raise ValueError("RFW custom demographic_group must equal rfw_group")
    if set(frame["demographic_group_source"].astype(str)) != {"dataset_provided"}:
        raise ValueError("RFW custom demographic group source must be dataset-provided")

    for column in (
        "official_pair_protocol_used",
        "official_result_eligible",
        "strict_unseen_identity_evidence",
    ):
        values = frame[column].map(lambda value: _coerce_boolean(value, column=column))
        if values.any():
            raise ValueError(f"RFW custom {column} must be false")

    expected_probe_types = {
        "development_pool": "not_applicable",
        "calibration_pool": "not_applicable",
        "gallery": "not_applicable",
        "registered_probe": "registered",
        "known_unknown_probe": "known_unknown",
        "unknown_unknown_probe": "unknown_unknown",
    }
    expected_splits = {
        "development_pool": "development",
        "calibration_pool": "calibration",
        "gallery": "test",
        "registered_probe": "test",
        "known_unknown_probe": "test",
        "unknown_unknown_probe": "test",
    }
    expected_dataset_roles = {
        "development_pool": "compressor_fit_development",
        "calibration_pool": "threshold_calibration",
        "gallery": "evaluation_test_custom_protocol",
        "registered_probe": "evaluation_test_custom_protocol",
        "known_unknown_probe": "evaluation_test_custom_protocol",
        "unknown_unknown_probe": "evaluation_test_custom_protocol",
    }
    role_frames: dict[str, pd.DataFrame] = {}
    for role in _CUSTOM_ROLES:
        rows = frame.loc[frame["protocol_role"].eq(role)].copy()
        if set(rows["rfw_group"]) != set(RFW_GROUPS):
            raise ValueError(f"RFW custom role {role} is missing demographic groups")
        indexes = pd.to_numeric(rows["protocol_index"], errors="coerce")
        if (
            indexes.isna().any()
            or not indexes.map(lambda value: float(value).is_integer()).all()
        ):
            raise ValueError(f"RFW custom {role} protocol_index must be integers")
        if sorted(indexes.astype(int).tolist()) != list(range(len(rows))):
            raise ValueError(
                f"RFW custom {role} protocol_index must be unique and "
                "contiguous from zero"
            )
        if set(rows["probe_type"]) != {expected_probe_types[role]}:
            raise ValueError(f"RFW custom {role} has an invalid probe_type")
        if set(rows["split"].astype(str)) != {expected_splits[role]}:
            raise ValueError(f"RFW custom {role} has an invalid split")
        if set(rows["dataset_role"].astype(str)) != {expected_dataset_roles[role]}:
            raise ValueError(f"RFW custom {role} has an invalid dataset_role")
        if role in {"development_pool", "calibration_pool", "gallery"}:
            if not rows["is_mated"].isna().all():
                raise ValueError(f"RFW custom {role} is_mated must be missing")
        else:
            expected_mated = role == "registered_probe"
            mated = rows["is_mated"].map(
                lambda value: _coerce_boolean(value, column="is_mated")
            )
            if not mated.eq(expected_mated).all():
                raise ValueError(f"RFW custom {role} has invalid is_mated values")
        role_frames[role] = rows.sort_values(
            "protocol_index", kind="stable"
        ).reset_index(drop=True)

    gallery_ids = set(role_frames["gallery"]["identity_id"])
    registered_ids = set(role_frames["registered_probe"]["identity_id"])
    known_unknown_ids = set(role_frames["known_unknown_probe"]["identity_id"])
    non_mated_ids = set(role_frames["unknown_unknown_probe"]["identity_id"])
    if gallery_ids != registered_ids:
        raise ValueError("RFW custom gallery and registered identity sets differ")
    test_identity_sets = {
        "gallery": gallery_ids,
        "known_unknown": known_unknown_ids,
        "unknown_unknown": non_mated_ids,
    }
    names = list(test_identity_sets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            if test_identity_sets[left_name].intersection(
                test_identity_sets[right_name]
            ):
                raise ValueError(
                    f"RFW custom test identity roles overlap: {left_name}/{right_name}"
                )
    identity_groups = frame.groupby("identity_id")["rfw_group"].nunique()
    if (identity_groups != 1).any():
        raise ValueError("RFW custom identity maps to multiple demographic groups")
    if set(frame["source_dataset_role"].astype(str)) != {"evaluation_test_only"}:
        raise ValueError("RFW custom source_dataset_role is invalid")
    source_indexes = pd.to_numeric(frame["source_protocol_index"], errors="coerce")
    if (
        source_indexes.isna().any()
        or not source_indexes.map(
            lambda value: float(value).is_integer() and float(value) >= 0.0
        ).all()
    ):
        raise ValueError(
            "RFW custom source_protocol_index must be non-negative integers"
        )
    source_index_frame = frame.assign(_source_index=source_indexes.astype(int))
    if source_index_frame.duplicated(["rfw_group", "_source_index"]).any():
        raise ValueError("RFW custom source protocol indexes are duplicated")

    validate_identity_disjoint_splits(
        frame.loc[:, ["image_id", "identity_id", "split", "image_path"]]
    )
    return OpenSetProtocol(
        gallery=role_frames["gallery"],
        registered_probes=role_frames["registered_probe"],
        known_unknown_probes=role_frames["known_unknown_probe"],
        unknown_unknown_probes=role_frames["unknown_unknown_probe"],
    )


def select_rfw_custom_protocol_fraction(
    bundle_or_manifest: RFWCustomOpenSetBundle | pd.DataFrame,
    data_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Select a nested quick-tier identity fraction within every group/role.

    Gallery and registered probes share one identity selection. Development,
    calibration, known-unknown, and unknown-unknown identities are selected in
    separate demographic-group namespaces. A full fraction returns a plain
    copy. Reduced outputs retain ``source_custom_protocol_index`` and rebase
    ``protocol_index`` independently for every role.
    """

    if isinstance(bundle_or_manifest, RFWCustomOpenSetBundle):
        frame = bundle_or_manifest.manifest
    elif isinstance(bundle_or_manifest, pd.DataFrame):
        frame = bundle_or_manifest
    else:
        raise TypeError(
            "bundle_or_manifest must be an RFWCustomOpenSetBundle or DataFrame"
        )
    adapt_rfw_custom_manifest_to_open_set_protocol(frame)
    if isinstance(data_fraction, (bool, np.bool_)):
        raise ValueError("data_fraction must be in (0, 1]")
    try:
        fraction = float(data_fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("data_fraction must be in (0, 1]") from exc
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("data_fraction must be in (0, 1]")
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    try:
        resolved_seed = int(seed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("seed must be an integer") from exc
    if resolved_seed != seed:
        raise ValueError("seed must be an integer")
    if fraction == 1.0:
        return frame.copy()

    selected_identity_sets: dict[tuple[str, str], set[str]] = {}
    independent_roles = (
        "development_pool",
        "calibration_pool",
        "known_unknown_probe",
        "unknown_unknown_probe",
    )
    for role in independent_roles:
        role_rows = frame.loc[frame["protocol_role"].astype(str).eq(role)]
        for group in RFW_GROUPS:
            identities = sorted(
                set(
                    role_rows.loc[
                        role_rows["rfw_group"].astype(str).eq(group),
                        "identity_id",
                    ].astype(str)
                ),
                key=lambda identity_id: (
                    _stable_key(
                        seed=resolved_seed,
                        namespace=(
                            f"{RFW_CUSTOM_PROTOCOL_FAMILY_UID}:quick-tier:"
                            f"{role}:{group}"
                        ),
                        value=identity_id,
                    ),
                    identity_id,
                ),
            )
            if not identities:
                raise ValueError(
                    f"RFW custom {role}/{group} has no identities to select"
                )
            count = max(1, math.ceil(len(identities) * fraction))
            selected_identity_sets[(role, group)] = set(identities[:count])

    gallery_rows = frame.loc[frame["protocol_role"].astype(str).eq("gallery")]
    for group in RFW_GROUPS:
        identities = sorted(
            set(
                gallery_rows.loc[
                    gallery_rows["rfw_group"].astype(str).eq(group),
                    "identity_id",
                ].astype(str)
            ),
            key=lambda identity_id: (
                _stable_key(
                    seed=resolved_seed,
                    namespace=(
                        f"{RFW_CUSTOM_PROTOCOL_FAMILY_UID}:quick-tier:"
                        f"gallery-registered:{group}"
                    ),
                    value=identity_id,
                ),
                identity_id,
            ),
        )
        if not identities:
            raise ValueError(f"RFW custom gallery/{group} has no identities")
        count = max(1, math.ceil(len(identities) * fraction))
        shared = set(identities[:count])
        selected_identity_sets[("gallery", group)] = shared
        selected_identity_sets[("registered_probe", group)] = shared

    selection = np.zeros(len(frame), dtype=bool)
    rows = frame.reset_index(drop=True)
    for (role, group), identities in selected_identity_sets.items():
        selection |= (
            rows["protocol_role"].astype(str).eq(role)
            & rows["rfw_group"].astype(str).eq(group)
            & rows["identity_id"].astype(str).isin(identities)
        ).to_numpy(dtype=bool)
    selected = rows.loc[selection].copy()
    selected["source_custom_protocol_index"] = selected["protocol_index"]
    selected["scope_data_fraction"] = fraction
    selected["scope_seed"] = resolved_seed
    selected["scope_is_full"] = False
    role_order = {role: index for index, role in enumerate(_CUSTOM_ROLES)}
    selected["_role_order"] = selected["protocol_role"].map(role_order)
    selected = selected.sort_values(
        ["_role_order", "source_custom_protocol_index"], kind="stable"
    ).drop(columns="_role_order")
    for role in _CUSTOM_ROLES:
        role_indexes = selected.index[selected["protocol_role"].astype(str).eq(role)]
        selected.loc[role_indexes, "protocol_index"] = range(len(role_indexes))
    selected["protocol_index"] = pd.to_numeric(selected["protocol_index"]).astype(
        "int64"
    )
    selected = selected.reset_index(drop=True)
    adapt_rfw_custom_manifest_to_open_set_protocol(selected)
    return selected


def build_rfw_custom_open_set_bundle(
    source_manifest: pd.DataFrame,
    *,
    source_archive_sha256: str,
    gallery_identity_count_per_group: int | Mapping[str, int],
    enrollment_count: int = 1,
    seed: int = 42,
    development_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
    unknown_unknown_fraction: float = 0.50,
) -> RFWCustomOpenSetBundle:
    """Derive a deterministic, demographic-aware RFW custom 1:N protocol.

    The source is the official RFW image manifest, but official pair rows and
    folds are never consumed. Within each demographic group, source identities
    are deterministically divided into development, calibration, and test.
    Gallery and non-gallery roles are then derived only from test identities.
    Source identity strings that occur in more than one RFW group are excluded
    before splitting to prevent hidden leakage across group-prefixed IDs.
    """

    source = _normalized_source_manifest(source_manifest)
    archive_sha256 = _sha256_text(
        source_archive_sha256,
        name="source_archive_sha256",
    )
    counts = _group_counts(gallery_identity_count_per_group)
    enrollment = _positive_integer(enrollment_count, name="enrollment_count")
    development_ratio = _open_unit_fraction(
        development_fraction,
        name="development_fraction",
    )
    calibration_ratio = _open_unit_fraction(
        calibration_fraction,
        name="calibration_fraction",
    )
    if development_ratio + calibration_ratio >= 1.0:
        raise ValueError(
            "development_fraction and calibration_fraction must leave a test split"
        )
    unknown_ratio = _open_unit_fraction(
        unknown_unknown_fraction,
        name="unknown_unknown_fraction",
    )
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    try:
        resolved_seed = int(seed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("seed must be an integer") from exc
    if resolved_seed != seed:
        raise ValueError("seed must be an integer")

    archive_paths = set(source["source_archive_path"].astype(str))
    if len(archive_paths) != 1:
        raise ValueError(
            "RFW source manifest must reference exactly one source archive"
        )
    source_archive_path = next(iter(archive_paths))

    source_contract_columns = [
        "image_id",
        "identity_id",
        "source_identity_id",
        "image_path",
        "rfw_group",
        "source_label",
        "face_index",
        "protocol_index",
        "source_archive_path",
        "archive_member",
    ]
    source_manifest_sha256 = _logical_frame_sha256(source, source_contract_columns)
    source_group_counts = source.groupby("source_identity_id")["rfw_group"].nunique()
    ambiguous_source_identities = tuple(
        sorted(source_group_counts.loc[source_group_counts > 1].index.astype(str))
    )
    ambiguous_digest = hashlib.sha256(
        "\n".join(ambiguous_source_identities).encode("utf-8")
    ).hexdigest()
    eligible_source = source.loc[
        ~source["source_identity_id"].isin(ambiguous_source_identities)
    ].copy()
    if eligible_source.empty:
        raise ValueError("RFW custom source has no identities after leakage exclusion")

    development_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    gallery_parts: list[pd.DataFrame] = []
    registered_parts: list[pd.DataFrame] = []
    known_unknown_parts: list[pd.DataFrame] = []
    unknown_unknown_parts: list[pd.DataFrame] = []
    selected_gallery_ids: list[str] = []
    selected_known_unknown_ids: list[str] = []
    selected_unknown_unknown_ids: list[str] = []
    split_identity_counts_by_group: dict[str, dict[str, int]] = {}
    for group in RFW_GROUPS:
        group_rows = eligible_source.loc[eligible_source["rfw_group"].eq(group)].copy()
        all_group_ids = sorted(set(group_rows["identity_id"].astype(str)))
        all_group_ids.sort(
            key=lambda identity_id: (
                _stable_key(
                    seed=resolved_seed,
                    namespace=(
                        f"{RFW_CUSTOM_PROTOCOL_FAMILY_UID}:{group}:"
                        "development-calibration-test"
                    ),
                    value=identity_id,
                ),
                identity_id,
            )
        )
        development_count = int(len(all_group_ids) * development_ratio)
        calibration_count = int(len(all_group_ids) * calibration_ratio)
        test_count = len(all_group_ids) - development_count - calibration_count
        if development_count < 1 or calibration_count < 1 or test_count < 3:
            raise ValueError(
                f"RFW custom {group} split is too small: development="
                f"{development_count}, calibration={calibration_count}, "
                f"test={test_count}"
            )
        development_ids = set(all_group_ids[:development_count])
        calibration_end = development_count + calibration_count
        calibration_ids = set(all_group_ids[development_count:calibration_end])
        test_ids = set(all_group_ids[calibration_end:])
        if (
            development_ids.intersection(calibration_ids)
            or development_ids.intersection(test_ids)
            or calibration_ids.intersection(test_ids)
        ):
            raise RuntimeError("RFW custom identity split construction overlapped")
        development_parts.append(
            group_rows.loc[group_rows["identity_id"].isin(development_ids)]
        )
        calibration_parts.append(
            group_rows.loc[group_rows["identity_id"].isin(calibration_ids)]
        )
        test_rows = group_rows.loc[group_rows["identity_id"].isin(test_ids)].copy()
        split_identity_counts_by_group[group] = {
            "development": int(len(development_ids)),
            "calibration": int(len(calibration_ids)),
            "test": int(len(test_ids)),
        }

        sizes = test_rows.groupby("identity_id")["image_id"].nunique()
        eligible_gallery_ids = [
            str(identity_id)
            for identity_id, image_count in sizes.items()
            if int(image_count) > enrollment
        ]
        eligible_gallery_ids.sort(
            key=lambda identity_id: (
                _stable_key(
                    seed=resolved_seed,
                    namespace=f"{RFW_CUSTOM_PROTOCOL_FAMILY_UID}:{group}:gallery",
                    value=identity_id,
                ),
                identity_id,
            )
        )
        requested = counts[group]
        if len(eligible_gallery_ids) < requested:
            raise ValueError(
                f"RFW custom {group} has only {len(eligible_gallery_ids)} "
                f"identities with more than {enrollment} images; requested "
                f"{requested}"
            )
        gallery_ids = set(eligible_gallery_ids[:requested])
        non_gallery_ids = test_ids.difference(gallery_ids)
        if len(non_gallery_ids) < 2:
            raise ValueError(
                f"RFW custom {group} must reserve at least two non-gallery "
                "identities for known-unknown and unknown-unknown probes"
            )
        ordered_non_gallery_ids = sorted(
            non_gallery_ids,
            key=lambda identity_id: (
                _stable_key(
                    seed=resolved_seed,
                    namespace=(
                        f"{RFW_CUSTOM_PROTOCOL_FAMILY_UID}:{group}:"
                        "known-unknown-vs-unknown-unknown"
                    ),
                    value=identity_id,
                ),
                identity_id,
            ),
        )
        unknown_count = round(len(ordered_non_gallery_ids) * unknown_ratio)
        unknown_count = min(
            max(1, int(unknown_count)), len(ordered_non_gallery_ids) - 1
        )
        unknown_unknown_ids = set(ordered_non_gallery_ids[:unknown_count])
        known_unknown_ids = non_gallery_ids.difference(unknown_unknown_ids)
        selected_gallery_ids.extend(sorted(gallery_ids))
        selected_known_unknown_ids.extend(sorted(known_unknown_ids))
        selected_unknown_unknown_ids.extend(sorted(unknown_unknown_ids))

        for identity_id in sorted(gallery_ids):
            identity_rows = test_rows.loc[
                test_rows["identity_id"].eq(identity_id)
            ].copy()
            identity_rows["_selection_key"] = identity_rows["image_id"].map(
                lambda image_id: _stable_key(
                    seed=resolved_seed,
                    namespace=(
                        f"{RFW_CUSTOM_PROTOCOL_FAMILY_UID}:{group}:"
                        f"{identity_id}:enrollment"
                    ),
                    value=str(image_id),
                )
            )
            identity_rows = identity_rows.sort_values(
                ["_selection_key", "image_id"], kind="stable"
            ).drop(columns="_selection_key")
            gallery_parts.append(identity_rows.iloc[:enrollment])
            registered_parts.append(identity_rows.iloc[enrollment:])
        known_unknown_parts.append(
            test_rows.loc[test_rows["identity_id"].isin(known_unknown_ids)]
        )
        unknown_unknown_parts.append(
            test_rows.loc[test_rows["identity_id"].isin(unknown_unknown_ids)]
        )

    development_source = pd.concat(development_parts, ignore_index=True)
    calibration_source = pd.concat(calibration_parts, ignore_index=True)
    gallery_source = pd.concat(gallery_parts, ignore_index=True)
    registered_source = pd.concat(registered_parts, ignore_index=True)
    known_unknown_source = pd.concat(known_unknown_parts, ignore_index=True)
    unknown_unknown_source = pd.concat(unknown_unknown_parts, ignore_index=True)
    assignment = pd.concat(
        [
            development_source.assign(_custom_role="development_pool"),
            calibration_source.assign(_custom_role="calibration_pool"),
            gallery_source.assign(_custom_role="gallery"),
            registered_source.assign(_custom_role="registered_probe"),
            known_unknown_source.assign(_custom_role="known_unknown_probe"),
            unknown_unknown_source.assign(_custom_role="unknown_unknown_probe"),
        ],
        ignore_index=True,
    )
    if assignment["image_id"].duplicated().any():
        raise RuntimeError("RFW custom role assignment duplicated image rows")
    if set(assignment["image_id"]) != set(eligible_source["image_id"]):
        raise RuntimeError("RFW custom role assignment did not partition source rows")

    protocol_contract_sha256 = _logical_frame_sha256(
        assignment,
        ["_custom_role", "rfw_group", "identity_id", "image_id"],
    )
    uid_payload = {
        "family": RFW_CUSTOM_PROTOCOL_FAMILY_UID,
        "source_manifest_sha256": source_manifest_sha256,
        "protocol_contract_sha256": protocol_contract_sha256,
        "source_archive_sha256": archive_sha256,
        "seed": resolved_seed,
        "enrollment_count": enrollment,
        "development_fraction": development_ratio,
        "calibration_fraction": calibration_ratio,
        "unknown_unknown_fraction": unknown_ratio,
        "gallery_identity_count_by_group": counts,
        "excluded_cross_group_source_identity_sha256": ambiguous_digest,
    }
    uid_digest = hashlib.sha256(
        json.dumps(uid_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    protocol_uid = f"{RFW_CUSTOM_PROTOCOL_FAMILY_UID}-{uid_digest[:16]}"

    development = _role_frame(
        development_source,
        role="development_pool",
        protocol_uid=protocol_uid,
    )
    calibration = _role_frame(
        calibration_source,
        role="calibration_pool",
        protocol_uid=protocol_uid,
    )
    gallery = _role_frame(
        gallery_source,
        role="gallery",
        protocol_uid=protocol_uid,
    )
    registered = _role_frame(
        registered_source,
        role="registered_probe",
        protocol_uid=protocol_uid,
    )
    known_unknown = _role_frame(
        known_unknown_source,
        role="known_unknown_probe",
        protocol_uid=protocol_uid,
    )
    unknown_unknown = _role_frame(
        unknown_unknown_source,
        role="unknown_unknown_probe",
        protocol_uid=protocol_uid,
    )
    manifest = pd.concat(
        [
            development,
            calibration,
            gallery,
            registered,
            known_unknown,
            unknown_unknown,
        ],
        ignore_index=True,
        sort=False,
    )
    protocol = adapt_rfw_custom_manifest_to_open_set_protocol(
        manifest,
        expected_protocol_uid=protocol_uid,
    )

    role_image_counts = {
        role: int(len(rows))
        for role, rows in manifest.groupby("protocol_role", sort=True)
    }
    group_role_image_counts = {
        str(group): {
            str(role): int(count)
            for role, count in rows.groupby("protocol_role").size().to_dict().items()
        }
        for group, rows in manifest.groupby("rfw_group", sort=True)
    }
    group_role_identity_counts = {
        str(group): {
            str(role): int(count)
            for role, count in rows.groupby("protocol_role")["identity_id"]
            .nunique()
            .to_dict()
            .items()
        }
        for group, rows in manifest.groupby("rfw_group", sort=True)
    }
    summary = {
        "artifact_type": RFW_CUSTOM_ARTIFACT_TYPE,
        "dataset": RFW_CUSTOM_DATASET_ID,
        "protocol_family_uid": RFW_CUSTOM_PROTOCOL_FAMILY_UID,
        "protocol_uid": protocol_uid,
        "protocol_kind": "custom_1_to_n_open_set_identification",
        "official_protocol": False,
        "official_result_eligible": False,
        "official_pair_protocol_used": False,
        "open_set_protocol": True,
        "strict_unseen_identity_evidence": False,
        "checkpoint_overlap_status": "UNKNOWN",
        "identity_overlap_with_model_training": "UNKNOWN",
        "identity_split_unit": "source_identity_id_with_group_ambiguity_excluded",
        "demographic_field": "rfw_group",
        "demographic_group_source": "dataset_provided",
        "known_unknown_definition": (
            "deterministic custom subset of non-gallery test identities"
        ),
        "unknown_unknown_definition": (
            "disjoint deterministic custom subset of non-gallery test identities"
        ),
        "compressor_fit_allowed": True,
        "compressor_fit_split": "development",
        "calibration_fit_allowed": True,
        "calibration_fit_split": "calibration",
        "evaluation_split": "test",
        "seed": resolved_seed,
        "enrollment_count": enrollment,
        "development_fraction": development_ratio,
        "calibration_fraction": calibration_ratio,
        "test_fraction": 1.0 - development_ratio - calibration_ratio,
        "unknown_unknown_fraction_within_non_gallery": unknown_ratio,
        "gallery_identity_count_by_group": counts,
        "gallery_identity_count": int(len(set(selected_gallery_ids))),
        "known_unknown_identity_count": int(len(set(selected_known_unknown_ids))),
        "unknown_unknown_identity_count": int(len(set(selected_unknown_unknown_ids))),
        "non_mated_identity_count": int(
            len(set(selected_known_unknown_ids).union(selected_unknown_unknown_ids))
        ),
        "source_image_count": int(len(source)),
        "selected_image_count": int(len(manifest)),
        "excluded_cross_group_source_identity_count": int(
            len(ambiguous_source_identities)
        ),
        "excluded_cross_group_source_identity_sha256": ambiguous_digest,
        "excluded_cross_group_image_count": int(len(source) - len(eligible_source)),
        "source_archive_path": source_archive_path,
        "source_archive_sha256": archive_sha256,
        "source_manifest_logical_sha256": source_manifest_sha256,
        "source_protocol_logical_sha256": source_manifest_sha256,
        "protocol_contract_logical_sha256": protocol_contract_sha256,
        "split_identity_counts_by_group": split_identity_counts_by_group,
        "split_image_counts": {
            str(split): int(count)
            for split, count in manifest["split"].value_counts().to_dict().items()
        },
        "split_identity_counts": {
            str(split): int(count)
            for split, count in manifest.groupby("split")["identity_id"]
            .nunique()
            .to_dict()
            .items()
        },
        "role_image_counts": role_image_counts,
        "group_role_image_counts": group_role_image_counts,
        "group_role_identity_counts": group_role_identity_counts,
        "protocol_note": (
            "Custom RFW 1:N diagnostic with demographic-stratified, "
            "identity-disjoint development/calibration/test splits, derived "
            "from identity/image metadata. "
            "It is not the official RFW 1:1 verification protocol, does not "
            "use official pair folds, and is not strict unseen-identity evidence."
        ),
    }
    bundle = RFWCustomOpenSetBundle(
        manifest=manifest,
        protocol=protocol,
        gallery_identities=tuple(sorted(set(selected_gallery_ids))),
        known_unknown_identities=tuple(sorted(set(selected_known_unknown_ids))),
        unknown_unknown_identities=tuple(sorted(set(selected_unknown_unknown_ids))),
        non_mated_identities=tuple(
            sorted(set(selected_known_unknown_ids).union(selected_unknown_unknown_ids))
        ),
        summary=summary,
    )
    validate_rfw_custom_open_set_bundle(bundle)
    return bundle


def validate_rfw_custom_open_set_bundle(
    bundle: RFWCustomOpenSetBundle,
) -> None:
    """Fail closed unless manifest, roles, summary, and UID describe one bundle."""

    if not isinstance(bundle, RFWCustomOpenSetBundle):
        raise TypeError("bundle must be an RFWCustomOpenSetBundle")
    summary = bundle.summary
    if not isinstance(summary, dict):
        raise TypeError("RFW custom bundle summary must be a dictionary")
    if summary.get("artifact_type") != RFW_CUSTOM_ARTIFACT_TYPE:
        raise ValueError("RFW custom summary artifact_type mismatch")
    if summary.get("protocol_family_uid") != RFW_CUSTOM_PROTOCOL_FAMILY_UID:
        raise ValueError("RFW custom summary protocol family mismatch")
    if summary.get("official_protocol") is not False:
        raise ValueError("RFW custom summary must not claim official protocol status")
    if summary.get("official_pair_protocol_used") is not False:
        raise ValueError("RFW custom summary must not claim official pair usage")
    if summary.get("official_result_eligible") is not False:
        raise ValueError(
            "RFW custom summary must not claim official result eligibility"
        )
    if summary.get("checkpoint_overlap_status") != "UNKNOWN":
        raise ValueError("RFW custom checkpoint overlap status must be UNKNOWN")
    protocol_uid = str(summary.get("protocol_uid", ""))
    protocol = adapt_rfw_custom_manifest_to_open_set_protocol(
        bundle.manifest,
        expected_protocol_uid=protocol_uid,
    )

    expected_by_role = {
        "gallery": protocol.gallery,
        "registered_probe": protocol.registered_probes,
        "known_unknown_probe": protocol.known_unknown_probes,
        "unknown_unknown_probe": protocol.unknown_unknown_probes,
    }
    supplied_by_role = {
        "gallery": bundle.protocol.gallery,
        "registered_probe": bundle.protocol.registered_probes,
        "known_unknown_probe": bundle.protocol.known_unknown_probes,
        "unknown_unknown_probe": bundle.protocol.unknown_unknown_probes,
    }
    for role, expected in expected_by_role.items():
        expected_ids = expected["image_id"].astype(str).tolist()
        supplied_ids = supplied_by_role[role]["image_id"].astype(str).tolist()
        if supplied_ids != expected_ids:
            raise ValueError(f"RFW custom bundle {role} does not match its manifest")

    logical_assignment = bundle.manifest.assign(
        _custom_role=bundle.manifest["protocol_role"].astype(str)
    )
    contract_sha256 = _logical_frame_sha256(
        logical_assignment,
        ["_custom_role", "rfw_group", "identity_id", "image_id"],
    )
    if summary.get("protocol_contract_logical_sha256") != contract_sha256:
        raise ValueError("RFW custom protocol contract SHA mismatch")
    source_manifest_sha256 = str(summary.get("source_manifest_logical_sha256", ""))
    _sha256_text(
        source_manifest_sha256,
        name="source_manifest_logical_sha256",
    )
    if summary.get("source_protocol_logical_sha256") != source_manifest_sha256:
        raise ValueError("RFW custom source protocol SHA mismatch")
    archive_sha256 = _sha256_text(
        summary.get("source_archive_sha256"),
        name="source_archive_sha256",
    )
    excluded_sha256 = str(
        summary.get("excluded_cross_group_source_identity_sha256", "")
    )
    _sha256_text(
        excluded_sha256,
        name="excluded_cross_group_source_identity_sha256",
    )
    counts = summary.get("gallery_identity_count_by_group")
    if not isinstance(counts, dict) or set(counts) != set(RFW_GROUPS):
        raise ValueError("RFW custom summary gallery group counts are invalid")
    normalized_counts = {
        group: _positive_integer(
            counts[group],
            name=f"gallery_identity_count_by_group[{group!r}]",
        )
        for group in RFW_GROUPS
    }
    uid_payload = {
        "family": RFW_CUSTOM_PROTOCOL_FAMILY_UID,
        "source_manifest_sha256": source_manifest_sha256,
        "protocol_contract_sha256": contract_sha256,
        "source_archive_sha256": archive_sha256,
        "seed": int(summary.get("seed")),
        "enrollment_count": int(summary.get("enrollment_count")),
        "development_fraction": float(summary.get("development_fraction")),
        "calibration_fraction": float(summary.get("calibration_fraction")),
        "unknown_unknown_fraction": float(
            summary.get("unknown_unknown_fraction_within_non_gallery")
        ),
        "gallery_identity_count_by_group": normalized_counts,
        "excluded_cross_group_source_identity_sha256": excluded_sha256,
    }
    uid_digest = hashlib.sha256(
        json.dumps(uid_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_protocol_uid = f"{RFW_CUSTOM_PROTOCOL_FAMILY_UID}-{uid_digest[:16]}"
    if protocol_uid != expected_protocol_uid:
        raise ValueError("RFW custom protocol_uid does not match its science contract")

    if int(summary.get("selected_image_count", -1)) != len(bundle.manifest):
        raise ValueError("RFW custom summary selected_image_count mismatch")
    expected_split_counts = {
        str(split): int(count)
        for split, count in bundle.manifest.groupby("split")["identity_id"]
        .nunique()
        .to_dict()
        .items()
    }
    if summary.get("split_identity_counts") != expected_split_counts:
        raise ValueError("RFW custom summary split identity counts mismatch")
    expected_role_counts = {
        str(role): int(count)
        for role, count in bundle.manifest["protocol_role"]
        .value_counts()
        .to_dict()
        .items()
    }
    if summary.get("role_image_counts") != expected_role_counts:
        raise ValueError("RFW custom summary role image counts mismatch")

    gallery_ids = tuple(sorted(set(protocol.gallery["identity_id"].astype(str))))
    known_ids = tuple(
        sorted(set(protocol.known_unknown_probes["identity_id"].astype(str)))
    )
    unknown_ids = tuple(
        sorted(set(protocol.unknown_unknown_probes["identity_id"].astype(str)))
    )
    if bundle.gallery_identities != gallery_ids:
        raise ValueError("RFW custom gallery identity list mismatch")
    if bundle.known_unknown_identities != known_ids:
        raise ValueError("RFW custom known-unknown identity list mismatch")
    if bundle.unknown_unknown_identities != unknown_ids:
        raise ValueError("RFW custom unknown-unknown identity list mismatch")
    if bundle.non_mated_identities != tuple(sorted(set(known_ids).union(unknown_ids))):
        raise ValueError("RFW custom non-mated identity list mismatch")


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _artifact_entry(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": _file_sha256(path),
        "byte_count": int(path.stat().st_size),
    }


def write_rfw_custom_open_set_bundle(
    bundle: RFWCustomOpenSetBundle,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Persist custom artifacts under names that cannot mimic Official RFW."""

    validate_rfw_custom_open_set_bundle(bundle)
    protocol = adapt_rfw_custom_manifest_to_open_set_protocol(
        bundle.manifest,
        expected_protocol_uid=str(bundle.summary.get("protocol_uid", "")),
    )
    if set(protocol.gallery["image_id"]) != set(bundle.protocol.gallery["image_id"]):
        raise ValueError("RFW custom bundle gallery does not match its manifest")
    if set(protocol.registered_probes["image_id"]) != set(
        bundle.protocol.registered_probes["image_id"]
    ):
        raise ValueError(
            "RFW custom bundle registered probes do not match its manifest"
        )
    if set(protocol.known_unknown_probes["image_id"]) != set(
        bundle.protocol.known_unknown_probes["image_id"]
    ):
        raise ValueError(
            "RFW custom bundle known-unknown probes do not match its manifest"
        )
    if set(protocol.unknown_unknown_probes["image_id"]) != set(
        bundle.protocol.unknown_unknown_probes["image_id"]
    ):
        raise ValueError("RFW custom bundle non-mated probes do not match its manifest")
    if bundle.summary.get("artifact_type") != RFW_CUSTOM_ARTIFACT_TYPE:
        raise ValueError("RFW custom summary artifact_type mismatch")
    if bundle.summary.get("official_protocol") is not False:
        raise ValueError("RFW custom summary must not claim official protocol status")

    directory = Path(output_dir).expanduser().resolve()
    targets = {
        "rfw_custom_open_set_manifest.csv": (
            directory / "rfw_custom_open_set_manifest.csv"
        ),
        "rfw_custom_development_pool.csv": (
            directory / "rfw_custom_development_pool.csv"
        ),
        "rfw_custom_calibration_pool.csv": (
            directory / "rfw_custom_calibration_pool.csv"
        ),
        "rfw_custom_gallery.csv": directory / "rfw_custom_gallery.csv",
        "rfw_custom_mated_probes.csv": (directory / "rfw_custom_mated_probes.csv"),
        "rfw_custom_known_unknown_probes.csv": (
            directory / "rfw_custom_known_unknown_probes.csv"
        ),
        "rfw_custom_unknown_unknown_probes.csv": (
            directory / "rfw_custom_unknown_unknown_probes.csv"
        ),
        "rfw_custom_summary.json": directory / "rfw_custom_summary.json",
        "_SUCCESS": directory / "_SUCCESS",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite RFW custom outputs: "
            + ", ".join(str(path) for path in existing)
        )
    directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        targets["_SUCCESS"].unlink(missing_ok=True)

    _atomic_write_csv(targets["rfw_custom_open_set_manifest.csv"], bundle.manifest)
    _atomic_write_csv(
        targets["rfw_custom_development_pool.csv"],
        bundle.manifest.loc[
            bundle.manifest["protocol_role"].eq("development_pool")
        ].reset_index(drop=True),
    )
    _atomic_write_csv(
        targets["rfw_custom_calibration_pool.csv"],
        bundle.manifest.loc[
            bundle.manifest["protocol_role"].eq("calibration_pool")
        ].reset_index(drop=True),
    )
    _atomic_write_csv(targets["rfw_custom_gallery.csv"], protocol.gallery)
    _atomic_write_csv(
        targets["rfw_custom_mated_probes.csv"], protocol.registered_probes
    )
    _atomic_write_csv(
        targets["rfw_custom_known_unknown_probes.csv"],
        protocol.known_unknown_probes,
    )
    _atomic_write_csv(
        targets["rfw_custom_unknown_unknown_probes.csv"],
        protocol.unknown_unknown_probes,
    )
    artifact_names = [
        name for name in targets if name not in {"rfw_custom_summary.json", "_SUCCESS"}
    ]
    persisted_summary = dict(bundle.summary)
    persisted_summary["artifacts"] = {
        name: _artifact_entry(targets[name]) for name in artifact_names
    }
    _atomic_write_text(
        targets["rfw_custom_summary.json"],
        json.dumps(
            persisted_summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    summary_entry = _artifact_entry(targets["rfw_custom_summary.json"])
    _atomic_write_text(
        targets["_SUCCESS"],
        json.dumps(
            {
                "artifact_type": RFW_CUSTOM_ARTIFACT_TYPE,
                "protocol_family_uid": RFW_CUSTOM_PROTOCOL_FAMILY_UID,
                "protocol_uid": bundle.summary["protocol_uid"],
                "official_protocol": False,
                "summary_sha256": summary_entry["sha256"],
                "summary_byte_count": summary_entry["byte_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
    )
    return targets


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def load_rfw_custom_open_set_bundle(
    input_dir: str | Path,
) -> RFWCustomOpenSetBundle:
    """Load a completed custom bundle after SHA and contract validation."""

    directory = Path(input_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    success_path = directory / "_SUCCESS"
    summary_path = directory / "rfw_custom_summary.json"
    if not success_path.is_file():
        raise FileNotFoundError(
            f"completed RFW custom marker is missing: {success_path}"
        )
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    success = _read_json_object(success_path, label="RFW custom success marker")
    if success.get("artifact_type") != RFW_CUSTOM_ARTIFACT_TYPE:
        raise ValueError("RFW custom success artifact_type mismatch")
    if success.get("protocol_family_uid") != RFW_CUSTOM_PROTOCOL_FAMILY_UID:
        raise ValueError("RFW custom success protocol family mismatch")
    if success.get("official_protocol") is not False:
        raise ValueError("RFW custom success marker claims official status")
    if _file_sha256(summary_path) != _sha256_text(
        success.get("summary_sha256"),
        name="summary_sha256",
    ):
        raise ValueError("RFW custom summary SHA mismatch")
    if int(success.get("summary_byte_count", -1)) != int(summary_path.stat().st_size):
        raise ValueError("RFW custom summary byte-count mismatch")

    summary = _read_json_object(summary_path, label="RFW custom summary")
    if summary.get("artifact_type") != RFW_CUSTOM_ARTIFACT_TYPE:
        raise ValueError("RFW custom summary artifact_type mismatch")
    if summary.get("protocol_family_uid") != RFW_CUSTOM_PROTOCOL_FAMILY_UID:
        raise ValueError("RFW custom summary protocol family mismatch")
    if summary.get("protocol_uid") != success.get("protocol_uid"):
        raise ValueError("RFW custom success/summary protocol_uid mismatch")
    if summary.get("official_protocol") is not False:
        raise ValueError("RFW custom summary claims official status")
    if summary.get("official_pair_protocol_used") is not False:
        raise ValueError("RFW custom summary claims official pair usage")
    if summary.get("checkpoint_overlap_status") != "UNKNOWN":
        raise ValueError("RFW custom checkpoint overlap status must be UNKNOWN")
    _sha256_text(
        summary.get("source_archive_sha256"),
        name="source_archive_sha256",
    )
    _sha256_text(
        summary.get("source_protocol_logical_sha256"),
        name="source_protocol_logical_sha256",
    )

    expected_artifacts = {
        "rfw_custom_open_set_manifest.csv",
        "rfw_custom_development_pool.csv",
        "rfw_custom_calibration_pool.csv",
        "rfw_custom_gallery.csv",
        "rfw_custom_mated_probes.csv",
        "rfw_custom_known_unknown_probes.csv",
        "rfw_custom_unknown_unknown_probes.csv",
    }
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("RFW custom summary artifact registry is incomplete")
    artifact_paths: dict[str, Path] = {}
    for name in sorted(expected_artifacts):
        entry = artifacts[name]
        if not isinstance(entry, dict) or entry.get("path") != name:
            raise ValueError(f"RFW custom artifact registry path mismatch: {name}")
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if _file_sha256(path) != _sha256_text(
            entry.get("sha256"),
            name=f"artifacts[{name}].sha256",
        ):
            raise ValueError(f"RFW custom artifact SHA mismatch: {name}")
        if int(entry.get("byte_count", -1)) != int(path.stat().st_size):
            raise ValueError(f"RFW custom artifact byte-count mismatch: {name}")
        artifact_paths[name] = path

    manifest = pd.read_csv(artifact_paths["rfw_custom_open_set_manifest.csv"])
    protocol = adapt_rfw_custom_manifest_to_open_set_protocol(
        manifest,
        expected_protocol_uid=str(summary["protocol_uid"]),
    )
    role_artifacts = {
        "development_pool": "rfw_custom_development_pool.csv",
        "calibration_pool": "rfw_custom_calibration_pool.csv",
        "gallery": "rfw_custom_gallery.csv",
        "registered_probe": "rfw_custom_mated_probes.csv",
        "known_unknown_probe": "rfw_custom_known_unknown_probes.csv",
        "unknown_unknown_probe": "rfw_custom_unknown_unknown_probes.csv",
    }
    for role, name in role_artifacts.items():
        persisted = pd.read_csv(artifact_paths[name])
        expected_ids = (
            manifest.loc[manifest["protocol_role"].astype(str).eq(role), "image_id"]
            .astype(str)
            .tolist()
        )
        actual_ids = persisted["image_id"].astype(str).tolist()
        if actual_ids != expected_ids:
            raise ValueError(f"RFW custom {role} artifact does not match the manifest")
    if int(summary.get("selected_image_count", -1)) != len(manifest):
        raise ValueError("RFW custom summary selected_image_count mismatch")

    gallery_ids = tuple(sorted(set(protocol.gallery["identity_id"].astype(str))))
    known_unknown_ids = tuple(
        sorted(set(protocol.known_unknown_probes["identity_id"].astype(str)))
    )
    unknown_unknown_ids = tuple(
        sorted(set(protocol.unknown_unknown_probes["identity_id"].astype(str)))
    )
    loaded = RFWCustomOpenSetBundle(
        manifest=manifest,
        protocol=protocol,
        gallery_identities=gallery_ids,
        known_unknown_identities=known_unknown_ids,
        unknown_unknown_identities=unknown_unknown_ids,
        non_mated_identities=tuple(
            sorted(set(known_unknown_ids).union(unknown_unknown_ids))
        ),
        summary=summary,
    )
    validate_rfw_custom_open_set_bundle(loaded)
    return loaded
