from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"image_id", "identity_id", "split", "image_path"}
ALLOWED_SPLITS = {"development", "calibration", "test"}


@dataclass(frozen=True)
class OpenSetProtocol:
    gallery: pd.DataFrame
    registered_probes: pd.DataFrame
    known_unknown_probes: pd.DataFrame
    unknown_unknown_probes: pd.DataFrame


def validate_identity_disjoint_splits(manifest: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"missing manifest columns: {sorted(missing)}")

    unknown = set(manifest["split"]).difference(ALLOWED_SPLITS)
    if unknown:
        raise ValueError(f"unknown split names: {sorted(unknown)}")

    split_counts = manifest.groupby("identity_id")["split"].nunique()
    leaked = split_counts[split_counts > 1].index.tolist()
    if leaked:
        raise ValueError(f"identity leakage detected: {leaked[:10]}")


def build_open_set_protocol(
    manifest: pd.DataFrame,
    gallery_identities: Sequence[str],
    unknown_unknown_identities: Sequence[str],
    enrollment_count: int,
    seed: int,
) -> OpenSetProtocol:
    validate_identity_disjoint_splits(manifest)
    if enrollment_count < 1:
        raise ValueError("enrollment_count must be positive")

    gallery_ids = set(gallery_identities)
    unknown_unknown_ids = set(unknown_unknown_identities)
    overlap = gallery_ids.intersection(unknown_unknown_ids)
    if overlap:
        raise ValueError(f"gallery and unknown_unknown identities overlap: {sorted(overlap)}")

    test_rows = manifest.loc[manifest["split"] == "test"].copy()
    registered = test_rows.loc[test_rows["identity_id"].isin(gallery_ids)].copy()
    unknown_unknown = test_rows.loc[test_rows["identity_id"].isin(unknown_unknown_ids)].copy()
    known_unknown = test_rows.loc[
        ~test_rows["identity_id"].isin(gallery_ids)
        & ~test_rows["identity_id"].isin(unknown_unknown_ids)
    ].copy()

    if registered.empty:
        raise ValueError("registered identity rows are required")
    if known_unknown.empty or unknown_unknown.empty:
        raise ValueError("both known_unknown and unknown_unknown probe sets are required")

    group_sizes = registered.groupby("identity_id")["image_id"].nunique()
    too_small = group_sizes[group_sizes <= enrollment_count].index.tolist()
    if too_small:
        raise ValueError(
            "registered probe set is empty after enrollment for identities: "
            f"{too_small[:10]}"
        )

    gallery = (
        registered.groupby("identity_id", group_keys=False)
        .sample(n=enrollment_count, random_state=seed, replace=False)
        .sort_values(["identity_id", "image_id"])
    )
    registered_probes = registered.loc[~registered["image_id"].isin(gallery["image_id"])].copy()
    if registered_probes.empty:
        raise ValueError("registered probe set is empty after enrollment")

    return OpenSetProtocol(
        gallery=gallery.reset_index(drop=True),
        registered_probes=registered_probes.sort_values(["identity_id", "image_id"]).reset_index(drop=True),
        known_unknown_probes=known_unknown.sort_values(["identity_id", "image_id"]).reset_index(drop=True),
        unknown_unknown_probes=unknown_unknown.sort_values(["identity_id", "image_id"]).reset_index(drop=True),
    )


def _stable_key(value: str, *, seed: int, namespace: str) -> str:
    payload = f"{namespace}:{seed}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_calibration_protocol(
    manifest: pd.DataFrame,
    *,
    split_name: str,
    gallery_identity_count: int,
    enrollment_count: int,
    seed: int,
) -> OpenSetProtocol:
    """Build a deterministic development/calibration open-set protocol.

    This helper never reads test identities. All non-gallery identities in the
    requested split are treated as same-domain known-unknown probes. The final
    test-only unknown-unknown group remains empty by construction.
    """

    validate_identity_disjoint_splits(manifest)
    if split_name not in {"development", "calibration"}:
        raise ValueError("split_name must be development or calibration")
    if gallery_identity_count < 1 or enrollment_count < 1:
        raise ValueError("gallery_identity_count and enrollment_count must be positive")

    rows = manifest.loc[manifest["split"].eq(split_name)].copy()
    group_sizes = rows.groupby("identity_id")["image_id"].nunique()
    eligible = [
        str(identity_id)
        for identity_id, count in group_sizes.items()
        if int(count) > enrollment_count
    ]
    eligible.sort(
        key=lambda value: _stable_key(
            value, seed=seed, namespace=f"{split_name}:gallery-identity"
        )
    )
    if len(eligible) < gallery_identity_count:
        raise ValueError(
            f"{split_name} has only {len(eligible)} identities with more than "
            f"{enrollment_count} images; requested {gallery_identity_count}"
        )
    gallery_ids = set(eligible[:gallery_identity_count])
    registered = rows.loc[rows["identity_id"].astype(str).isin(gallery_ids)].copy()

    gallery_parts = []
    for identity_id, group in registered.groupby("identity_id", sort=True):
        ordered = group.assign(
            _stable_order=group["image_id"].astype(str).map(
                lambda value: _stable_key(
                    value,
                    seed=seed,
                    namespace=f"{split_name}:{identity_id}:enrollment",
                )
            )
        ).sort_values("_stable_order")
        gallery_parts.append(ordered.head(enrollment_count).drop(columns="_stable_order"))
    gallery = pd.concat(gallery_parts, ignore_index=True)
    registered_probes = registered.loc[
        ~registered["image_id"].isin(gallery["image_id"])
    ].copy()
    known_unknown = rows.loc[~rows["identity_id"].astype(str).isin(gallery_ids)].copy()
    if known_unknown.empty:
        raise ValueError(
            f"{split_name} known-unknown probes are empty; reserve non-gallery identities"
        )
    empty_unknown = rows.iloc[0:0].copy()
    return OpenSetProtocol(
        gallery=gallery.sort_values(["identity_id", "image_id"]).reset_index(drop=True),
        registered_probes=registered_probes.sort_values(
            ["identity_id", "image_id"]
        ).reset_index(drop=True),
        known_unknown_probes=known_unknown.sort_values(
            ["identity_id", "image_id"]
        ).reset_index(drop=True),
        unknown_unknown_probes=empty_unknown.reset_index(drop=True),
    )


def build_survface_matched_calibration_protocol(
    manifest: pd.DataFrame,
    *,
    gallery_identity_count: int = 3000,
    seed: int = 42,
) -> OpenSetProtocol:
    """Mirror the official 3,000-ID half-gallery regime on training IDs.

    All training identities must have at least two images. Identities are
    selected by a stable seeded hash. For each selected watch-list identity,
    ``floor(n / 2)`` deterministically ordered images form the gallery and the
    remainder form registered probes. Every image from every non-gallery
    training identity is a non-mated (known-unknown) calibration probe. Test
    rows are ignored by construction.
    """

    validate_identity_disjoint_splits(manifest)
    if gallery_identity_count < 1:
        raise ValueError("gallery_identity_count must be positive")
    if "protocol_role" not in manifest.columns:
        raise ValueError("SurvFace calibration requires protocol_role")
    training = manifest.loc[
        manifest["protocol_role"].astype(str).eq("training")
    ].copy()
    if training.empty:
        raise ValueError("SurvFace training rows are required")
    if set(training["split"].astype(str)) - {"development", "calibration"}:
        raise ValueError("SurvFace training rows have an unexpected split")
    sizes = training.groupby("identity_id")["image_id"].nunique()
    too_small = sizes.loc[sizes < 2]
    if not too_small.empty:
        raise ValueError(
            "half-gallery calibration requires at least two images per "
            f"identity: {too_small.index.astype(str).tolist()[:10]}"
        )
    if gallery_identity_count >= len(sizes):
        raise ValueError(
            "gallery_identity_count must reserve at least one non-mated "
            f"identity: gallery={gallery_identity_count}, identities={len(sizes)}"
        )

    identities = sorted(
        sizes.index.astype(str),
        key=lambda value: _stable_key(
            value,
            seed=seed,
            namespace="survface:training-matched-calibration:gallery-identity",
        ),
    )
    gallery_ids = set(identities[:gallery_identity_count])
    registered = training.loc[
        training["identity_id"].astype(str).isin(gallery_ids)
    ].copy()
    gallery_parts: list[pd.DataFrame] = []
    registered_parts: list[pd.DataFrame] = []
    for identity_id, group in registered.groupby("identity_id", sort=True):
        ordered = group.assign(
            _stable_order=group["image_id"].astype(str).map(
                lambda value: _stable_key(
                    value,
                    seed=seed,
                    namespace=(
                        "survface:training-matched-calibration:"
                        f"{identity_id}:half-gallery"
                    ),
                )
            )
        ).sort_values(["_stable_order", "image_id"], kind="stable")
        gallery_count = len(ordered) // 2
        gallery_parts.append(
            ordered.iloc[:gallery_count].drop(columns="_stable_order")
        )
        registered_parts.append(
            ordered.iloc[gallery_count:].drop(columns="_stable_order")
        )
    gallery = pd.concat(gallery_parts, ignore_index=True)
    registered_probes = pd.concat(registered_parts, ignore_index=True)
    known_unknown = training.loc[
        ~training["identity_id"].astype(str).isin(gallery_ids)
    ].copy()
    empty_unknown = training.iloc[0:0].copy()
    if (
        gallery["identity_id"].nunique() != gallery_identity_count
        or registered_probes["identity_id"].nunique() != gallery_identity_count
        or known_unknown.empty
    ):
        raise RuntimeError("SurvFace matched calibration construction failed")
    return OpenSetProtocol(
        gallery=gallery.sort_values(
            ["identity_id", "image_id"], kind="stable"
        ).reset_index(drop=True),
        registered_probes=registered_probes.sort_values(
            ["identity_id", "image_id"], kind="stable"
        ).reset_index(drop=True),
        known_unknown_probes=known_unknown.sort_values(
            ["identity_id", "image_id"], kind="stable"
        ).reset_index(drop=True),
        unknown_unknown_probes=empty_unknown.reset_index(drop=True),
    )


def build_survface_official_protocol(manifest: pd.DataFrame) -> OpenSetProtocol:
    """Preserve the official SurvFace gallery/mated/unmated roles and order."""

    required = REQUIRED_COLUMNS.union({"protocol_role", "protocol_index"})
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"missing SurvFace protocol columns: {sorted(missing)}")
    validate_identity_disjoint_splits(manifest)
    roles = set(manifest["protocol_role"].astype(str))
    expected = {"gallery", "registered_probe", "unknown_unknown_probe"}
    if roles != expected:
        raise ValueError(
            f"SurvFace protocol roles must be {sorted(expected)}, got {sorted(roles)}"
        )

    role_frames: dict[str, pd.DataFrame] = {}
    for name in sorted(expected):
        frame = manifest.loc[manifest["protocol_role"].eq(name)].copy()
        indexes = pd.to_numeric(frame["protocol_index"], errors="coerce")
        if indexes.isna().any() or not indexes.map(
            lambda value: float(value).is_integer()
        ).all():
            raise ValueError(f"{name} protocol_index must contain integers")
        actual = sorted(indexes.astype(int).tolist())
        expected_indexes = list(range(len(frame)))
        if actual != expected_indexes:
            raise ValueError(
                f"{name} protocol_index must be unique and contiguous from 0"
            )
        role_frames[name] = frame.sort_values(
            "protocol_index", kind="stable"
        ).reset_index(drop=True)

    gallery_ids = set(role_frames["gallery"]["identity_id"].astype(str))
    registered_ids = set(
        role_frames["registered_probe"]["identity_id"].astype(str)
    )
    unknown_ids = set(
        role_frames["unknown_unknown_probe"]["identity_id"].astype(str)
    )
    if registered_ids != gallery_ids:
        missing_registered = sorted(gallery_ids.difference(registered_ids))
        missing_gallery = sorted(registered_ids.difference(gallery_ids))
        raise ValueError(
            "SurvFace registered/gallery identity sets differ: "
            f"gallery_only={missing_registered[:10]}, registered_only={missing_gallery[:10]}"
        )
    overlap = gallery_ids.intersection(unknown_ids)
    if overlap:
        raise ValueError(
            f"SurvFace unknown identities overlap gallery identities: {sorted(overlap)[:10]}"
        )

    empty_known = manifest.iloc[0:0].copy()
    return OpenSetProtocol(
        gallery=role_frames["gallery"],
        registered_probes=role_frames["registered_probe"],
        known_unknown_probes=empty_known.reset_index(drop=True),
        unknown_unknown_probes=role_frames["unknown_unknown_probe"],
    )


def rebase_survface_protocol_subset_indexes(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Give a selected official subset contiguous local protocol indexes.

    ``source_protocol_index`` retains the row's position in the complete QMUL-
    SurvFace protocol. ``protocol_index`` is then rebased independently inside
    each selected role so the subset itself remains a valid ordered protocol.
    The input row order is preserved.
    """

    required = {"protocol_role", "protocol_index"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(
            f"missing SurvFace protocol columns: {sorted(missing)}"
        )
    expected_roles = {
        "gallery",
        "registered_probe",
        "unknown_unknown_probe",
    }
    roles = set(manifest["protocol_role"].astype(str))
    if roles != expected_roles:
        raise ValueError(
            "SurvFace protocol roles must be "
            f"{sorted(expected_roles)}, got {sorted(roles)}"
        )

    rebased = manifest.copy()
    if "source_protocol_index" not in rebased.columns:
        rebased["source_protocol_index"] = rebased["protocol_index"]

    for role in sorted(expected_roles):
        role_mask = rebased["protocol_role"].astype(str).eq(role)
        source_indexes = pd.to_numeric(
            rebased.loc[role_mask, "source_protocol_index"],
            errors="coerce",
        )
        if source_indexes.isna().any() or not source_indexes.map(
            lambda value: float(value).is_integer()
        ).all():
            raise ValueError(
                f"{role} source_protocol_index must contain integers"
            )
        integer_indexes = source_indexes.astype(int)
        if integer_indexes.duplicated().any():
            raise ValueError(
                f"{role} source_protocol_index must be unique"
            )
        ordered_rows = integer_indexes.sort_values(kind="stable").index
        rebased.loc[ordered_rows, "source_protocol_index"] = (
            integer_indexes.loc[ordered_rows].to_numpy()
        )
        rebased.loc[ordered_rows, "protocol_index"] = range(len(ordered_rows))

    rebased["source_protocol_index"] = pd.to_numeric(
        rebased["source_protocol_index"]
    ).astype("int64")
    rebased["protocol_index"] = pd.to_numeric(
        rebased["protocol_index"]
    ).astype("int64")
    return rebased


def filter_protocol_to_available_embeddings(
    protocol: OpenSetProtocol,
    available_image_paths: Sequence[str | Path],
    *,
    project_root: str | Path,
) -> tuple[OpenSetProtocol, dict[str, dict[str, object]]]:
    """Filter extraction failures while recording exact protocol coverage."""

    root = Path(project_root).resolve()

    def canonical(value: str | Path) -> str:
        path = Path(value)
        return os.path.normcase(
            str((path if path.is_absolute() else root / path).resolve())
        )

    available = {canonical(value) for value in available_image_paths}
    report: dict[str, dict[str, object]] = {}

    def filtered(name: str, frame: pd.DataFrame) -> pd.DataFrame:
        mask = frame["image_path"].map(canonical).isin(available)
        missing_rows = frame.loc[~mask]
        report[name] = {
            "input_rows": int(len(frame)),
            "available_rows": int(mask.sum()),
            "missing_rows": int((~mask).sum()),
            "missing_image_ids": missing_rows["image_id"].astype(str).tolist(),
        }
        return frame.loc[mask].reset_index(drop=True)

    resolved = OpenSetProtocol(
        gallery=filtered("gallery", protocol.gallery),
        registered_probes=filtered("registered_probes", protocol.registered_probes),
        known_unknown_probes=filtered(
            "known_unknown_probes", protocol.known_unknown_probes
        ),
        unknown_unknown_probes=filtered(
            "unknown_unknown_probes", protocol.unknown_unknown_probes
        ),
    )
    if resolved.gallery.empty:
        raise ValueError("no gallery embeddings remain after extraction coverage filtering")
    if resolved.registered_probes.empty:
        raise ValueError("no registered probe embeddings remain after coverage filtering")
    return resolved, report
