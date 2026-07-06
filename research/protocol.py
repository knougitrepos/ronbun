from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
