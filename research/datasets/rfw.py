from __future__ import annotations

import json
import hashlib
import os
import re
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.datasets.sources import (
    DatasetIntegrityError,
    inspect_tar_archive,
    require_valid_tar,
)


RFW_GROUPS = ("African", "Asian", "Caucasian", "Indian")
RFW_OFFICIAL_PAIR_COUNT_PER_GROUP = 6_000
RFW_OFFICIAL_FOLD_COUNT = 10
RFW_OFFICIAL_PAIRS_PER_FOLD = 600
RFW_OFFICIAL_GENUINE_PER_FOLD = 300
_IMAGE_NAME = re.compile(r"^(?P<identity>.+)_(?P<index>\d{4})\.jpg$")


@dataclass(frozen=True)
class RFWVerificationBundle:
    """RFW official group-stratified 1:1 verification protocol."""

    manifest: pd.DataFrame
    pairs: pd.DataFrame
    landmarks: pd.DataFrame
    source_identities: tuple[str, ...]
    cross_group_source_identities: tuple[str, ...]
    summary: dict[str, Any]


def _relative_posix(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"dataset archive must be inside project_root: {path.resolve()}"
        ) from exc


def _read_text(handle: tarfile.TarFile, member_name: str) -> list[str]:
    try:
        member = handle.getmember(member_name)
    except KeyError as exc:
        raise DatasetIntegrityError(
            f"RFW protocol member missing: {member_name}"
        ) from exc
    source = handle.extractfile(member)
    if source is None:
        raise DatasetIntegrityError(f"RFW protocol member unreadable: {member_name}")
    with source:
        text = source.read().decode("utf-8-sig")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _image_key(filename: str) -> tuple[str, int]:
    match = _IMAGE_NAME.fullmatch(filename)
    if match is None:
        raise DatasetIntegrityError(f"unexpected RFW image filename: {filename}")
    return match.group("identity"), int(match.group("index"))


def _identity_id(group: str, source_identity_id: str) -> str:
    return f"rfw:{group.lower()}:{source_identity_id}"


def _image_id(group: str, filename: str) -> str:
    return f"rfw:{group.lower()}:{Path(filename).stem}"


def _parse_people(
    lines: list[str],
    *,
    group: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 2:
            raise DatasetIntegrityError(
                f"invalid RFW people row for {group}: {line!r}"
            )
        rows.append(
            {
                "source_identity_id": fields[0],
                "source_label": label,
                "declared_image_count": int(fields[1]),
            }
        )
    return pd.DataFrame(rows)


def _parse_images(
    lines: list[str],
    *,
    group: str,
    archive_relative_path: str,
    archive_members: set[str],
) -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for protocol_index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 2:
            raise DatasetIntegrityError(
                f"invalid RFW image row for {group}: {line!r}"
            )
        filename, label_text = fields
        source_identity_id, face_index = _image_key(filename)
        source_label = int(label_text)
        archive_member = (
            f"test/data/{group}/{source_identity_id}/{filename}"
        )
        if archive_member not in archive_members:
            raise DatasetIntegrityError(
                f"RFW image listed but absent from archive: {archive_member}"
            )
        key = (source_identity_id, face_index)
        if key in lookup:
            raise DatasetIntegrityError(f"duplicate RFW image key: {group} {key}")
        row = {
            "image_id": _image_id(group, filename),
            "identity_id": _identity_id(group, source_identity_id),
            "source_identity_id": source_identity_id,
            "split": "test",
            "image_path": f"tar://{archive_relative_path}#{archive_member}",
            "dataset": "rfw-v1",
            "dataset_role": "evaluation_test_only",
            "protocol_role": "verification_image",
            "rfw_group": group,
            "group_label_source": "dataset_provided",
            "source_label": source_label,
            "face_index": face_index,
            "protocol_index": protocol_index,
            "storage_kind": "tar_member_loose_jpg",
            "source_archive_path": archive_relative_path,
            "archive_member": archive_member,
        }
        rows.append(row)
        lookup[key] = row
    frame = pd.DataFrame(rows)
    if frame["image_id"].duplicated().any():
        raise DatasetIntegrityError(f"duplicate RFW image IDs in group {group}")
    return frame, lookup


def _parse_landmarks(
    lines: list[str],
    *,
    group: str,
    image_by_member: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    coordinate_names = [
        "left_eye_x",
        "left_eye_y",
        "right_eye_x",
        "right_eye_y",
        "nose_x",
        "nose_y",
        "left_mouth_x",
        "left_mouth_y",
        "right_mouth_x",
        "right_mouth_y",
    ]
    for protocol_index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 12:
            raise DatasetIntegrityError(
                f"invalid RFW landmark row for {group}: {line!r}"
            )
        archive_member = fields[0].lstrip("/")
        if archive_member not in image_by_member:
            raise DatasetIntegrityError(
                f"RFW landmark references unknown image: {archive_member}"
            )
        source_label = int(fields[1])
        image = image_by_member[archive_member]
        if source_label != image["source_label"]:
            raise DatasetIntegrityError(
                f"RFW landmark/image label mismatch: {archive_member}"
            )
        coordinates = [float(value) for value in fields[2:]]
        if not np.isfinite(coordinates).all():
            raise DatasetIntegrityError(
                f"non-finite RFW landmarks: {archive_member}"
            )
        if any(value < 0.0 or value >= 400.0 for value in coordinates):
            raise DatasetIntegrityError(
                f"RFW landmark outside 400x400 loose image bounds: {archive_member}"
            )
        row = {
            "image_id": image["image_id"],
            "rfw_group": group,
            "source_label": source_label,
            "protocol_index": protocol_index,
            **dict(zip(coordinate_names, coordinates, strict=True)),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame["image_id"].duplicated().any():
        raise DatasetIntegrityError(f"duplicate RFW landmarks in group {group}")
    return frame


def _parse_pairs(
    lines: list[str],
    *,
    group: str,
    image_lookup: dict[tuple[str, int], dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for official_index, line in enumerate(lines):
        fields = line.split()
        if len(fields) == 3:
            left_key = (fields[0], int(fields[1]))
            right_key = (fields[0], int(fields[2]))
            is_genuine = True
        elif len(fields) == 4:
            left_key = (fields[0], int(fields[1]))
            right_key = (fields[2], int(fields[3]))
            is_genuine = False
        else:
            raise DatasetIntegrityError(
                f"invalid RFW pair row for {group}: {line!r}"
            )
        try:
            left = image_lookup[left_key]
            right = image_lookup[right_key]
        except KeyError as exc:
            raise DatasetIntegrityError(
                f"RFW pair references unknown image in {group}: {line!r}"
            ) from exc
        if (left["identity_id"] == right["identity_id"]) != is_genuine:
            raise DatasetIntegrityError(
                f"RFW pair identity/label mismatch in {group}: {line!r}"
            )
        if left["image_id"] == right["image_id"]:
            raise DatasetIntegrityError(
                f"RFW self-pair is not allowed in {group}: {line!r}"
            )
        fold_index = official_index // RFW_OFFICIAL_PAIRS_PER_FOLD
        pair_index_in_fold = official_index % RFW_OFFICIAL_PAIRS_PER_FOLD
        rows.append(
            {
                "pair_id": (
                    f"rfw:{group.lower()}:fold{fold_index:02d}:"
                    f"pair{pair_index_in_fold:03d}"
                ),
                "rfw_group": group,
                "fold_index": fold_index,
                "official_index": official_index,
                "pair_index_in_fold": pair_index_in_fold,
                "left_image_id": left["image_id"],
                "right_image_id": right["image_id"],
                "left_identity_id": left["identity_id"],
                "right_identity_id": right["identity_id"],
                "left_source_identity_id": left["source_identity_id"],
                "right_source_identity_id": right["source_identity_id"],
                "is_genuine": is_genuine,
            }
        )
    frame = pd.DataFrame(rows)
    if frame["pair_id"].duplicated().any():
        raise DatasetIntegrityError(f"duplicate RFW pair IDs in group {group}")
    unordered_pairs = frame.apply(
        lambda row: tuple(
            sorted((str(row["left_image_id"]), str(row["right_image_id"])))
        ),
        axis=1,
    )
    if unordered_pairs.duplicated().any():
        raise DatasetIntegrityError(
            f"duplicate unordered RFW image pair in group {group}"
        )
    return frame


def _validate_official_group(
    *,
    group: str,
    manifest: pd.DataFrame,
    pairs: pd.DataFrame,
    landmarks: pd.DataFrame,
    people: pd.DataFrame,
) -> None:
    if len(pairs) != RFW_OFFICIAL_PAIR_COUNT_PER_GROUP:
        raise DatasetIntegrityError(
            f"RFW {group} pair count mismatch: "
            f"expected={RFW_OFFICIAL_PAIR_COUNT_PER_GROUP}, actual={len(pairs)}"
        )
    if sorted(pairs["fold_index"].unique().tolist()) != list(
        range(RFW_OFFICIAL_FOLD_COUNT)
    ):
        raise DatasetIntegrityError(f"RFW {group} must contain folds 0..9")
    fold_counts = pairs.groupby(["fold_index", "is_genuine"]).size().to_dict()
    expected = {
        (fold, is_genuine): RFW_OFFICIAL_GENUINE_PER_FOLD
        for fold in range(RFW_OFFICIAL_FOLD_COUNT)
        for is_genuine in (False, True)
    }
    if fold_counts != expected:
        raise DatasetIntegrityError(
            f"RFW {group} fold/label counts differ from official 300/300"
        )
    for fold, fold_rows in pairs.groupby("fold_index", sort=True):
        expected_flags = (
            [True] * RFW_OFFICIAL_GENUINE_PER_FOLD
            + [False] * RFW_OFFICIAL_GENUINE_PER_FOLD
        )
        if fold_rows["is_genuine"].tolist() != expected_flags:
            raise DatasetIntegrityError(
                f"RFW {group} fold {fold} does not preserve positive/negative order"
            )
    if len(landmarks) != len(manifest):
        raise DatasetIntegrityError(
            f"RFW {group} landmark/image count mismatch"
        )
    referenced_images = set(pairs["left_image_id"]).union(pairs["right_image_id"])
    if referenced_images != set(manifest["image_id"]):
        raise DatasetIntegrityError(
            f"RFW {group} pair references do not cover the official image list"
        )
    actual_counts = (
        manifest.groupby(["source_identity_id", "source_label"])
        .size()
        .rename("actual_image_count")
        .reset_index()
    )
    people_counts = people.merge(
        actual_counts,
        on=["source_identity_id", "source_label"],
        how="left",
        validate="one_to_one",
    )
    people_counts["actual_image_count"] = (
        people_counts["actual_image_count"].fillna(0).astype(int)
    )
    mismatches = people_counts.loc[
        people_counts["declared_image_count"]
        != people_counts["actual_image_count"]
    ]
    if not mismatches.empty:
        raise DatasetIntegrityError(
            f"RFW {group} people/image counts mismatch: "
            f"{mismatches.head(3).to_dict(orient='records')}"
        )


def build_rfw_verification_bundle(
    jpg_archive_path: str | Path,
    project_root: str | Path,
    *,
    strict_official: bool = True,
) -> RFWVerificationBundle:
    """Parse RFW's JPG archive and preserve its official 1:1 protocol.

    The returned manifest uses explicit ``tar://`` locators. It is a source
    protocol manifest, not a materialized 112x112 crop manifest.
    """

    archive = Path(jpg_archive_path).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"project root not found: {project}")
    archive_relative_path = _relative_posix(archive, project)
    report = inspect_tar_archive(archive, expected_prefixes=("test",))
    require_valid_tar(report)
    archive_digest = hashlib.sha256()
    with archive.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            archive_digest.update(chunk)

    manifests: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    landmark_frames: list[pd.DataFrame] = []
    people_frames: list[pd.DataFrame] = []
    with tarfile.open(archive, mode="r:*") as handle:
        members = handle.getmembers()
        member_names = {member.name for member in members}
        actual_jpg_members = {
            member.name
            for member in members
            if member.isfile()
            and member.name.startswith("test/data/")
            and member.name.lower().endswith(".jpg")
        }
        for group in RFW_GROUPS:
            base = f"test/txts/{group}"
            image_lines = _read_text(
                handle, f"{base}/{group}_images.txt"
            )
            landmark_lines = _read_text(
                handle, f"{base}/{group}_lmk.txt"
            )
            people_lines = _read_text(
                handle, f"{base}/{group}_people.txt"
            )
            pair_lines = _read_text(
                handle, f"{base}/{group}_pairs.txt"
            )

            manifest, image_lookup = _parse_images(
                image_lines,
                group=group,
                archive_relative_path=archive_relative_path,
                archive_members=member_names,
            )
            image_by_member = {
                row["archive_member"]: row
                for row in manifest.to_dict(orient="records")
            }
            landmarks = _parse_landmarks(
                landmark_lines,
                group=group,
                image_by_member=image_by_member,
            )
            people = _parse_people(people_lines, group=group)
            people["rfw_group"] = group
            pairs = _parse_pairs(
                pair_lines,
                group=group,
                image_lookup=image_lookup,
            )
            if strict_official:
                _validate_official_group(
                    group=group,
                    manifest=manifest,
                    pairs=pairs,
                    landmarks=landmarks,
                    people=people,
                )
            manifests.append(manifest)
            pair_frames.append(pairs)
            landmark_frames.append(landmarks)
            people_frames.append(people)

        manifest = pd.concat(manifests, ignore_index=True)
        pairs = pd.concat(pair_frames, ignore_index=True)
        landmarks = pd.concat(landmark_frames, ignore_index=True)
        people = pd.concat(people_frames, ignore_index=True)
        listed_members = set(manifest["archive_member"])
        if actual_jpg_members != listed_members:
            missing = sorted(listed_members.difference(actual_jpg_members))
            extra = sorted(actual_jpg_members.difference(listed_members))
            raise DatasetIntegrityError(
                "RFW image list/archive mismatch: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )

    if manifest["image_id"].duplicated().any():
        raise DatasetIntegrityError("RFW image IDs are not unique across groups")
    if pairs["pair_id"].duplicated().any():
        raise DatasetIntegrityError("RFW pair IDs are not unique across groups")
    if set(landmarks["image_id"]) != set(manifest["image_id"]):
        raise DatasetIntegrityError("RFW landmark/image IDs differ")

    zero_people = people.loc[people["declared_image_count"] == 0]
    cross_group_identity_counts = (
        manifest.groupby("source_identity_id")["rfw_group"].nunique()
    )
    cross_group_source_identities = tuple(
        sorted(
            cross_group_identity_counts.loc[
                cross_group_identity_counts > 1
            ].index.astype(str)
        )
    )
    cross_group_payload = "\n".join(cross_group_source_identities).encode("utf-8")
    source_identities = tuple(sorted(manifest["source_identity_id"].unique()))
    summary = {
        "dataset": "rfw-v1",
        "role": "evaluation_test_only",
        "task": (
            "official_10fold_pair_verification"
            if strict_official
            else "unvalidated_pair_verification"
        ),
        "official_protocol_validated": bool(strict_official),
        "official_open_set_protocol": False,
        "compressor_fit_allowed": False,
        "source_representation": "jpg_loose_with_landmarks",
        "materialized_aligned_crop_manifest_ready": False,
        "image_count": int(len(manifest)),
        "identity_count_within_group": int(manifest["identity_id"].nunique()),
        "unique_source_identity_count": int(len(source_identities)),
        "pair_count": int(len(pairs)),
        "genuine_pair_count": int(pairs["is_genuine"].sum()),
        "impostor_pair_count": int((~pairs["is_genuine"]).sum()),
        "group_image_counts": {
            str(key): int(value)
            for key, value in manifest["rfw_group"].value_counts().to_dict().items()
        },
        "group_identity_counts": {
            str(key): int(value)
            for key, value in manifest.groupby("rfw_group")["identity_id"]
            .nunique()
            .to_dict()
            .items()
        },
        "group_pair_counts": {
            str(key): int(value)
            for key, value in pairs["rfw_group"].value_counts().to_dict().items()
        },
        "people_zero_image_row_count": int(len(zero_people)),
        "people_zero_image_rows": zero_people[
            ["rfw_group", "source_identity_id", "source_label"]
        ].to_dict(orient="records"),
        "source_identities_in_multiple_groups": int(
            len(cross_group_source_identities)
        ),
        "cross_group_source_identity_list_sha256": hashlib.sha256(
            cross_group_payload
        ).hexdigest().upper(),
        "source_archive_path": archive_relative_path,
        "source_archive_byte_count": int(archive.stat().st_size),
        "source_archive_sha256": archive_digest.hexdigest().upper(),
        "tar_member_count": int(report.member_count),
        "tar_readable_to_eof": bool(report.readable_to_eof),
        "protocol_note": (
            "RFW is a dataset-provided group-stratified 1:1 verification test. "
            "Do not fit PCA/PQ on these rows and do not report DIR/FPIR from this protocol."
        ),
    }
    return RFWVerificationBundle(
        manifest=manifest.sort_values(
            ["rfw_group", "protocol_index"]
        ).reset_index(drop=True),
        pairs=pairs.sort_values(["rfw_group", "official_index"]).reset_index(
            drop=True
        ),
        landmarks=landmarks.sort_values(
            ["rfw_group", "protocol_index"]
        ).reset_index(drop=True),
        source_identities=source_identities,
        cross_group_source_identities=cross_group_source_identities,
        summary=summary,
    )


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_rfw_verification_bundle(
    bundle: RFWVerificationBundle,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write source protocol metadata without extracting images."""

    directory = Path(output_dir).expanduser().resolve()
    targets = {
        "image_manifest.csv": directory / "image_manifest.csv",
        "pair_protocol.csv": directory / "pair_protocol.csv",
        "landmarks.csv": directory / "landmarks.csv",
        "source_identities.txt": directory / "source_identities.txt",
        "cross_group_source_identities.txt": (
            directory / "cross_group_source_identities.txt"
        ),
        "summary.json": directory / "summary.json",
        "_SUCCESS": directory / "_SUCCESS",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite RFW outputs: "
            + ", ".join(str(path) for path in existing)
        )
    directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        targets["_SUCCESS"].unlink(missing_ok=True)
    _atomic_write_csv(targets["image_manifest.csv"], bundle.manifest)
    _atomic_write_csv(targets["pair_protocol.csv"], bundle.pairs)
    _atomic_write_csv(targets["landmarks.csv"], bundle.landmarks)
    _atomic_write_text(
        targets["source_identities.txt"],
        "".join(f"{identity}\n" for identity in bundle.source_identities),
    )
    _atomic_write_text(
        targets["cross_group_source_identities.txt"],
        "".join(
            f"{identity}\n" for identity in bundle.cross_group_source_identities
        ),
    )
    _atomic_write_text(
        targets["summary.json"],
        json.dumps(bundle.summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    _atomic_write_text(
        targets["_SUCCESS"],
        json.dumps(
            {
                "dataset": "rfw-v1",
                "official_protocol_validated": bool(
                    bundle.summary.get("official_protocol_validated", False)
                ),
                "source_archive_sha256": bundle.summary.get(
                    "source_archive_sha256"
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
    )
    return targets


def _stable_pair_key(seed: int, pair_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{pair_id}".encode("utf-8")).hexdigest()


def select_rfw_protocol_scope(
    bundle: RFWVerificationBundle,
    *,
    mode: str,
    data_fraction: float,
    seed: int,
) -> RFWVerificationBundle:
    """Select a deterministic dev subset without relabeling it as official.

    Development selection is stratified by RFW group, fold, and genuine/
    impostor label. Only ``mode='real', data_fraction=1.0`` preserves the full
    official protocol result boundary.
    """

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"dev", "real"}:
        raise ValueError("mode must be 'dev' or 'real'")
    if not 0.0 < data_fraction <= 1.0:
        raise ValueError("data_fraction must be in (0, 1]")
    if normalized_mode == "real" and data_fraction != 1.0:
        raise ValueError("RFW real mode requires data_fraction=1.0")

    if normalized_mode == "real":
        group_pair_counts = bundle.pairs["rfw_group"].value_counts().to_dict()
        full_pair_counts_valid = (
            len(bundle.pairs)
            == len(RFW_GROUPS) * RFW_OFFICIAL_PAIR_COUNT_PER_GROUP
            and all(
                group_pair_counts.get(group) == RFW_OFFICIAL_PAIR_COUNT_PER_GROUP
                for group in RFW_GROUPS
            )
        )
        if (
            not bundle.summary.get("official_protocol_validated", False)
            or not full_pair_counts_valid
        ):
            raise DatasetIntegrityError(
                "RFW real mode requires a strict official protocol bundle"
            )
        summary = {
            **bundle.summary,
            "scope_mode": "real",
            "data_fraction": 1.0,
            "scope_seed": int(seed),
            "official_result_eligible": True,
        }
        return RFWVerificationBundle(
            manifest=bundle.manifest.copy(),
            pairs=bundle.pairs.copy(),
            landmarks=bundle.landmarks.copy(),
            source_identities=bundle.source_identities,
            cross_group_source_identities=bundle.cross_group_source_identities,
            summary=summary,
        )

    selected_pair_ids: set[str] = set()
    for _, stratum in bundle.pairs.groupby(
        ["rfw_group", "fold_index", "is_genuine"],
        sort=True,
        observed=True,
    ):
        ordered_ids = sorted(
            stratum["pair_id"].astype(str),
            key=lambda pair_id: _stable_pair_key(seed, pair_id),
        )
        selected_count = max(1, int(len(ordered_ids) * data_fraction))
        selected_pair_ids.update(ordered_ids[:selected_count])

    pairs = bundle.pairs.loc[
        bundle.pairs["pair_id"].isin(selected_pair_ids)
    ].reset_index(drop=True)
    selected_image_ids = set(pairs["left_image_id"]).union(pairs["right_image_id"])
    manifest = bundle.manifest.loc[
        bundle.manifest["image_id"].isin(selected_image_ids)
    ].reset_index(drop=True)
    landmarks = bundle.landmarks.loc[
        bundle.landmarks["image_id"].isin(selected_image_ids)
    ].reset_index(drop=True)
    source_identities = tuple(sorted(manifest["source_identity_id"].unique()))
    cross_group_counts = manifest.groupby("source_identity_id")["rfw_group"].nunique()
    cross_group_source_identities = tuple(
        sorted(cross_group_counts.loc[cross_group_counts > 1].index.astype(str))
    )
    cross_group_payload = "\n".join(cross_group_source_identities).encode("utf-8")
    summary = {
        **bundle.summary,
        "task": "dev_group_fold_label_stratified_pair_subset",
        "source_official_protocol_validated": bool(
            bundle.summary.get("official_protocol_validated", False)
        ),
        "official_protocol_validated": False,
        "scope_mode": "dev",
        "data_fraction": float(data_fraction),
        "scope_seed": int(seed),
        "official_result_eligible": False,
        "selected_pair_count": int(len(pairs)),
        "selected_image_count": int(len(manifest)),
        "pair_count": int(len(pairs)),
        "image_count": int(len(manifest)),
        "identity_count_within_group": int(manifest["identity_id"].nunique()),
        "unique_source_identity_count": int(len(source_identities)),
        "genuine_pair_count": int(pairs["is_genuine"].sum()),
        "impostor_pair_count": int((~pairs["is_genuine"]).sum()),
        "group_pair_counts": {
            str(key): int(value)
            for key, value in pairs["rfw_group"].value_counts().to_dict().items()
        },
        "group_image_counts": {
            str(key): int(value)
            for key, value in manifest["rfw_group"].value_counts().to_dict().items()
        },
        "group_identity_counts": {
            str(key): int(value)
            for key, value in manifest.groupby("rfw_group")["identity_id"]
            .nunique()
            .to_dict()
            .items()
        },
        "source_identities_in_multiple_groups": int(
            len(cross_group_source_identities)
        ),
        "cross_group_source_identity_list_sha256": hashlib.sha256(
            cross_group_payload
        ).hexdigest().upper(),
        "selection_strata": ["rfw_group", "fold_index", "is_genuine"],
    }
    return RFWVerificationBundle(
        manifest=manifest,
        pairs=pairs,
        landmarks=landmarks,
        source_identities=source_identities,
        cross_group_source_identities=cross_group_source_identities,
        summary=summary,
    )
