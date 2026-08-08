from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import pickle
import pickletools
import tarfile
from typing import Any, Iterator

import numpy as np
import pandas as pd
from PIL import Image

from research.datasets.rfw import (
    RFW_GROUPS,
    RFW_OFFICIAL_PAIR_COUNT_PER_GROUP,
)
from research.datasets.sources import DatasetIntegrityError


_MEMBER_BY_GROUP = {
    group: f"RFW_test/{group}_test.bin" for group in RFW_GROUPS
}
_FORBIDDEN_PICKLE_OPCODES = {
    "STACK_GLOBAL",
    "BUILD",
    "INST",
    "OBJ",
    "NEWOBJ",
    "NEWOBJ_EX",
    "EXT1",
    "EXT2",
    "EXT4",
    "PERSID",
    "BINPERSID",
}
_ALLOWED_PICKLE_GLOBALS = {"_codecs encode"}
_PAIR_COLUMNS = {
    "pair_id",
    "rfw_group",
    "fold_index",
    "official_index",
    "is_genuine",
}


@dataclass(frozen=True)
class RFWAlignedBinSummary:
    archive_path: Path
    archive_sha256: str
    groups: tuple[str, ...]
    pair_count: int
    encoded_image_occurrence_count: int
    image_size: tuple[int, int]


@dataclass(frozen=True)
class RFWAlignedPairBatch:
    faces: np.ndarray
    occurrences: pd.DataFrame


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pickle_program(payload: bytes, *, member_name: str) -> None:
    try:
        for opcode, argument, position in pickletools.genops(payload):
            if opcode.name == "GLOBAL" and argument not in _ALLOWED_PICKLE_GLOBALS:
                raise DatasetIntegrityError(
                    "RFW aligned BIN contains a forbidden pickle global: "
                    f"member={member_name}, global={argument!r}, position={position}"
                )
            if opcode.name in _FORBIDDEN_PICKLE_OPCODES:
                raise DatasetIntegrityError(
                    "RFW aligned BIN contains a forbidden pickle opcode: "
                    f"member={member_name}, opcode={opcode.name}, "
                    f"argument={argument!r}, position={position}"
                )
    except DatasetIntegrityError:
        raise
    except Exception as exc:
        raise DatasetIntegrityError(
            f"RFW aligned BIN pickle stream is invalid: {member_name}"
        ) from exc


class _RestrictedRFWUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("_codecs", "encode"):
            from codecs import encode

            return encode
        raise DatasetIntegrityError(
            f"RFW aligned BIN requested forbidden pickle global: {module}.{name}"
        )


def _load_member_payload(
    archive: tarfile.TarFile,
    member_name: str,
) -> tuple[list[bytes], list[bool]]:
    try:
        member = archive.getmember(member_name)
    except KeyError as exc:
        raise DatasetIntegrityError(
            f"RFW aligned BIN member is missing: {member_name}"
        ) from exc
    source = archive.extractfile(member)
    if source is None:
        raise DatasetIntegrityError(
            f"RFW aligned BIN member is unreadable: {member_name}"
        )
    with source:
        serialized = source.read()
    _validate_pickle_program(serialized, member_name=member_name)
    try:
        payload: Any = _RestrictedRFWUnpickler(
            BytesIO(serialized), encoding="bytes"
        ).load()
    except Exception as exc:
        raise DatasetIntegrityError(
            f"RFW aligned BIN payload could not be decoded: {member_name}"
        ) from exc
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise DatasetIntegrityError(
            f"RFW aligned BIN payload must be a two-item tuple: {member_name}"
        )
    encoded, labels = payload
    if not isinstance(encoded, list) or not all(
        isinstance(value, bytes) for value in encoded
    ):
        raise DatasetIntegrityError(
            f"RFW aligned BIN images must be a list of bytes: {member_name}"
        )
    if not isinstance(labels, list) or not all(
        isinstance(value, (bool, np.bool_)) for value in labels
    ):
        raise DatasetIntegrityError(
            f"RFW aligned BIN labels must be a list of booleans: {member_name}"
        )
    if len(encoded) != 2 * len(labels):
        raise DatasetIntegrityError(
            f"RFW aligned BIN image/pair counts are inconsistent: {member_name}"
        )
    return encoded, [bool(value) for value in labels]


def _decode_rgb_112(payload: bytes, *, occurrence_id: str) -> np.ndarray:
    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            converted = image.convert("RGB")
            if converted.size != (112, 112):
                raise DatasetIntegrityError(
                    "RFW aligned BIN image must be 112x112: "
                    f"occurrence={occurrence_id}, size={converted.size}"
                )
            array = np.asarray(converted, dtype=np.uint8)
    except DatasetIntegrityError:
        raise
    except Exception as exc:
        raise DatasetIntegrityError(
            f"RFW aligned BIN image decode failed: {occurrence_id}"
        ) from exc
    if array.shape != (112, 112, 3):
        raise DatasetIntegrityError(
            f"RFW aligned BIN RGB shape is invalid: {occurrence_id} {array.shape}"
        )
    return array


def inspect_rfw_aligned_bin_archive(
    archive_path: str | Path,
    *,
    expected_sha256: str | None = None,
    strict_official: bool = True,
) -> RFWAlignedBinSummary:
    path = Path(archive_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    archive_sha256 = _sha256_file(path)
    if expected_sha256 is not None and (
        archive_sha256.lower() != str(expected_sha256).strip().lower()
    ):
        raise DatasetIntegrityError(
            "RFW aligned BIN archive SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={archive_sha256}"
        )
    total_pairs = 0
    total_images = 0
    with tarfile.open(path, mode="r:*") as archive:
        regular_members = {
            member.name for member in archive.getmembers() if member.isfile()
        }
        expected_members = set(_MEMBER_BY_GROUP.values())
        if regular_members != expected_members:
            raise DatasetIntegrityError(
                "RFW aligned BIN archive members differ from the four official "
                f"group files: missing={sorted(expected_members - regular_members)}, "
                f"unexpected={sorted(regular_members - expected_members)}"
            )
        for group in RFW_GROUPS:
            encoded, labels = _load_member_payload(
                archive, _MEMBER_BY_GROUP[group]
            )
            if strict_official and len(labels) != RFW_OFFICIAL_PAIR_COUNT_PER_GROUP:
                raise DatasetIntegrityError(
                    f"RFW aligned BIN {group} must contain 6000 pairs"
                )
            total_pairs += len(labels)
            total_images += len(encoded)
    return RFWAlignedBinSummary(
        archive_path=path,
        archive_sha256=archive_sha256,
        groups=RFW_GROUPS,
        pair_count=total_pairs,
        encoded_image_occurrence_count=total_images,
        image_size=(112, 112),
    )


def iter_rfw_aligned_pair_batches(
    archive_path: str | Path,
    pairs: pd.DataFrame,
    *,
    batch_size: int = 128,
    expected_sha256: str | None = None,
    strict_official: bool = True,
) -> Iterator[RFWAlignedPairBatch]:
    """Yield aligned RGB pair occurrences in official pair order.

    The BIN stores two encoded images per pair and may repeat an image across
    pairs. Occurrence IDs are therefore pair-side IDs rather than claims of
    unique source images.
    """

    missing = sorted(_PAIR_COLUMNS - set(pairs.columns))
    if missing:
        raise ValueError(f"RFW pairs are missing required columns: {missing}")
    if pairs.empty:
        raise ValueError("RFW pairs must be non-empty")
    resolved_batch_size = int(batch_size)
    if isinstance(batch_size, (bool, np.bool_)) or resolved_batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    summary = inspect_rfw_aligned_bin_archive(
        archive_path,
        expected_sha256=expected_sha256,
        strict_official=strict_official,
    )
    path = summary.archive_path
    with tarfile.open(path, mode="r:*") as archive:
        for group in RFW_GROUPS:
            selected = pairs.loc[
                pairs["rfw_group"].astype(str).eq(group)
            ].sort_values("official_index")
            if selected.empty:
                if strict_official:
                    raise DatasetIntegrityError(
                        f"strict RFW aligned evaluation is missing group {group}"
                    )
                continue
            if selected["official_index"].duplicated().any():
                raise DatasetIntegrityError(
                    f"RFW pair official_index is duplicated in group {group}"
                )
            encoded, labels = _load_member_payload(
                archive, _MEMBER_BY_GROUP[group]
            )
            batch_faces: list[np.ndarray] = []
            batch_rows: list[dict[str, Any]] = []
            for row in selected.itertuples(index=False):
                official_index = int(row.official_index)
                if not 0 <= official_index < len(labels):
                    raise DatasetIntegrityError(
                        f"RFW official_index is outside BIN range: {official_index}"
                    )
                if bool(labels[official_index]) != bool(row.is_genuine):
                    raise DatasetIntegrityError(
                        "RFW aligned BIN label differs from JPG protocol: "
                        f"pair_id={row.pair_id}"
                    )
                for side, encoded_index in (
                    ("left", 2 * official_index),
                    ("right", 2 * official_index + 1),
                ):
                    occurrence_id = f"{row.pair_id}:{side}"
                    batch_faces.append(
                        _decode_rgb_112(
                            encoded[encoded_index], occurrence_id=occurrence_id
                        )
                    )
                    batch_rows.append(
                        {
                            "occurrence_id": occurrence_id,
                            "pair_id": str(row.pair_id),
                            "side": side,
                            "rfw_group": group,
                            "fold_index": int(row.fold_index),
                            "official_index": official_index,
                            "is_genuine": bool(row.is_genuine),
                        }
                    )
                    if len(batch_faces) == resolved_batch_size:
                        yield RFWAlignedPairBatch(
                            faces=np.stack(batch_faces).astype(
                                np.uint8, copy=False
                            ),
                            occurrences=pd.DataFrame.from_records(batch_rows),
                        )
                        batch_faces = []
                        batch_rows = []
            if batch_faces:
                yield RFWAlignedPairBatch(
                    faces=np.stack(batch_faces).astype(np.uint8, copy=False),
                    occurrences=pd.DataFrame.from_records(batch_rows),
                )
