from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import uuid
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable


class DatasetIntegrityError(ValueError):
    """Raised when a dataset source is incomplete, unsafe, or inconsistent."""


@dataclass(frozen=True)
class SourceArtifact:
    """One physical representation of a logical dataset."""

    dataset: str
    logical_content: str
    representation: str
    path: str
    present: bool
    byte_count: int | None
    sha256: str | None
    canonical_candidate: bool
    note: str


@dataclass(frozen=True)
class DatasetSourceInventory:
    """Physical files found for one logical dataset."""

    dataset: str
    artifacts: tuple[SourceArtifact, ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
            "summary": self.summary,
        }


@dataclass(frozen=True)
class TarInspection:
    """Read-only integrity and path-safety result for a tar archive."""

    archive_path: str
    readable_to_eof: bool
    member_count: int
    regular_file_count: int
    directory_count: int
    total_member_bytes: int
    unsafe_members: tuple[str, ...]
    duplicate_members: tuple[str, ...]
    error: str | None

    @property
    def valid(self) -> bool:
        return (
            self.readable_to_eof
            and not self.unsafe_members
            and not self.duplicate_members
            and self.error is None
        )


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 of a file without loading it into memory."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source file not found: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _relative_or_absolute(path: Path, project_root: Path | None) -> str:
    resolved = path.expanduser().resolve()
    if project_root is None:
        return str(resolved)
    try:
        return resolved.relative_to(project_root.expanduser().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _source_artifact(
    *,
    dataset: str,
    logical_content: str,
    representation: str,
    path: Path,
    project_root: Path | None,
    verify_sha256: bool,
    canonical_candidate: bool,
    note: str,
) -> SourceArtifact:
    present = path.is_file()
    return SourceArtifact(
        dataset=dataset,
        logical_content=logical_content,
        representation=representation,
        path=_relative_or_absolute(path, project_root),
        present=present,
        byte_count=int(path.stat().st_size) if present else None,
        sha256=sha256_file(path) if present and verify_sha256 else None,
        canonical_candidate=canonical_candidate,
        note=note,
    )


def inspect_rfw_sources(
    rfw_root: str | Path,
    *,
    project_root: str | Path | None = None,
    verify_sha256: bool = False,
) -> DatasetSourceInventory:
    """Identify the local RFW representations without combining them."""

    root = Path(rfw_root).expanduser().resolve()
    project = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else None
    )
    artifacts = (
        _source_artifact(
            dataset="rfw-v1",
            logical_content="rfw_official_test",
            representation="jpg_loose_with_protocol",
            path=root / "images" / "test.tar.gz",
            project_root=project,
            verify_sha256=verify_sha256,
            canonical_candidate=True,
            note="400x400 loose JPG plus official pairs and landmarks",
        ),
        _source_artifact(
            dataset="rfw-v1",
            logical_content="rfw_official_test",
            representation="aligned_pair_bin",
            path=root / "bin_for_mxnet" / "RFW_test.tar.gz",
            project_root=project,
            verify_sha256=verify_sha256,
            canonical_candidate=True,
            note="112x112 aligned pair BIN; alternative representation of the same test set",
        ),
        _source_artifact(
            dataset="rfw-v1",
            logical_content="rfw_documentation",
            representation="readme",
            path=root / "readme.txt",
            project_root=project,
            verify_sha256=verify_sha256,
            canonical_candidate=False,
            note="provider protocol and overlap warning",
        ),
    )
    present_representations = [
        artifact.representation
        for artifact in artifacts
        if artifact.present and artifact.logical_content == "rfw_official_test"
    ]
    return DatasetSourceInventory(
        dataset="rfw-v1",
        artifacts=artifacts,
        summary={
            "root": _relative_or_absolute(root, project),
            "logical_dataset_count": 1,
            "present_test_representations": present_representations,
            "representations_are_alternatives": True,
            "double_count_if_combined": len(present_representations) > 1,
        },
    )


def inspect_balancedface_sources(
    balancedface_root: str | Path,
    *,
    project_root: str | Path | None = None,
    verify_sha256: bool = False,
) -> DatasetSourceInventory:
    """Identify local Equalizedface JPG and RecordIO alternatives."""

    root = Path(balancedface_root).expanduser().resolve()
    project = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else None
    )
    artifacts = (
        _source_artifact(
            dataset="bupt-balancedface-equalizedface",
            logical_content="balancedface_training",
            representation="jpg_archive",
            path=root / "images" / "Equalizedface.tar.gz",
            project_root=project,
            verify_sha256=verify_sha256,
            canonical_candidate=True,
            note="named-race JPG tree; archive integrity must pass before use",
        ),
        _source_artifact(
            dataset="bupt-balancedface-equalizedface",
            logical_content="balancedface_training",
            representation="mxnet_recordio_aligned",
            path=root / "rec_for_mxnet" / "Equalizedface.tar.gz",
            project_root=project,
            verify_sha256=verify_sha256,
            canonical_candidate=True,
            note="112x112 aligned RecordIO; alternative representation of the same images",
        ),
        _source_artifact(
            dataset="bupt-balancedface-equalizedface",
            logical_content="balancedface_metadata",
            representation="mxnet_list_index",
            path=root / "rec_for_mxnet" / "train_balancedface.lst",
            project_root=project,
            verify_sha256=verify_sha256,
            canonical_candidate=False,
            note="image, identity-label, and race-label metadata for RecordIO",
        ),
    )
    present_representations = [
        artifact.representation
        for artifact in artifacts
        if artifact.present and artifact.logical_content == "balancedface_training"
    ]
    return DatasetSourceInventory(
        dataset="bupt-balancedface-equalizedface",
        artifacts=artifacts,
        summary={
            "root": _relative_or_absolute(root, project),
            "logical_dataset_count": 1,
            "present_training_representations": present_representations,
            "representations_are_alternatives": True,
            "double_count_if_combined": len(present_representations) > 1,
        },
    )


_WINDOWS_RESERVED_NAME = re.compile(
    r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$",
    flags=re.IGNORECASE,
)


def _unsafe_tar_reason(name: str, member: tarfile.TarInfo) -> str | None:
    if not name or "\x00" in name or "\\" in name:
        return "empty, NUL-containing, or backslash path"
    normalized_input = name[:-1] if member.isdir() and name.endswith("/") else name
    raw_parts = normalized_input.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        return "non-normalized path"
    if any(
        ":" in part
        or part.endswith((" ", "."))
        or _WINDOWS_RESERVED_NAME.fullmatch(part) is not None
        for part in raw_parts
    ):
        return "Windows-unsafe path component"
    path = PurePosixPath(normalized_input)
    windows_path = PureWindowsPath(name)
    if path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        return "absolute or drive-qualified path"
    if member.issym() or member.islnk():
        return "link member"
    if not (member.isfile() or member.isdir()):
        return "special member"
    return None


def _normalized_prefixes(
    expected_prefixes: Iterable[str] | None,
) -> tuple[tuple[str, ...], ...]:
    if expected_prefixes is None:
        return ()
    values = (
        (expected_prefixes,)
        if isinstance(expected_prefixes, str)
        else tuple(expected_prefixes)
    )
    prefixes: list[tuple[str, ...]] = []
    for value in values:
        text = str(value)
        parts = text.split("/")
        if (
            not text
            or text.startswith("/")
            or "\\" in text
            or any(part in {"", ".", ".."} for part in parts)
            or any(
                ":" in part
                or part.endswith((" ", "."))
                or _WINDOWS_RESERVED_NAME.fullmatch(part) is not None
                for part in parts
            )
        ):
            raise ValueError(f"invalid expected tar prefix: {value!r}")
        prefixes.append(tuple(parts))
    return tuple(prefixes)


def inspect_tar_archive(
    archive_path: str | Path,
    *,
    expected_prefixes: Iterable[str] | None = None,
) -> TarInspection:
    """Read a tar archive to EOF and reject unsafe or duplicate members.

    This function performs no extraction. A truncated archive returns a report
    with ``readable_to_eof=False`` rather than being silently accepted.
    """

    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"tar archive not found: {archive}")
    prefixes = _normalized_prefixes(expected_prefixes)
    unsafe: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    canonical_seen: dict[str, str] = {}
    member_count = 0
    regular_file_count = 0
    directory_count = 0
    total_member_bytes = 0
    error: str | None = None
    readable_to_eof = False

    try:
        with tarfile.open(archive, mode="r|*") as handle:
            for member in handle:
                member_count += 1
                name = member.name
                if name in seen:
                    duplicates.append(name)
                seen.add(name)
                canonical_name = name.rstrip("/").casefold()
                previous_name = canonical_seen.get(canonical_name)
                if previous_name is not None and previous_name != name.rstrip("/"):
                    unsafe.append(
                        f"{name}: case-insensitive collision with {previous_name}"
                    )
                canonical_seen[canonical_name] = name.rstrip("/")
                reason = _unsafe_tar_reason(name, member)
                if reason is not None:
                    unsafe.append(f"{name}: {reason}")
                elif prefixes:
                    parts = tuple(PurePosixPath(name).parts)
                    if not any(parts[: len(prefix)] == prefix for prefix in prefixes):
                        unsafe.append(f"{name}: outside expected prefixes")
                if member.isfile():
                    regular_file_count += 1
                    total_member_bytes += int(member.size)
                elif member.isdir():
                    directory_count += 1
                handle.members.clear()
            readable_to_eof = True
    except (tarfile.TarError, EOFError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    return TarInspection(
        archive_path=str(archive),
        readable_to_eof=readable_to_eof,
        member_count=member_count,
        regular_file_count=regular_file_count,
        directory_count=directory_count,
        total_member_bytes=total_member_bytes,
        unsafe_members=tuple(unsafe),
        duplicate_members=tuple(duplicates),
        error=error,
    )


def require_valid_tar(report: TarInspection) -> None:
    """Raise with a concise reason unless a tar inspection is fully valid."""

    if report.valid:
        return
    details: list[str] = []
    if not report.readable_to_eof:
        details.append(f"not readable to EOF ({report.error})")
    if report.unsafe_members:
        details.append(f"unsafe members={list(report.unsafe_members[:3])}")
    if report.duplicate_members:
        details.append(f"duplicate members={list(report.duplicate_members[:3])}")
    raise DatasetIntegrityError(
        f"invalid tar archive {report.archive_path}: " + "; ".join(details)
    )


def safe_extract_tar_archive(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    expected_prefixes: Iterable[str] | None = None,
) -> Path:
    """Extract a validated tar into a new directory without overwriting files."""

    archive = Path(archive_path).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite extraction target: {target}")
    report = inspect_tar_archive(archive, expected_prefixes=expected_prefixes)
    require_valid_tar(report)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.extracting")
    temporary.mkdir()
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            seen: set[str] = set()
            canonical_seen: set[str] = set()
            prefixes = _normalized_prefixes(expected_prefixes)
            for member in handle:
                reason = _unsafe_tar_reason(member.name, member)
                canonical_name = member.name.rstrip("/").casefold()
                if (
                    reason is not None
                    or member.name in seen
                    or canonical_name in canonical_seen
                ):
                    raise DatasetIntegrityError(
                        f"archive changed or unsafe member encountered during extraction: "
                        f"{member.name}"
                    )
                seen.add(member.name)
                canonical_seen.add(canonical_name)
                parts = PurePosixPath(member.name).parts
                if prefixes and not any(
                    parts[: len(prefix)] == prefix for prefix in prefixes
                ):
                    raise DatasetIntegrityError(
                        f"member outside expected prefixes during extraction: "
                        f"{member.name}"
                    )
                destination = temporary.joinpath(*parts)
                try:
                    destination.resolve().relative_to(temporary.resolve())
                except ValueError as exc:
                    raise DatasetIntegrityError(
                        f"tar member escapes extraction root: {member.name}"
                    ) from exc
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    handle.members.clear()
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise DatasetIntegrityError(
                        f"failed to read tar member: {member.name}"
                    )
                with source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                handle.members.clear()
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def write_source_inventory(
    inventory: DatasetSourceInventory,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a source inventory without mutating raw data."""

    target = Path(output_path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite source inventory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                inventory.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
