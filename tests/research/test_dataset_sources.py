from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from research.datasets import (
    DatasetIntegrityError,
    inspect_balancedface_sources,
    inspect_rfw_sources,
    inspect_tar_archive,
    require_valid_tar,
    safe_extract_tar_archive,
)


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w:gz") as handle:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))


def test_source_inventory_identifies_alternative_representations(tmp_path: Path):
    rfw_root = tmp_path / "data" / "raw" / "RFW"
    _write_tar(rfw_root / "images" / "test.tar.gz", {"test/file.txt": b"x"})
    _write_tar(
        rfw_root / "bin_for_mxnet" / "RFW_test.tar.gz",
        {"African_test.bin": b"x"},
    )
    (rfw_root / "readme.txt").write_text("RFW", encoding="utf-8")

    balanced_root = tmp_path / "data" / "raw" / "RFW-balancedface"
    _write_tar(
        balanced_root / "images" / "Equalizedface.tar.gz",
        {"race_per_7000/Caucasian/a/a.jpg": b"x"},
    )
    _write_tar(
        balanced_root / "rec_for_mxnet" / "Equalizedface.tar.gz",
        {"Equalizedface/property": b"1,112,112"},
    )
    (balanced_root / "rec_for_mxnet" / "train_balancedface.lst").write_text(
        "a/a.jpg\t0\t0\n",
        encoding="utf-8",
    )

    rfw = inspect_rfw_sources(rfw_root, project_root=tmp_path)
    balanced = inspect_balancedface_sources(balanced_root, project_root=tmp_path)

    assert rfw.summary["double_count_if_combined"] is True
    assert balanced.summary["double_count_if_combined"] is True
    assert all(artifact.present for artifact in rfw.artifacts)
    assert all(artifact.present for artifact in balanced.artifacts)


def test_tar_inspection_and_extraction_reject_path_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.tar.gz"
    _write_tar(archive, {"../escape.txt": b"escape"})

    report = inspect_tar_archive(archive)

    assert report.valid is False
    assert report.unsafe_members
    with pytest.raises(DatasetIntegrityError, match="unsafe members"):
        require_valid_tar(report)
    with pytest.raises(DatasetIntegrityError):
        safe_extract_tar_archive(archive, tmp_path / "output")
    assert not (tmp_path / "escape.txt").exists()


def test_safe_tar_extraction_refuses_existing_target(tmp_path: Path):
    archive = tmp_path / "safe.tar.gz"
    _write_tar(archive, {"test/file.txt": b"safe"})
    target = tmp_path / "output"
    target.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        safe_extract_tar_archive(
            archive,
            target,
            expected_prefixes=("test",),
        )


def test_safe_tar_extraction_writes_only_expected_prefix(tmp_path: Path):
    archive = tmp_path / "safe.tar.gz"
    _write_tar(archive, {"test/nested/file.txt": b"safe"})

    target = safe_extract_tar_archive(
        archive,
        tmp_path / "output",
        expected_prefixes="test",
    )

    assert (target / "test" / "nested" / "file.txt").read_bytes() == b"safe"


def test_tar_inspection_rejects_wrong_or_empty_expected_prefix(tmp_path: Path):
    archive = tmp_path / "safe.tar.gz"
    _write_tar(archive, {"test/file.txt": b"safe"})

    wrong = inspect_tar_archive(archive, expected_prefixes=("other",))

    assert wrong.valid is False
    assert any("outside expected prefixes" in item for item in wrong.unsafe_members)
    with pytest.raises(ValueError, match="invalid expected tar prefix"):
        inspect_tar_archive(archive, expected_prefixes=("",))


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "test/NUL.txt",
        "test/D:escape.txt",
        "test/file.txt:stream",
        "test/trailing./file.txt",
        "test//double.txt",
    ],
)
def test_tar_inspection_rejects_windows_unsafe_names(
    tmp_path: Path,
    unsafe_name: str,
):
    archive = tmp_path / "unsafe-windows.tar.gz"
    _write_tar(archive, {unsafe_name: b"x"})

    report = inspect_tar_archive(archive)

    assert report.valid is False
    assert report.unsafe_members


def test_tar_inspection_rejects_case_insensitive_collision(tmp_path: Path):
    archive = tmp_path / "case-collision.tar.gz"
    _write_tar(
        archive,
        {
            "test/File.txt": b"one",
            "test/file.txt": b"two",
        },
    )

    report = inspect_tar_archive(archive)

    assert report.valid is False
    assert any("case-insensitive collision" in item for item in report.unsafe_members)
