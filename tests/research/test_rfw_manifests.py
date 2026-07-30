from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pandas as pd
import pytest
import research.datasets.rfw as rfw_module

from research.datasets import (
    DatasetIntegrityError,
    RFW_GROUPS,
    build_rfw_verification_bundle,
    select_rfw_protocol_scope,
    write_rfw_verification_bundle,
)


def _add_member(handle: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    handle.addfile(info, io.BytesIO(payload))


def _make_rfw_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with tarfile.open(path, mode="w:gz") as handle:
        for group_index, group in enumerate(RFW_GROUPS):
            first = f"m.fixture{group_index}a"
            second = f"m.fixture{group_index}b"
            filenames = [
                f"{first}_0001.jpg",
                f"{first}_0002.jpg",
                f"{second}_0001.jpg",
                f"{second}_0002.jpg",
            ]
            image_lines = [
                f"{filenames[0]}\t0",
                f"{filenames[1]}\t0",
                f"{filenames[2]}\t1",
                f"{filenames[3]}\t1",
            ]
            landmark_lines = []
            for label, filename in enumerate(filenames):
                identity = filename.rsplit("_", 1)[0]
                source_label = 0 if label < 2 else 1
                member = f"test/data/{group}/{identity}/{filename}"
                _add_member(handle, member, b"jpeg")
                landmark_lines.append(
                    f"/{member}\t{source_label}\t"
                    "10\t10\t20\t10\t15\t15\t11\t20\t19\t20"
                )
            base = f"test/txts/{group}"
            _add_member(
                handle,
                f"{base}/{group}_images.txt",
                ("\n".join(image_lines) + "\n").encode(),
            )
            _add_member(
                handle,
                f"{base}/{group}_lmk.txt",
                ("\n".join(landmark_lines) + "\n").encode(),
            )
            _add_member(
                handle,
                f"{base}/{group}_people.txt",
                f"{first}\t2\n{second}\t2\n".encode(),
            )
            _add_member(
                handle,
                f"{base}/{group}_pairs.txt",
                (
                    f"{first}\t1\t2\n"
                    f"{second}\t1\t2\n"
                    f"{first}\t1\t{second}\t1\n"
                    f"{first}\t2\t{second}\t2\n"
                ).encode(),
            )


def test_rfw_bundle_preserves_group_pair_and_landmark_roles(tmp_path: Path):
    archive = tmp_path / "data" / "raw" / "RFW" / "images" / "test.tar.gz"
    _make_rfw_fixture(archive)

    bundle = build_rfw_verification_bundle(
        archive,
        tmp_path,
        strict_official=False,
    )

    assert len(bundle.manifest) == 16
    assert len(bundle.pairs) == 16
    assert len(bundle.landmarks) == 16
    assert set(bundle.manifest["rfw_group"]) == set(RFW_GROUPS)
    assert set(bundle.manifest["split"]) == {"test"}
    assert set(bundle.manifest["dataset_role"]) == {"evaluation_test_only"}
    assert bundle.pairs.groupby("rfw_group")["is_genuine"].sum().eq(2).all()
    assert bundle.summary["compressor_fit_allowed"] is False
    assert bundle.summary["official_open_set_protocol"] is False
    assert all(path.startswith("tar://") for path in bundle.manifest["image_path"])

    output = write_rfw_verification_bundle(bundle, tmp_path / "interim")
    assert pd.read_csv(output["image_manifest.csv"]).shape[0] == 16
    assert pd.read_csv(output["pair_protocol.csv"]).shape[0] == 16
    assert output["_SUCCESS"].is_file()


def test_rfw_bundle_is_deterministic(tmp_path: Path):
    archive = tmp_path / "data" / "raw" / "RFW" / "images" / "test.tar.gz"
    _make_rfw_fixture(archive)

    first = build_rfw_verification_bundle(
        archive,
        tmp_path,
        strict_official=False,
    )
    second = build_rfw_verification_bundle(
        archive,
        tmp_path,
        strict_official=False,
    )

    pd.testing.assert_frame_equal(first.manifest, second.manifest)
    pd.testing.assert_frame_equal(first.pairs, second.pairs)
    pd.testing.assert_frame_equal(first.landmarks, second.landmarks)


def test_rfw_dev_scope_is_group_fold_label_stratified(tmp_path: Path):
    archive = tmp_path / "data" / "raw" / "RFW" / "images" / "test.tar.gz"
    _make_rfw_fixture(archive)
    bundle = build_rfw_verification_bundle(
        archive,
        tmp_path,
        strict_official=False,
    )

    subset = select_rfw_protocol_scope(
        bundle,
        mode="dev",
        data_fraction=0.50,
        seed=7,
    )

    assert len(subset.pairs) == 8
    assert (
        subset.pairs.groupby(["rfw_group", "is_genuine"]).size().eq(1).all()
    )
    assert subset.summary["official_result_eligible"] is False
    assert subset.summary["official_protocol_validated"] is False
    assert subset.summary["pair_count"] == len(subset.pairs)
    assert subset.summary["selection_strata"] == [
        "rfw_group",
        "fold_index",
        "is_genuine",
    ]
    with pytest.raises(DatasetIntegrityError, match="strict official"):
        select_rfw_protocol_scope(
            bundle,
            mode="real",
            data_fraction=1.0,
            seed=7,
        )
    with pytest.raises(DatasetIntegrityError, match="strict official"):
        select_rfw_protocol_scope(
            subset,
            mode="real",
            data_fraction=1.0,
            seed=7,
        )


def test_rfw_overwrite_invalidates_old_success_before_writing(
    tmp_path: Path,
    monkeypatch,
):
    archive = tmp_path / "data" / "raw" / "RFW" / "images" / "test.tar.gz"
    _make_rfw_fixture(archive)
    bundle = build_rfw_verification_bundle(
        archive,
        tmp_path,
        strict_official=False,
    )
    output_dir = tmp_path / "interim"
    paths = write_rfw_verification_bundle(bundle, output_dir)
    assert paths["_SUCCESS"].is_file()

    def fail_write(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(rfw_module, "_atomic_write_csv", fail_write)
    with pytest.raises(RuntimeError, match="simulated"):
        write_rfw_verification_bundle(bundle, output_dir, overwrite=True)
    assert not paths["_SUCCESS"].exists()
