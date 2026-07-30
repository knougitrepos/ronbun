from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pandas as pd
import pytest
import research.datasets.balancedface as balancedface_module

from research.datasets import (
    BALANCEDFACE_RACE_LABELS,
    DatasetIntegrityError,
    build_balancedface_index_bundle,
    select_balancedface_index_scope,
    write_balancedface_index_bundle,
)


def _make_list(path: Path, *, reverse: bool = False) -> list[str]:
    rows: list[str] = []
    for race_label in BALANCEDFACE_RACE_LABELS:
        for identity_index in range(3):
            identity = f"m.group{race_label}_{identity_index}"
            for image_index in range(2):
                rows.append(
                    f"{identity}/{image_index:06d}.jpg\t"
                    f"{race_label * 10 + identity_index}\t{race_label}"
                )
    if reverse:
        rows.reverse()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return rows


def _make_recordio_archive(path: Path) -> None:
    members = {
        "Equalizedface/property": b"27999,112,112",
        "Equalizedface/train.idx": b"idx",
        "Equalizedface/train.rec": b"rec",
    }
    with tarfile.open(path, mode="w:gz") as handle:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))


def test_balancedface_bundle_excludes_rfw_overlap_before_identity_split(
    tmp_path: Path,
):
    list_path = tmp_path / "data" / "raw" / "balanced" / "train.lst"
    _make_list(list_path)
    recordio = list_path.parent / "Equalizedface.tar.gz"
    _make_recordio_archive(recordio)

    bundle = build_balancedface_index_bundle(
        list_path,
        recordio,
        tmp_path,
        rfw_source_identity_ids={"m.group0_0", "m.not_present"},
        seed=9,
        development_fraction=0.50,
        strict_official=False,
    )

    assert bundle.summary["excluded_rfw_overlap_identity_count"] == 1
    assert bundle.summary["excluded_rfw_overlap_image_count"] == 2
    assert "m.group0_0" not in set(bundle.manifest["source_identity_id"])
    assert set(bundle.manifest["split"]) == {"development", "calibration"}
    assert "test" not in set(bundle.manifest["split"])
    assert bundle.manifest.groupby("identity_id")["split"].nunique().max() == 1
    assert set(bundle.manifest["storage_kind"]) == {
        "mxnet_recordio_pending_decoder"
    }
    assert bundle.summary["final_evaluation_allowed"] is False
    assert bundle.summary["recordio_decoder_implemented"] is False
    assert bundle.summary["recordio_archive_integrity_verified"] is True
    assert bundle.summary["source_row_index_is_recordio_key"] is False

    paths = write_balancedface_index_bundle(bundle, tmp_path / "interim")
    assert pd.read_csv(paths["source_index_manifest.csv"]).shape[0] == 22
    assert pd.read_csv(paths["excluded_rfw_overlap_identities.csv"]).shape[0] == 1
    assert paths["_SUCCESS"].is_file()


def test_balancedface_identity_split_is_independent_of_input_row_order(
    tmp_path: Path,
):
    first_list = tmp_path / "first" / "train.lst"
    second_list = tmp_path / "second" / "train.lst"
    _make_list(first_list)
    _make_list(second_list, reverse=True)
    first_recordio = first_list.parent / "Equalizedface.tar.gz"
    second_recordio = second_list.parent / "Equalizedface.tar.gz"
    _make_recordio_archive(first_recordio)
    _make_recordio_archive(second_recordio)

    first = build_balancedface_index_bundle(
        first_list,
        first_recordio,
        tmp_path,
        rfw_source_identity_ids={"m.not_present"},
        seed=11,
        development_fraction=0.50,
        strict_official=False,
    )
    second = build_balancedface_index_bundle(
        second_list,
        second_recordio,
        tmp_path,
        rfw_source_identity_ids={"m.not_present"},
        seed=11,
        development_fraction=0.50,
        strict_official=False,
    )

    first_splits = (
        first.manifest[["source_identity_id", "split"]]
        .drop_duplicates()
        .sort_values("source_identity_id")
        .reset_index(drop=True)
    )
    second_splits = (
        second.manifest[["source_identity_id", "split"]]
        .drop_duplicates()
        .sort_values("source_identity_id")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(first_splits, second_splits)

    subset = select_balancedface_index_scope(
        first,
        mode="dev",
        data_fraction=0.50,
        seed=5,
    )
    assert subset.summary["full_source_index"] is False
    assert subset.summary["strict_official_index_validated"] is False
    assert subset.summary["selected_image_count"] == len(subset.manifest)
    assert set(subset.manifest["split"]) == {"development", "calibration"}
    assert subset.manifest.groupby("identity_id")["split"].nunique().max() == 1
    with pytest.raises(DatasetIntegrityError, match="full strict source index"):
        select_balancedface_index_scope(
            subset,
            mode="real",
            data_fraction=1.0,
            seed=5,
        )


@pytest.mark.parametrize(
    "unsafe_path",
    ["identity//image.jpg", "identity/./image.jpg", "identity/D:evil.jpg"],
)
def test_balancedface_list_rejects_non_normalized_windows_paths(
    tmp_path: Path,
    unsafe_path: str,
):
    list_path = tmp_path / "train.lst"
    list_path.write_text(f"{unsafe_path}\t0\t0\n", encoding="utf-8")
    recordio = tmp_path / "Equalizedface.tar.gz"
    _make_recordio_archive(recordio)

    with pytest.raises(DatasetIntegrityError, match="unsafe or unexpected"):
        build_balancedface_index_bundle(
            list_path,
            recordio,
            tmp_path,
            rfw_source_identity_ids={"m.not_present"},
            strict_official=False,
        )


def test_balancedface_overlap_exclusion_cannot_remove_an_entire_group(
    tmp_path: Path,
):
    list_path = tmp_path / "train.lst"
    _make_list(list_path)
    recordio = tmp_path / "Equalizedface.tar.gz"
    _make_recordio_archive(recordio)

    with pytest.raises(DatasetIntegrityError, match="leave all four groups"):
        build_balancedface_index_bundle(
            list_path,
            recordio,
            tmp_path,
            rfw_source_identity_ids={
                "m.group0_0",
                "m.group0_1",
                "m.group0_2",
            },
            strict_official=False,
        )


def test_balancedface_overwrite_invalidates_old_success_before_writing(
    tmp_path: Path,
    monkeypatch,
):
    list_path = tmp_path / "train.lst"
    _make_list(list_path)
    recordio = tmp_path / "Equalizedface.tar.gz"
    _make_recordio_archive(recordio)
    bundle = build_balancedface_index_bundle(
        list_path,
        recordio,
        tmp_path,
        rfw_source_identity_ids={"m.not_present"},
        strict_official=False,
    )
    output_dir = tmp_path / "interim"
    paths = write_balancedface_index_bundle(bundle, output_dir)
    assert paths["_SUCCESS"].is_file()

    def fail_write(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(balancedface_module, "_atomic_write_csv", fail_write)
    with pytest.raises(RuntimeError, match="simulated"):
        write_balancedface_index_bundle(bundle, output_dir, overwrite=True)
    assert not paths["_SUCCESS"].exists()
