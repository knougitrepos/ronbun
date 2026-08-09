from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.experiments import rfw_custom_pipeline
from research.experiments.rfw_custom_pipeline import (
    RFW_CUSTOM_ALIGNED_BIN_MODE,
    materialize_rfw_custom_aligned_bundle,
)
from research.preprocessing.aligned_crops import validate_aligned_crop_bundle


def _source_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["i1", "i2", "i3"],
            "identity_id": ["p1", "p1", "p2"],
            "split": ["test", "test", "test"],
            "image_path": ["source/i1.jpg", "source/i2.jpg", "source/i3.jpg"],
            "protocol_uid": ["rfw-custom-test"] * 3,
            "official_result_eligible": [False] * 3,
        }
    )


def _install_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duplicate_mismatch: bool = False,
) -> None:
    pairs = pd.DataFrame(
        {
            "pair_id": ["pair-1", "pair-2"],
            "left_image_id": ["i1", "i1"],
            "right_image_id": ["i2", "i3"],
        }
    )
    official = SimpleNamespace(
        manifest=pd.DataFrame({"image_id": ["i1", "i2", "i3"]}),
        pairs=pairs,
        summary={"source_archive_sha256": "A" * 64},
    )
    monkeypatch.setattr(
        rfw_custom_pipeline,
        "build_rfw_verification_bundle",
        lambda *args, **kwargs: official,
    )
    face_1 = np.full((112, 112, 3), 1, dtype=np.uint8)
    repeated = np.full(
        (112, 112, 3),
        9 if duplicate_mismatch else 1,
        dtype=np.uint8,
    )
    faces = np.stack(
        [
            face_1,
            np.full((112, 112, 3), 2, dtype=np.uint8),
            repeated,
            np.full((112, 112, 3), 3, dtype=np.uint8),
        ]
    )
    occurrences = pd.DataFrame(
        {
            "pair_id": ["pair-1", "pair-1", "pair-2", "pair-2"],
            "side": ["left", "right", "left", "right"],
        }
    )
    monkeypatch.setattr(
        rfw_custom_pipeline,
        "iter_rfw_aligned_pair_batches",
        lambda *args, **kwargs: iter(
            [SimpleNamespace(faces=faces, occurrences=occurrences)]
        ),
    )


def test_rfw_custom_materializer_deduplicates_pair_occurrences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_sources(monkeypatch)
    jpg = tmp_path / "test.tar.gz"
    aligned = tmp_path / "RFW_test.tar.gz"
    jpg.write_bytes(b"jpg")
    aligned.write_bytes(b"aligned")
    output = tmp_path / "aligned_112"

    manifest = materialize_rfw_custom_aligned_bundle(
        _source_manifest(),
        project_root=tmp_path,
        jpg_archive_path=jpg,
        aligned_bin_archive_path=aligned,
        output_dir=output,
        expected_jpg_archive_sha256="a" * 64,
        expected_aligned_bin_archive_sha256="b" * 64,
    )

    assert manifest["official_protocol"] is False
    assert manifest["checkpoint_overlap_status"] == "UNKNOWN"
    faces = np.load(output / "aligned_faces.npy", allow_pickle=False)
    assert faces.shape == (3, 112, 112, 3)
    assert faces[:, 0, 0, 0].tolist() == [1, 2, 3]
    index = pd.read_csv(output / "aligned_index.csv")
    assert index["sample_id"].tolist() == ["i1", "i2", "i3"]
    validate_aligned_crop_bundle(
        output,
        dataset_id="rfw_custom",
        expected_source_count=3,
        preprocessing_mode=RFW_CUSTOM_ALIGNED_BIN_MODE,
        require_full_coverage=True,
    )


def test_rfw_custom_materializer_rejects_inconsistent_repeated_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_sources(monkeypatch, duplicate_mismatch=True)
    jpg = tmp_path / "test.tar.gz"
    aligned = tmp_path / "RFW_test.tar.gz"
    jpg.write_bytes(b"jpg")
    aligned.write_bytes(b"aligned")

    with pytest.raises(ValueError, match="decode differently"):
        materialize_rfw_custom_aligned_bundle(
            _source_manifest(),
            project_root=tmp_path,
            jpg_archive_path=jpg,
            aligned_bin_archive_path=aligned,
            output_dir=tmp_path / "aligned_112",
            expected_jpg_archive_sha256="a" * 64,
            expected_aligned_bin_archive_sha256="b" * 64,
        )
