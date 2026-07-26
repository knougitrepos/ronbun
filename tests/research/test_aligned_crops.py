from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from research.preprocessing.aligned_crops import materialize_aligned_crops
from research.runtime.hashing import sha256_file


@dataclass
class _Face:
    det_score: float
    bbox: np.ndarray
    kps: np.ndarray


class _Detector:
    def get(self, image_bgr: np.ndarray) -> list[_Face]:
        if int(image_bgr[0, 0, 2]) == 0:
            return []
        landmarks = np.array(
            [[30, 40], [80, 40], [56, 60], [38, 82], [74, 82]],
            dtype=np.float32,
        )
        return [
            _Face(0.8, np.array([10, 10, 90, 100]), landmarks),
            _Face(0.9, np.array([15, 15, 95, 105]), landmarks),
        ]


def _write_image(path: Path, red: int) -> None:
    array = np.zeros((120, 120, 3), dtype=np.uint8)
    array[..., 0] = red
    Image.fromarray(array, mode="RGB").save(path)


def _manifest(tmp_path: Path) -> pd.DataFrame:
    _write_image(tmp_path / "ok.png", 10)
    _write_image(tmp_path / "no_face.png", 0)
    return pd.DataFrame(
        {
            "image_id": ["ok", "none", "missing"],
            "identity_id": ["person-1", "person-2", "person-3"],
            "split": ["test", "test", "test"],
            "image_path": ["ok.png", "no_face.png", "missing.png"],
        }
    )


def test_materializer_writes_interpretable_index_and_single_bundle(
    tmp_path: Path,
) -> None:
    output = tmp_path / "aligned"
    result = materialize_aligned_crops(
        _manifest(tmp_path),
        project_root=tmp_path,
        output_dir=output,
        dataset_id="lfw",
        detector=_Detector(),
        aligner=lambda image, landmarks: np.full((112, 112, 3), image[0, 0]),
    )

    assert result.aligned_faces.shape == (1, 112, 112, 3)
    assert result.aligned_faces.dtype == np.uint8
    assert result.aligned_index["sample_id"].tolist() == ["ok"]
    assert result.aligned_index["selected_face_index"].tolist() == [1]
    assert set(result.failed_index["alignment_failure_reason"]) == {
        "no_face_detected",
        "source_file_not_found",
    }
    assert (output / "_SUCCESS").is_file()
    assert (output / "aligned_index.csv").is_file()
    assert (output / "failed_samples.csv").is_file()
    manifest = json.loads((output / "bundle_manifest.json").read_text("utf-8"))
    for entry in manifest["outputs"].values():
        artifact = output / entry["path"]
        assert artifact.is_file()
        assert sha256_file(artifact) == entry["sha256"]


def test_materializer_overwrite_replaces_canonical_bundle(tmp_path: Path) -> None:
    output = tmp_path / "aligned"
    manifest = _manifest(tmp_path).iloc[[0]].copy()
    first = materialize_aligned_crops(
        manifest,
        project_root=tmp_path,
        output_dir=output,
        dataset_id="lfw",
        detector=_Detector(),
        aligner=lambda image, landmarks: np.zeros((112, 112, 3), dtype=np.uint8),
    )
    first_hash = first.aligned_index.loc[0, "aligned_content_sha256"]

    second = materialize_aligned_crops(
        manifest,
        project_root=tmp_path,
        output_dir=output,
        dataset_id="lfw",
        detector=_Detector(),
        aligner=lambda image, landmarks: np.ones((112, 112, 3), dtype=np.uint8),
        overwrite=True,
    )
    assert second.aligned_index.loc[0, "aligned_content_sha256"] != first_hash
    assert len(list(tmp_path.glob(".aligned.backup-*"))) == 0
    assert len(list(tmp_path.glob(".aligned.staging-*"))) == 0


def test_materializer_validates_before_loading_insightface(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="required columns"):
        materialize_aligned_crops(
            pd.DataFrame(),
            project_root=tmp_path,
            output_dir=tmp_path / "out",
            dataset_id="lfw",
        )

    duplicate = pd.DataFrame(
        {
            "image_id": ["same", "same"],
            "identity_id": ["a", "b"],
            "split": ["test", "test"],
            "image_path": ["a.png", "b.png"],
        }
    )
    with pytest.raises(ValueError, match="must be unique"):
        materialize_aligned_crops(
            duplicate,
            project_root=tmp_path,
            output_dir=tmp_path / "out",
            dataset_id="lfw",
        )
