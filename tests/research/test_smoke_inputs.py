from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research.embeddings.smoke_inputs import resolve_smoke_input_batch


def _lfw_fixture(tmp_path: Path) -> Path:
    rows: list[dict[str, str]] = []
    for index, color in enumerate(
        ((255, 0, 0), (0, 255, 0), (0, 0, 255), (90, 80, 70))
    ):
        identity = "same" if index < 2 else f"identity-{index}"
        image_path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (250, 250), color=color).save(image_path)
        rows.append(
            {
                "image_id": f"lfw:{index}",
                "identity_id": identity,
                "image_path": str(image_path),
            }
        )
    manifest = tmp_path / "face_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image_id", "identity_id", "image_path"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_resolve_smoke_input_automatically_builds_deterministic_lfw_batch(
    tmp_path,
):
    manifest = _lfw_fixture(tmp_path)

    first = resolve_smoke_input_batch(
        tmp_path,
        source_color_order="rgb",
        lfw_manifest_path=manifest,
        max_images=3,
        seed=42,
    )
    second = resolve_smoke_input_batch(
        tmp_path,
        source_color_order="rgb",
        lfw_manifest_path=manifest,
        max_images=3,
        seed=42,
    )

    assert first.aligned_faces.shape == (3, 112, 112, 3)
    assert first.aligned_faces.dtype == np.uint8
    assert np.array_equal(first.aligned_faces, second.aligned_faces)
    assert first.metadata["smoke_only"] is True
    assert first.metadata["quantitative_experiment_input"] is False
    identities = {
        row["identity_id"] for row in first.metadata["source_files"]
    }
    assert len(identities) == 3


def test_resolve_smoke_input_converts_rgb_fallback_to_declared_bgr(tmp_path):
    manifest = _lfw_fixture(tmp_path)

    rgb = resolve_smoke_input_batch(
        tmp_path,
        source_color_order="rgb",
        lfw_manifest_path=manifest,
        max_images=1,
    )
    bgr = resolve_smoke_input_batch(
        tmp_path,
        source_color_order="bgr",
        lfw_manifest_path=manifest,
        max_images=1,
    )

    assert np.array_equal(rgb.aligned_faces[..., ::-1], bgr.aligned_faces)


def test_resolve_smoke_input_prefers_explicit_npy(tmp_path):
    faces = np.zeros((2, 112, 112, 3), dtype=np.uint8)
    path = tmp_path / "aligned.npy"
    np.save(path, faces)

    result = resolve_smoke_input_batch(
        tmp_path,
        source_color_order="rgb",
        explicit_path=path,
        max_images=1,
    )

    assert result.aligned_faces.shape == (1, 112, 112, 3)
    assert result.metadata["source_type"] == "explicit_aligned_array"


def test_resolve_smoke_input_explains_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="data_preparation.ipynb"):
        resolve_smoke_input_batch(
            tmp_path,
            source_color_order="rgb",
        )
