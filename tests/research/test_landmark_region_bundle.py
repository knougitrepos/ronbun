from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.explainability.gradcam.features import summarize_saliency_features
from research.explainability.gradcam.landmark_regions import (
    ARC_FACE_112_LANDMARKS,
    REGION_NAMES,
    build_insightface_106_region_masks,
    materialize_landmark_region_bundle,
    read_landmark_region_bundle,
)


def _ellipse(
    center: tuple[float, float],
    radii: tuple[float, float],
    count: int,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack(
        [
            center[0] + radii[0] * np.cos(angles),
            center[1] + radii[1] * np.sin(angles),
        ]
    ).astype(np.float32)


def _dense_landmarks() -> np.ndarray:
    points = np.zeros((106, 2), dtype=np.float32)
    points[0:33] = _ellipse((56.0, 66.0), (38.0, 50.0), 33)
    points[33:43] = _ellipse((38.0, 52.0), (7.0, 3.0), 10)
    points[43:52] = _ellipse((38.0, 44.0), (10.0, 2.0), 9)
    points[52:72] = _ellipse((56.0, 92.0), (14.0, 7.0), 20)
    points[72:87] = _ellipse((56.0, 72.0), (7.0, 13.0), 15)
    points[87:97] = _ellipse((74.0, 52.0), (7.0, 3.0), 10)
    points[97:106] = _ellipse((74.0, 44.0), (10.0, 2.0), 9)
    return points


def test_dense_landmarks_measure_face_and_multiple_regions_at_cam_resolution():
    masks = build_insightface_106_region_masks(
        ARC_FACE_112_LANDMARKS,
        _dense_landmarks(),
        image_size=(7, 7),
    )

    assert set(masks) == set(REGION_NAMES)
    assert all(mask.shape == (1, 7, 7) for mask in masks.values())
    assert all(mask.any() for mask in masks.values())
    assert not np.any(masks["face"] & masks["outside_face"])
    assert np.all(masks["face"] | masks["outside_face"])

    heatmap = np.where(masks["face"], 1.0, 0.0).astype(np.float32)
    features = summarize_saliency_features(heatmap, region_masks=masks)
    assert features.loc[0, "face_attention"] == pytest.approx(1.0)
    assert features.loc[0, "outside_face_attention"] == pytest.approx(0.0)
    for column in (
        "left_eye_attention",
        "right_eye_attention",
        "nose_attention",
        "mouth_attention",
        "left_cheek_attention",
        "right_cheek_attention",
        "jaw_attention",
    ):
        assert np.isfinite(features.loc[0, column])


class _InjectedLandmarkModel:
    _ronbun_session_providers = ("injected_model",)
    _ronbun_model_sha256 = "injected-test-model"

    def get(self, image_bgr, face):
        assert image_bgr.shape == (112, 112, 3)
        return _dense_landmarks()


def test_landmark_bundle_round_trip_and_sample_order_guard(tmp_path):
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    faces = np.zeros((2, 112, 112, 3), dtype=np.uint8)
    np.save(aligned / "aligned_faces.npy", faces, allow_pickle=False)
    pd.DataFrame(
        {
            "sample_id": ["sample-a", "sample-b"],
            "identity_id": ["identity-a", "identity-b"],
            "split": ["test", "test"],
            "aligned_face_index": [0, 1],
        }
    ).to_csv(aligned / "aligned_index.csv", index=False)
    (aligned / "bundle_manifest.json").write_text(
        json.dumps({"dataset_id": "lfw"}),
        encoding="utf-8",
    )
    (aligned / "_SUCCESS").write_text("complete\n", encoding="utf-8")

    output = tmp_path / "landmark-regions"
    provider = materialize_landmark_region_bundle(
        aligned,
        output_dir=output,
        dataset_id="lfw",
        landmark_model=_InjectedLandmarkModel(),
    )
    reopened = read_landmark_region_bundle(output)

    assert provider.region_mask_uid == reopened.region_mask_uid
    assert reopened.sample_count == 2
    selected = reopened.subset([1], expected_sample_ids=["sample-b"])
    assert selected.sample_count == 1
    masks = selected.build_region_masks([0], image_size=(7, 7))
    assert set(masks) == set(REGION_NAMES)
    with pytest.raises(ValueError, match="do not match expected"):
        reopened.subset([1], expected_sample_ids=["sample-a"])
