from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.explainability.gradcam.features import summarize_saliency_features
from research.explainability.gradcam.landmark_regions import (
    ARC_FACE_112_LANDMARKS,
    FACE_COVERAGE_REVIEW_THRESHOLD,
    INSIGHTFACE_106_TO_5_INDICES,
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


def test_low_face_coverage_is_valid_for_small_surveillance_faces():
    center = np.asarray([56.0, 70.0], dtype=np.float32)
    compact = center + (_dense_landmarks() - center) * 0.60

    masks = build_insightface_106_region_masks(
        ARC_FACE_112_LANDMARKS,
        compact,
    )

    assert all(mask.any() for mask in masks.values())
    assert float(masks["face"].mean()) < FACE_COVERAGE_REVIEW_THRESHOLD


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

    mask_index = pd.read_csv(output / "mask_index.csv")
    assert mask_index["landmark_quality_status"].tolist() == ["ok", "ok"]


class _InvalidLandmarkModel:
    _ronbun_session_providers = ("injected_model",)
    _ronbun_model_sha256 = "injected-invalid-model"

    def get(self, image_bgr, face):
        return np.zeros((106, 2), dtype=np.float32)


class _CompactLandmarkModel:
    _ronbun_session_providers = ("injected_model",)
    _ronbun_model_sha256 = "injected-compact-model"

    def get(self, image_bgr, face):
        center = np.asarray([56.0, 70.0], dtype=np.float32)
        return center + (_dense_landmarks() - center) * 0.60


class _MissingCheekLandmarkModel:
    _ronbun_session_providers = ("injected_model",)
    _ronbun_model_sha256 = "injected-missing-cheek-model"

    def get(self, image_bgr, face):
        points = _dense_landmarks()
        points[33:43] = _ellipse((20.0, 60.0), (35.0, 35.0), 10)
        return points


def test_low_coverage_is_recorded_in_landmark_bundle(tmp_path):
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    np.save(
        aligned / "aligned_faces.npy",
        np.zeros((1, 112, 112, 3), dtype=np.uint8),
        allow_pickle=False,
    )
    pd.DataFrame(
        {
            "sample_id": ["sample-compact"],
            "identity_id": ["identity-compact"],
            "split": ["test"],
            "aligned_face_index": [0],
        }
    ).to_csv(aligned / "aligned_index.csv", index=False)
    (aligned / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "survface",
                "preprocessing": {"mode": "official_face_crop_resize"},
            }
        ),
        encoding="utf-8",
    )
    (aligned / "_SUCCESS").write_text("complete\n", encoding="utf-8")

    output = tmp_path / "landmark-regions"
    materialize_landmark_region_bundle(
        aligned,
        output_dir=output,
        dataset_id="survface",
        landmark_model=_CompactLandmarkModel(),
    )

    mask_index = pd.read_csv(output / "mask_index.csv")
    assert mask_index.loc[0, "landmark_quality_status"] == "low_face_coverage"
    assert (
        mask_index.loc[0, "face_coverage_fraction"]
        < FACE_COVERAGE_REVIEW_THRESHOLD
    )
    manifest = json.loads(
        (output / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["materializer_version"] == "1.4.0"
    assert manifest["quality_control"]["low_face_coverage_count"] == 1
    expected_five = _CompactLandmarkModel().get(None, None)[
        list(INSIGHTFACE_106_TO_5_INDICES)
    ]
    stored_five = np.load(output / "aligned_landmarks_5.npy")
    assert np.allclose(stored_five[0], expected_five)
    assert (
        manifest["landmark_anchor_policy"]
        == "insightface_106_derived_five"
    )


def test_rfw_aligned_bin_bundle_uses_predicted_five_point_anchor(tmp_path):
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    np.save(
        aligned / "aligned_faces.npy",
        np.zeros((1, 112, 112, 3), dtype=np.uint8),
        allow_pickle=False,
    )
    pd.DataFrame(
        {
            "sample_id": ["rfw-sample"],
            "identity_id": ["rfw-identity"],
            "split": ["test"],
            "aligned_face_index": [0],
        }
    ).to_csv(aligned / "aligned_index.csv", index=False)
    (aligned / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "rfw_custom",
                "preprocessing": {"mode": "rfw_official_aligned_bin"},
            }
        ),
        encoding="utf-8",
    )
    (aligned / "_SUCCESS").write_text("complete\n", encoding="utf-8")

    output = tmp_path / "landmark-regions"
    materialize_landmark_region_bundle(
        aligned,
        output_dir=output,
        dataset_id="rfw_custom",
        landmark_model=_InjectedLandmarkModel(),
    )

    manifest = json.loads(
        (output / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_preprocessing_mode"] == "rfw_official_aligned_bin"
    assert manifest["landmark_anchor_policy"] == "insightface_106_derived_five"
    expected_five = _dense_landmarks()[list(INSIGHTFACE_106_TO_5_INDICES)]
    assert np.allclose(np.load(output / "aligned_landmarks_5.npy")[0], expected_five)


def test_missing_semantic_region_is_recorded_without_fabricating_attention(
    tmp_path,
):
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    np.save(
        aligned / "aligned_faces.npy",
        np.zeros((1, 112, 112, 3), dtype=np.uint8),
        allow_pickle=False,
    )
    pd.DataFrame(
        {
            "sample_id": ["sample-missing-cheek"],
            "identity_id": ["identity-missing-cheek"],
            "split": ["test"],
            "aligned_face_index": [0],
        }
    ).to_csv(aligned / "aligned_index.csv", index=False)
    (aligned / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "survface",
                "preprocessing": {"mode": "official_face_crop_resize"},
            }
        ),
        encoding="utf-8",
    )
    (aligned / "_SUCCESS").write_text("complete\n", encoding="utf-8")

    output = tmp_path / "landmark-regions"
    materialize_landmark_region_bundle(
        aligned,
        output_dir=output,
        dataset_id="survface",
        landmark_model=_MissingCheekLandmarkModel(),
    )

    mask_index = pd.read_csv(output / "mask_index.csv")
    assert mask_index.loc[0, "left_cheek_pixel_count"] == 0
    assert mask_index.loc[0, "right_cheek_pixel_count"] > 0
    assert mask_index.loc[0, "semantic_region_mask_count"] == 6
    assert mask_index.loc[0, "missing_semantic_regions"] == "left_cheek"
    assert (
        mask_index.loc[0, "landmark_quality_status"]
        == "missing_semantic_regions"
    )

    provider = read_landmark_region_bundle(output)
    masks = provider.build_region_masks([0], image_size=(7, 7))
    assert not masks["left_cheek"].any()
    assert masks["right_cheek"].any()
    features = summarize_saliency_features(
        np.ones((1, 7, 7), dtype=np.float32),
        region_masks=masks,
    )
    assert np.isnan(features.loc[0, "left_cheek_attention"])
    assert np.isfinite(features.loc[0, "right_cheek_attention"])
    assert features.loc[0, "semantic_region_mask_count"] == 6

    manifest = json.loads(
        (output / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    quality = manifest["quality_control"]
    assert quality["samples_with_missing_semantic_regions"] == 1
    assert quality["missing_semantic_region_counts"]["left_cheek"] == 1


def test_landmark_failure_reports_sample_and_removes_staging(tmp_path):
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    np.save(
        aligned / "aligned_faces.npy",
        np.zeros((1, 112, 112, 3), dtype=np.uint8),
        allow_pickle=False,
    )
    pd.DataFrame(
        {
            "sample_id": ["sample-invalid"],
            "identity_id": ["identity-invalid"],
            "split": ["test"],
            "aligned_face_index": [0],
        }
    ).to_csv(aligned / "aligned_index.csv", index=False)
    (aligned / "bundle_manifest.json").write_text(
        json.dumps({"dataset_id": "survface"}),
        encoding="utf-8",
    )
    (aligned / "_SUCCESS").write_text("complete\n", encoding="utf-8")

    output = tmp_path / "landmark-regions"
    with pytest.raises(
        RuntimeError,
        match=(
            "landmark mask generation failed for aligned row 0, "
            "sample_id=sample-invalid"
        ),
    ):
        materialize_landmark_region_bundle(
            aligned,
            output_dir=output,
            dataset_id="survface",
            landmark_model=_InvalidLandmarkModel(),
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".landmark-regions.staging-*"))
