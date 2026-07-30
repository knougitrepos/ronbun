import numpy as np
import pytest

from research.explainability.gradcam.features import (
    SEMANTIC_ATTENTION_COLUMNS,
    summarize_saliency_features,
)
from research.explainability.gradcam.metrics import occlude_by_saliency
from research.explainability.gradcam.regions import build_landmark_region_masks
from research.explainability.gradcam.templates import (
    ELIGIBLE_REASON,
    MISSING_IDENTITY_REASON,
    SINGLETON_IDENTITY_REASON,
    ZERO_RESIDUAL_REASON,
    build_leave_one_out_identity_templates,
)


def test_leave_one_out_templates_exclude_self_and_never_cross_scope():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    bundle = build_leave_one_out_identity_templates(
        sample_ids=["a-train-1", "a-train-2", "a-test-1"],
        identity_ids=["identity-a", "identity-a", "identity-a"],
        normalized_embeddings=embeddings,
        model_uid="arcface-checkpoint",
        scope_ids=["train", "train", "test"],
    )

    np.testing.assert_allclose(bundle.templates[0], embeddings[1])
    np.testing.assert_allclose(bundle.templates[1], embeddings[0])
    assert bundle.template_member_counts.tolist() == [1, 1, 0]
    assert bundle.eligible.tolist() == [True, True, False]
    assert bundle.exclusion_reasons.tolist() == [
        ELIGIBLE_REASON,
        ELIGIBLE_REASON,
        SINGLETON_IDENTITY_REASON,
    ]
    assert np.isnan(bundle.templates[2]).all()


def test_leave_one_out_templates_retain_singleton_missing_and_zero_residual_rows():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    bundle = build_leave_one_out_identity_templates(
        sample_ids=["c-1", "c-2", "c-3", "singleton", "missing"],
        identity_ids=["identity-c", "identity-c", "identity-c", "identity-d", None],
        normalized_embeddings=embeddings,
        model_uid="arcface-checkpoint",
    )

    assert bundle.exclusion_reasons.tolist() == [
        ZERO_RESIDUAL_REASON,
        ELIGIBLE_REASON,
        ELIGIBLE_REASON,
        SINGLETON_IDENTITY_REASON,
        MISSING_IDENTITY_REASON,
    ]
    assert bundle.template_member_counts.tolist() == [2, 2, 2, 0, 0]
    assert bundle.eligible.tolist() == [False, True, True, False, False]
    assert np.isnan(bundle.templates[0]).all()
    assert np.isnan(bundle.target_scores[[0, 3, 4]]).all()


def test_spatial_features_capture_hotspot_and_keep_zero_map_undefined():
    heatmaps = np.zeros((2, 4, 4), dtype=np.float32)
    heatmaps[0, 0, 3] = 2.0
    left_eye_mask = np.zeros((4, 4), dtype=bool)
    left_eye_mask[0, 3] = True

    features = summarize_saliency_features(
        heatmaps,
        region_masks={"left_eye": left_eye_mask},
    )

    hotspot = features.iloc[0]
    assert hotspot["quadrant_top_left"] == pytest.approx(0.0)
    assert hotspot["quadrant_top_right"] == pytest.approx(1.0)
    assert hotspot["quadrant_bottom_left"] == pytest.approx(0.0)
    assert hotspot["quadrant_bottom_right"] == pytest.approx(0.0)
    assert hotspot["saliency_center_x"] == pytest.approx(1.0)
    assert hotspot["saliency_center_y"] == pytest.approx(0.0)
    assert hotspot["saliency_spread"] == pytest.approx(0.0)
    assert hotspot["left_right_asymmetry"] == pytest.approx(1.0)
    assert hotspot["saliency_entropy"] == pytest.approx(0.0)
    assert bool(hotspot["saliency_valid"])
    assert hotspot["left_eye_attention"] == pytest.approx(1.0)
    assert hotspot["semantic_region_mask_count"] == 1
    assert np.isnan(hotspot["right_eye_attention"])
    assert np.isnan(hotspot["outside_face_attention"])
    assert np.isnan(hotspot["face_attention"])
    assert not bool(hotspot["face_mask_available"])

    zero_map = features.iloc[1]
    assert zero_map["saliency_mass"] == pytest.approx(0.0)
    assert not bool(zero_map["saliency_valid"])
    for column in (
        "quadrant_top_left",
        "quadrant_top_right",
        "quadrant_bottom_left",
        "quadrant_bottom_right",
        "saliency_center_x",
        "saliency_center_y",
        "saliency_spread",
        "left_right_asymmetry",
        "saliency_entropy",
        "left_eye_attention",
        "maximum_region_concentration",
    ):
        assert np.isnan(zero_map[column])


def test_missing_semantic_masks_are_not_guessed_from_sparse_landmarks():
    landmarks = np.array(
        [
            [
                [1.0, 1.0],
                [3.0, 1.0],
                [2.0, 2.0],
            ]
        ],
        dtype=np.float64,
    )

    masks = build_landmark_region_masks(
        landmarks,
        region_point_indices={
            "left_eye": [0],
            "right_eye": [1],
        },
        region_radii_pixels={
            "left_eye": 0.25,
            "right_eye": 0.25,
        },
        image_size=(4, 5),
    )

    assert set(masks) == {"left_eye", "right_eye"}
    assert masks["left_eye"].shape == (1, 4, 5)
    assert masks["left_eye"][0, 1, 1]
    assert masks["left_eye"].sum() == 1
    assert masks["right_eye"][0, 1, 3]
    assert masks["right_eye"].sum() == 1

    features = summarize_saliency_features(
        np.ones((1, 4, 5), dtype=np.float32),
        region_masks=masks,
    )
    for column in SEMANTIC_ATTENTION_COLUMNS:
        if column not in {"left_eye_attention", "right_eye_attention"}:
            assert np.isnan(features.loc[0, column])
    assert features.loc[0, "semantic_region_mask_count"] == 2
    assert np.isnan(features.loc[0, "outside_face_attention"])
    assert np.isnan(features.loc[0, "face_attention"])


def test_random_occlusion_is_deterministic_per_sample_id_across_batch_order():
    images = np.arange(3 * 4 * 4 * 2, dtype=np.float32).reshape(3, 4, 4, 2)
    heatmaps = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)
    sample_ids = np.array(["sample-a", "sample-b", "sample-c"])

    canonical = occlude_by_saliency(
        images,
        heatmaps,
        fraction=0.25,
        strategy="random",
        fill_value=(-1.0, -2.0),
        seed=37,
        sample_ids=sample_ids,
    )
    repeated = occlude_by_saliency(
        images,
        heatmaps,
        fraction=0.25,
        strategy="random",
        fill_value=(-1.0, -2.0),
        seed=37,
        sample_ids=sample_ids,
    )

    permutation = np.array([2, 0, 1])
    shuffled = occlude_by_saliency(
        images[permutation],
        heatmaps[permutation],
        fraction=0.25,
        strategy="random",
        fill_value=(-1.0, -2.0),
        seed=37,
        sample_ids=sample_ids[permutation],
    )
    inverse_permutation = np.argsort(permutation)

    np.testing.assert_array_equal(canonical, repeated)
    np.testing.assert_array_equal(canonical, shuffled[inverse_permutation])
    assert np.count_nonzero(np.all(canonical == (-1.0, -2.0), axis=-1)) == 12
