import numpy as np
import pandas as pd

from research.calibration import (
    apply_threshold_model,
    assign_quality_groups,
    fit_conditional_threshold,
    fit_global_threshold,
    paired_method_comparison,
)


def _rows(prefix: str, count: int = 1200) -> pd.DataFrame:
    index = np.arange(count)
    is_mated = index % 2 == 0
    quality = (index % 100) / 99.0
    non_mated_score = 0.25 + 0.35 * (1.0 - quality)
    mated_score = 0.72 + 0.20 * quality
    score = np.where(is_mated, mated_score, non_mated_score)
    return pd.DataFrame(
        {
            "sample_id": [f"{prefix}-{value}" for value in index],
            "identity_id": [f"{prefix}-identity-{value // 2}" for value in index],
            "is_mated": is_mated,
            "score": score,
            "fiqa_score": quality,
            "top_k_correct": is_mated,
        }
    )


def test_quality_group_boundaries_are_frozen_and_higher_quality_maps_high():
    groups = assign_quality_groups(
        np.asarray([0.1, 0.5, 0.9]),
        cutpoints=(0.4, 0.7),
        labels=("low", "mid", "high"),
    )
    assert groups.tolist() == ["low", "mid", "high"]


def test_global_and_quality_models_use_calibration_only_and_evaluate_same_test():
    calibration = _rows("cal")
    test = _rows("test")
    global_model = fit_global_threshold(
        calibration,
        target_fpir=0.10,
        safety_fraction=0.25,
        score_space="negative_squared_l2_adc",
    )
    quality_model = fit_conditional_threshold(
        calibration,
        target_fpir=0.10,
        bin_count=2,
        shrinkage_strength=50.0,
        minimum_group_non_mated=50,
        safety_fraction=0.25,
        score_space="negative_squared_l2_adc",
    )
    global_result = apply_threshold_model(test, global_model)
    quality_result = apply_threshold_model(test, quality_model)
    paired = paired_method_comparison(global_result, quality_result)

    assert global_model.quality_column is None
    assert quality_model.group_labels == ("low", "high")
    assert quality_model.quality_cutpoints == tuple(
        quality_model.quality_cutpoints
    )
    assert quality_model.calibration_partition["partition_unit"] == (
        "identity_cluster"
    )
    assert quality_model.calibration_partition["method"] == (
        "sha256_partition_key"
    )
    assert all(
        group.raw_threshold is None
        or group.threshold_before_safety >= group.raw_threshold
        for group in quality_model.groups
    )
    assert all(
        group.fit_target_met is not False for group in quality_model.groups
    )
    assert global_result.summary["threshold_fit_on_test"] is False
    assert quality_result.summary["threshold_fit_on_test"] is False
    assert paired["score_space"] == "negative_squared_l2_adc"
    assert paired["tpir_at_rank_k"]["total"] == int(test["is_mated"].sum())


def test_sparse_quality_group_falls_back_to_global_threshold():
    calibration = _rows("cal", count=400)
    model = fit_conditional_threshold(
        calibration,
        target_fpir=0.10,
        bin_count=3,
        minimum_group_non_mated=10_000,
        safety_fraction=0.20,
        score_space="cosine_similarity",
    )

    assert all(group.used_global_fallback for group in model.groups)
    assert all(
        group.final_threshold
        == max(
            model.global_final_threshold,
            group.safety_threshold
            if group.safety_threshold is not None
            else model.global_final_threshold,
        )
        for group in model.groups
    )
    assert all(
        group.safety_threshold is None
        or group.final_threshold >= group.safety_threshold
        for group in model.groups
    )
    assert all(
        group.safety_threshold is None or group.safety_target_met is True
        for group in model.groups
    )


def test_sparse_fallback_keeps_available_group_safety_threshold():
    calibration = _rows("cal", count=1200)
    model = fit_conditional_threshold(
        calibration,
        target_fpir=0.10,
        bin_count=2,
        minimum_group_non_mated=100,
        safety_fraction=0.80,
        score_space="cosine_similarity",
    )

    guarded_fallbacks = [
        group
        for group in model.groups
        if group.used_global_fallback and group.safety_threshold is not None
    ]
    assert guarded_fallbacks
    assert all(
        group.final_threshold >= group.safety_threshold
        for group in guarded_fallbacks
    )
    assert all(group.safety_target_met is True for group in guarded_fallbacks)
