from __future__ import annotations

import pandas as pd

from research.evaluation.saliency_compression import (
    saliency_geometry_associations,
    saliency_retrieval_associations,
)


def test_geometry_wrapper_marks_scope_and_keeps_profile_grain():
    rows: list[dict[str, object]] = []
    for profile in ("pca_128", "pca_64"):
        for index in range(6):
            rows.append(
                {
                    "extraction_uid": "extract-1",
                    "dataset_id": "lfw",
                    "sample_id": f"{profile}-sample-{index}",
                    "model_uid": "model-a",
                    "compression_family": "pca",
                    "compression_profile": profile,
                    "identity_id": f"identity-{index // 2}",
                    "saliency_entropy": float(index),
                    "angular_error_rad": float(index) / 10.0,
                }
            )
    associations = saliency_geometry_associations(
        pd.DataFrame.from_records(rows),
        saliency_features=("saliency_entropy",),
        sensitivity_metrics=("angular_error_rad",),
        bootstrap_repeats=0,
    )

    assert set(associations["analysis_scope"]) == {"geometry"}
    assert len(associations) == 2


def test_retrieval_wrapper_separates_policy_and_mated_status():
    rows: list[dict[str, object]] = []
    for policy in ("fixed_origin", "recalibrated"):
        for is_mated in (False, True):
            for index in range(6):
                rows.append(
                    {
                        "dataset_id": "lfw",
                        "model_uid": "model-a",
                        "compression_family": "pca",
                        "compression_profile": "pca_128",
                        "threshold_source_split": "calibration",
                        "evaluation_split": "test",
                        "threshold_policy": policy,
                        "is_mated": is_mated,
                        "identity_id": f"identity-{index // 2}",
                        "saliency_entropy": float(index),
                        "top1_score_drift": float(index) / 10.0,
                    }
                )
    associations = saliency_retrieval_associations(
        pd.DataFrame.from_records(rows),
        saliency_features=("saliency_entropy",),
        sensitivity_metrics=("top1_score_drift",),
        bootstrap_repeats=0,
    )

    assert set(associations["analysis_scope"]) == {"retrieval"}
    assert len(associations) == 4
    assert set(associations["threshold_policy"]) == {
        "fixed_origin",
        "recalibrated",
    }
    assert set(associations["is_mated"]) == {False, True}
