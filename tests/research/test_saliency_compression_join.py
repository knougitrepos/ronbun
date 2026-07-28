from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.evaluation.saliency_compression import (
    annotate_compression_lineage,
    join_population_saliency_with_compression,
    join_population_saliency_with_retrieval,
    saliency_compression_associations,
)


def _saliency_frame(
    *,
    model_uid: str = "model-a",
    lineage_uid: str = "origin-a",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "extraction_uid": ["extract-1", "extract-1"],
            "dataset_id": ["lfw", "lfw"],
            "sample_id": ["sample-1", "sample-2"],
            "identity_id": ["identity-1", "identity-2"],
            "model_uid": [model_uid, model_uid],
            "origin_embedding_artifact_uid": [lineage_uid, lineage_uid],
            "saliency_spec_uid": ["saliency-v1", "saliency-v1"],
            "saliency_target_eligible": [True, True],
            "heatmap_available": [True, True],
            "saliency_entropy": [0.2, 0.8],
        }
    )


def _distortion_frame(
    *,
    model_uid: str = "model-a",
    lineage_uid: str = "origin-a",
    fallback_used: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id, base_error in (("sample-1", 0.1), ("sample-2", 0.2)):
        for profile, multiplier in (("pca_128", 1.0), ("pca_64", 2.0)):
            rows.append(
                {
                    "extraction_uid": "extract-1",
                    "dataset_id": "lfw",
                    "sample_id": sample_id,
                    "model_uid": model_uid,
                    "compression_family": "pca",
                    "compression_profile": profile,
                    "origin_embedding_artifact_uid": lineage_uid,
                    "origin_fallback_used": fallback_used,
                    "angular_error_rad": base_error * multiplier,
                    "reconstruction_mse": base_error * multiplier / 10.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def _retrieval_frame(
    *,
    model_uid: str = "model-a",
    lineage_uid: str = "origin-a",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query_id, drift, is_mated in (
        ("sample-1", 0.01, True),
        ("sample-2", 0.02, False),
    ):
        for profile in ("pca_128", "pca_64"):
            for policy, multiplier in (("fixed_origin", 1.0), ("recalibrated", 0.5)):
                rows.append(
                    {
                        "extraction_uid": "extract-1",
                        "dataset_id": "lfw",
                        "query_id": query_id,
                        "model_uid": model_uid,
                        "compression_family": "pca",
                        "compression_profile": profile,
                        "origin_embedding_artifact_uid": lineage_uid,
                        "origin_fallback_used": False,
                        "threshold_policy": policy,
                        "threshold_source_split": "calibration",
                        "evaluation_split": "test",
                        "is_mated": is_mated,
                        "top1_score_drift": drift * multiplier,
                        "agreement_with_origin": True,
                        "threshold_crossing": False,
                    }
                )
    return pd.DataFrame.from_records(rows)


def test_geometry_and_retrieval_joins_keep_independent_row_grains():
    geometry = join_population_saliency_with_compression(
        _saliency_frame(),
        _distortion_frame(),
    ).sort_values(["sample_id", "compression_profile"], ignore_index=True)
    retrieval = join_population_saliency_with_retrieval(
        _saliency_frame(),
        _retrieval_frame(),
    )

    assert len(geometry) == 4
    assert len(retrieval) == 8
    assert not geometry["origin_fallback_used"].any()
    assert set(geometry["compression_profile"]) == {"pca_128", "pca_64"}
    assert geometry.groupby("sample_id").size().eq(2).all()
    assert retrieval.groupby(["sample_id", "compression_profile"]).size().eq(2).all()
    assert (
        geometry.loc[
            geometry["sample_id"].eq("sample-1"), "saliency_entropy"
        ].eq(0.2).all()
    )
    assert (
        retrieval.loc[
            retrieval["sample_id"].eq("sample-2"), "saliency_entropy"
        ].eq(0.8).all()
    )
    assert (
        retrieval.loc[
            retrieval["threshold_policy"].eq("fixed_origin"),
            "top1_score_drift",
        ].isin({0.01, 0.02}).all()
    )
    assert geometry["origin_embedding_artifact_uid"].eq("origin-a").all()
    assert retrieval["retrieval_metrics_available"].all()


def test_retrieval_may_cover_only_a_protocol_query_subset():
    retrieval = _retrieval_frame().loc[lambda frame: frame["query_id"].eq("sample-1")]

    joined = join_population_saliency_with_retrieval(
        _saliency_frame(),
        retrieval,
    )

    assert set(joined["sample_id"]) == {"sample-1"}
    assert len(joined) == 4


def test_combined_join_rejects_multi_policy_geometry_duplication():
    with pytest.raises(ValueError, match="would duplicate geometry metrics"):
        join_population_saliency_with_compression(
            _saliency_frame(),
            _distortion_frame(),
            retrieval_sensitivity=_retrieval_frame(),
        )


def test_lineage_annotation_refuses_to_overwrite_conflicting_provenance():
    frame = _distortion_frame().drop(
        columns=[
            "extraction_uid",
            "dataset_id",
            "model_uid",
            "origin_embedding_artifact_uid",
        ]
    )
    annotated = annotate_compression_lineage(
        frame,
        extraction_uid="extract-1",
        dataset_id="lfw",
        model_uid="model-a",
        origin_embedding_artifact_uid="origin-a",
    )
    assert annotated["model_uid"].eq("model-a").all()

    with pytest.raises(ValueError, match="conflicts"):
        annotate_compression_lineage(
            annotated,
            extraction_uid="extract-1",
            dataset_id="lfw",
            model_uid="model-b",
            origin_embedding_artifact_uid="origin-a",
        )


@pytest.mark.parametrize("duplicate_side", ["saliency", "distortion"])
def test_strict_join_rejects_duplicate_rows(duplicate_side: str):
    saliency = _saliency_frame()
    distortion = _distortion_frame()
    if duplicate_side == "saliency":
        saliency = pd.concat([saliency, saliency.iloc[[0]]], ignore_index=True)
        expected = "saliency_features rows are not unique"
    else:
        distortion = pd.concat(
            [distortion, distortion.iloc[[0]]],
            ignore_index=True,
        )
        expected = "embedding_distortion rows are not unique"

    with pytest.raises(ValueError, match=expected):
        join_population_saliency_with_compression(saliency, distortion)


def test_strict_join_rejects_origin_embedding_lineage_mismatch():
    with pytest.raises(
        ValueError,
        match="origin_embedding_artifact_uid differs",
    ):
        join_population_saliency_with_compression(
            _saliency_frame(lineage_uid="origin-saliency"),
            _distortion_frame(lineage_uid="origin-compression"),
        )


def test_strict_join_rejects_fallback_artifacts():
    with pytest.raises(ValueError, match="fallback-free artifacts"):
        join_population_saliency_with_compression(
            _saliency_frame(),
            _distortion_frame(fallback_used=True),
        )


def test_model_uid_prevents_cross_model_sample_mixing():
    saliency = pd.concat(
        [
            _saliency_frame(model_uid="model-a").assign(saliency_entropy=[0.1, 0.2]),
            _saliency_frame(model_uid="model-b").assign(saliency_entropy=[0.7, 0.8]),
        ],
        ignore_index=True,
    )
    distortion = pd.concat(
        [
            _distortion_frame(model_uid="model-a"),
            _distortion_frame(model_uid="model-b"),
        ],
        ignore_index=True,
    )

    joined = join_population_saliency_with_compression(saliency, distortion)
    by_model = joined.groupby("model_uid")["saliency_entropy"]
    assert set(by_model.get_group("model-a")) == {0.1, 0.2}
    assert set(by_model.get_group("model-b")) == {0.7, 0.8}

    with pytest.raises(ValueError, match="compression rows have no saliency"):
        join_population_saliency_with_compression(
            _saliency_frame(model_uid="model-a"),
            _distortion_frame(model_uid="model-b"),
        )


def _association_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for profile, direction in (("pca_128", 1.0), ("pca_64", -1.0)):
        for index in range(8):
            feature = float(index + 1)
            rows.append(
                {
                    "dataset_id": "lfw",
                    "model_uid": "model-a",
                    "compression_family": "pca",
                    "compression_profile": profile,
                    "identity_id": f"identity-{index // 2}",
                    "saliency_entropy": feature,
                    "angular_error_rad": (feature if direction > 0 else 9.0 - feature),
                }
            )
    return pd.DataFrame.from_records(rows)


def test_profile_specific_identity_bootstrap_is_reproducible():
    joined = _association_frame()
    kwargs = {
        "saliency_features": ("saliency_entropy",),
        "sensitivity_metrics": ("angular_error_rad",),
        "bootstrap_repeats": 80,
        "confidence_level": 0.90,
        "seed": 1701,
    }

    first = saliency_compression_associations(joined, **kwargs)
    repeated = saliency_compression_associations(joined, **kwargs)
    shuffled = saliency_compression_associations(
        joined.sample(frac=1.0, random_state=23),
        **kwargs,
    )

    pd.testing.assert_frame_equal(first, repeated)
    pd.testing.assert_frame_equal(first, shuffled)
    assert len(first) == 2
    by_profile = first.set_index("compression_profile")
    assert by_profile.loc["pca_128", "spearman_rho"] == pytest.approx(1.0)
    assert by_profile.loc["pca_64", "spearman_rho"] == pytest.approx(-1.0)
    assert (first["sample_count"] == 8).all()
    assert (first["identity_count"] == 4).all()
    assert (first["bootstrap_valid_repeats"] == 80).all()
    assert np.allclose(first["bootstrap_ci_low"], [1.0, -1.0])
    assert np.allclose(first["bootstrap_ci_high"], [1.0, -1.0])

# wrapper tests follow
