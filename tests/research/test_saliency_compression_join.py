from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.evaluation.saliency_compression import (
    WEIGHTED_RERANK_ALGORITHM_VERSION,
    WEIGHTED_RERANK_STRATEGY,
    _bootstrap_seed,
    _rank_spec,
    _spearman,
    _weighted_average_rank_batch,
    annotate_compression_lineage,
    derive_saliency_threshold_metrics,
    join_population_saliency_with_compression,
    join_population_saliency_with_retrieval,
    saliency_compression_associations,
    saliency_retrieval_associations,
    threshold_instability_associations,
    threshold_policy_event_comparisons,
    threshold_policy_saliency_rho_comparisons,
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
            for policy, multiplier in (
                ("frozen_origin", 1.0),
                ("recalibrated_compressed", 0.5),
            ):
                compressed_score = 0.70 + drift * multiplier
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
                        "top_k": 20,
                        "origin_top1_score": 0.70,
                        "compressed_top1_score": compressed_score,
                        "compressed_score_at_origin_top1": compressed_score,
                        "top1_score_drift": drift * multiplier,
                        "origin_winner_score_drift": drift * multiplier,
                        "origin_decision_threshold": 0.60,
                        "compressed_decision_threshold": 0.60,
                        "origin_accepted": True,
                        "compressed_accepted": True,
                        "origin_true_identity_rank": 1 if is_mated else np.nan,
                        "compressed_true_identity_rank": (
                            1 if is_mated else np.nan
                        ),
                        "origin_true_identity_score": (
                            0.70 if is_mated else np.nan
                        ),
                        "compressed_true_identity_score": (
                            compressed_score if is_mated else np.nan
                        ),
                        "origin_tpir_at_rank_k": is_mated,
                        "compressed_tpir_at_rank_k": is_mated,
                        "score_spaces_comparable": True,
                        "agreement_with_origin": True,
                        "threshold_crossing": False,
                        "threshold_crossing_direction": "none",
                    }
                )
    return pd.DataFrame.from_records(rows)


def test_threshold_metrics_separate_maximum_and_fixed_pair_score_shift():
    source = pd.DataFrame(
        {
            "is_mated": [True, False],
            "top_k": [20, 20],
            "origin_top1_score": [0.70, 0.70],
            "compressed_top1_score": [0.65, -0.40],
            "compressed_score_at_origin_top1": [0.62, np.nan],
            "top1_score_drift": [-0.05, np.nan],
            "origin_winner_score_drift": [-0.08, np.nan],
            "origin_decision_threshold": [0.60, 0.60],
            "compressed_decision_threshold": [0.55, -0.30],
            "origin_accepted": [True, True],
            "compressed_accepted": [True, False],
            "origin_true_identity_rank": [1, np.nan],
            "compressed_true_identity_rank": [1, np.nan],
            "origin_true_identity_score": [0.70, np.nan],
            "compressed_true_identity_score": [0.65, np.nan],
            "origin_tpir_at_rank_k": [True, False],
            "compressed_tpir_at_rank_k": [True, False],
            "score_spaces_comparable": [True, False],
            "threshold_crossing": [False, True],
            "threshold_crossing_direction": ["none", "accept_to_reject"],
        }
    )

    derived = derive_saliency_threshold_metrics(source)

    assert derived.loc[0, "absolute_top1_score_drift"] == pytest.approx(0.05)
    assert derived.loc[0, "absolute_origin_winner_score_drift"] == pytest.approx(
        0.08
    )
    assert derived.loc[0, "origin_threshold_margin"] == pytest.approx(0.10)
    assert derived.loc[0, "compressed_threshold_margin"] == pytest.approx(0.10)
    assert derived.loc[0, "threshold_margin_shift"] == pytest.approx(0.0)
    assert pd.isna(derived.loc[1, "absolute_top1_score_drift"])
    assert pd.isna(derived.loc[1, "absolute_origin_winner_score_drift"])
    assert pd.isna(derived.loc[1, "threshold_margin_shift"])
    assert derived.loc[1, "origin_threshold_distance"] == pytest.approx(0.10)
    assert derived.loc[1, "compressed_threshold_distance"] == pytest.approx(0.10)
    assert bool(derived.loc[1, "accept_to_reject_crossing"])
    assert not bool(derived.loc[1, "reject_to_accept_crossing"])
    assert derived.loc[1, "false_accept_loss"] == pytest.approx(1.0)
    assert pd.isna(derived.loc[0, "false_accept_loss"])
    assert derived["threshold_metric_derivation_version"].nunique() == 1


def test_threshold_metrics_reject_crossing_direction_mismatch():
    source = _retrieval_frame().iloc[[0]].assign(
        threshold_crossing=True,
        threshold_crossing_direction="none",
    )

    with pytest.raises(ValueError, match="disagrees"):
        derive_saliency_threshold_metrics(source)


def test_threshold_metrics_reject_score_crossing_mismatch():
    source = _retrieval_frame().iloc[[0]].assign(
        compressed_top1_score=0.50,
        compressed_score_at_origin_top1=0.50,
        top1_score_drift=-0.20,
        origin_winner_score_drift=-0.20,
        compressed_accepted=True,
        threshold_crossing=False,
        threshold_crossing_direction="none",
        compressed_true_identity_score=0.50,
        compressed_tpir_at_rank_k=False,
    )

    with pytest.raises(ValueError, match="compressed_accepted disagrees"):
        derive_saliency_threshold_metrics(source)


def test_threshold_metrics_split_tpir_threshold_and_rank_losses():
    source = pd.concat(
        [
            _retrieval_frame().iloc[[0]],
            _retrieval_frame().iloc[[0]],
        ],
        ignore_index=True,
    )
    source.loc[0, [
        "compressed_top1_score",
        "compressed_score_at_origin_top1",
        "compressed_true_identity_score",
    ]] = 0.50
    source.loc[0, ["top1_score_drift", "origin_winner_score_drift"]] = -0.20
    source.loc[0, "compressed_accepted"] = False
    source.loc[0, "compressed_tpir_at_rank_k"] = False
    source.loc[0, "threshold_crossing"] = True
    source.loc[0, "threshold_crossing_direction"] = "accept_to_reject"
    source.loc[1, [
        "compressed_true_identity_rank",
        "compressed_true_identity_score",
    ]] = np.nan
    source.loc[1, "compressed_tpir_at_rank_k"] = False

    derived = derive_saliency_threshold_metrics(source)

    assert derived["tpir_at_rank_k_loss"].tolist() == [1.0, 1.0]
    assert derived["tpir_threshold_loss"].tolist() == [1.0, 0.0]
    assert derived["tpir_rank_loss"].tolist() == [0.0, 1.0]


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
            retrieval["threshold_policy"].eq("frozen_origin"),
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


def test_retrieval_join_and_associations_keep_search_modes_separate():
    reconstruction = _retrieval_frame().assign(
        search_mode="pca_reconstruction_cosine"
    )
    direct = _retrieval_frame().assign(search_mode="pca_direct_cosine")
    joined = join_population_saliency_with_retrieval(
        _saliency_frame(),
        pd.concat([direct, reconstruction], ignore_index=True),
    )

    assert len(joined) == 16
    assert set(joined["search_mode"]) == {
        "pca_direct_cosine",
        "pca_reconstruction_cosine",
    }
    associations = saliency_retrieval_associations(
        joined,
        saliency_features=("saliency_entropy",),
        sensitivity_metrics=("top1_score_drift",),
        bootstrap_repeats=0,
    )
    assert set(associations["search_mode"]) == {
        "pca_direct_cosine",
        "pca_reconstruction_cosine",
    }
    assert associations["threshold_metric_derivation_version"].nunique() == 1


def test_retrieval_join_and_associations_keep_target_fpirs_separate():
    target_010 = _retrieval_frame().assign(target_fpir=0.10)
    target_001 = _retrieval_frame().assign(target_fpir=0.01)
    joined = join_population_saliency_with_retrieval(
        _saliency_frame(),
        pd.concat([target_010, target_001], ignore_index=True),
    )

    assert len(joined) == 16
    assert set(joined["target_fpir"]) == {0.10, 0.01}
    associations = saliency_retrieval_associations(
        joined,
        saliency_features=("saliency_entropy",),
        sensitivity_metrics=("top1_score_drift",),
        bootstrap_repeats=0,
    )
    assert set(associations["target_fpir"]) == {0.10, 0.01}


def test_threshold_instability_association_reports_event_denominators():
    joined = join_population_saliency_with_retrieval(
        _saliency_frame(),
        _retrieval_frame(),
    )

    associations = threshold_instability_associations(
        joined,
        predictors=("absolute_top1_score_drift",),
        event_metrics=("threshold_crossing",),
        bootstrap_repeats=0,
    )

    assert set(associations["instability_predictor"]) == {
        "absolute_top1_score_drift"
    }
    assert associations["event_count"].eq(0).all()
    assert associations["non_event_count"].eq(1).all()
    assert associations["event_rate"].eq(0.0).all()


def test_threshold_policy_comparison_is_paired_by_query_and_identity():
    joined = join_population_saliency_with_retrieval(
        _saliency_frame(),
        _retrieval_frame(),
    )
    recalibrated_sample_1 = joined["threshold_policy"].eq(
        "recalibrated_compressed"
    ) & joined["sample_id"].eq("sample-1")
    joined.loc[recalibrated_sample_1, "threshold_crossing"] = True

    comparisons = threshold_policy_event_comparisons(
        joined,
        event_metrics=("threshold_crossing",),
        bootstrap_repeats=20,
        seed=1701,
    )

    mated = comparisons.loc[comparisons["is_mated"]].reset_index(drop=True)
    assert len(mated) == 2
    assert mated["paired_query_count"].eq(1).all()
    assert mated["frozen_event_count"].eq(0).all()
    assert mated["recalibrated_event_count"].eq(1).all()
    assert mated["recalibrated_minus_frozen_rate"].eq(1.0).all()
    assert mated["introduced_event_count"].eq(1).all()
    assert mated["paired_bootstrap_valid_repeats"].eq(20).all()


def test_threshold_policy_saliency_rho_comparison_uses_paired_queries():
    rows = []
    frozen_events = (0.0, 0.0, 1.0, 1.0)
    recalibrated_events = (0.0, 1.0, 1.0, 1.0)
    for policy, events in (
        ("frozen_origin", frozen_events),
        ("recalibrated_compressed", recalibrated_events),
    ):
        for index, event in enumerate(events, start=1):
            rows.append(
                {
                    "extraction_uid": "extract-1",
                    "dataset_id": "lfw",
                    "sample_id": f"sample-{index}",
                    "model_uid": "model-a",
                    "compression_family": "pca",
                    "compression_profile": "pca_128",
                    "identity_id": f"identity-{index}",
                    "threshold_policy": policy,
                    "is_mated": True,
                    "target_fpir": 0.01,
                    "saliency_entropy": index / 10.0,
                    "tpir_at_rank_k_loss": event,
                }
            )

    comparisons = threshold_policy_saliency_rho_comparisons(
        pd.DataFrame.from_records(rows),
        saliency_features=("saliency_entropy",),
        event_metrics=("tpir_at_rank_k_loss",),
        bootstrap_repeats=20,
        bootstrap_batch_size=4,
        minimum_event_count=1,
        seed=1701,
    )

    assert len(comparisons) == 1
    row = comparisons.iloc[0]
    assert row["paired_query_count"] == 4
    assert row["frozen_event_count"] == 2
    assert row["recalibrated_event_count"] == 3
    assert row["recalibrated_minus_frozen_rho"] == pytest.approx(
        row["recalibrated_spearman_rho"] - row["frozen_spearman_rho"]
    )
    assert 0 < row["paired_bootstrap_valid_repeats"] <= 20


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


def test_weighted_rerank_identity_bootstrap_is_reproducible_without_concat(
    monkeypatch: pytest.MonkeyPatch,
):
    joined = _association_frame()
    common = {
        "saliency_features": ("saliency_entropy",),
        "sensitivity_metrics": ("angular_error_rad",),
        "bootstrap_repeats": 80,
        "confidence_level": 0.90,
        "seed": 1701,
    }
    legacy = saliency_compression_associations(joined, **common)

    def reject_concat(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "weighted rerank bootstrap must not materialize concat samples"
        )

    monkeypatch.setattr(pd, "concat", reject_concat)
    weighted = saliency_compression_associations(
        joined,
        **common,
        bootstrap_rank_strategy=WEIGHTED_RERANK_STRATEGY,
        bootstrap_batch_size=7,
    )
    shuffled = saliency_compression_associations(
        joined.sample(frac=1.0, random_state=23),
        **common,
        bootstrap_rank_strategy=WEIGHTED_RERANK_STRATEGY,
        bootstrap_batch_size=7,
    )

    pd.testing.assert_frame_equal(weighted, shuffled)
    assert np.allclose(weighted["spearman_rho"], legacy["spearman_rho"])
    assert weighted["bootstrap_rank_strategy"].eq(
        WEIGHTED_RERANK_STRATEGY
    ).all()
    assert weighted["association_algorithm_version"].eq(
        WEIGHTED_RERANK_ALGORITHM_VERSION
    ).all()
    assert (weighted["bootstrap_valid_repeats"] == 80).all()
    assert np.allclose(weighted["bootstrap_ci_low"], [1.0, -1.0])
    assert np.allclose(weighted["bootstrap_ci_high"], [1.0, -1.0])


def test_weighted_average_ranks_match_explicit_expanded_rows():
    values = np.array([1.0, 1.0, 2.0, 4.0, 4.0])
    row_weights = np.array(
        [
            [2, 0, 3, 1, 2],
            [0, 2, 1, 3, 0],
        ],
        dtype=np.int64,
    )

    actual = _weighted_average_rank_batch(
        row_weights,
        _rank_spec(values),
    )

    for batch_index, weights in enumerate(row_weights):
        expanded_values = np.repeat(values, weights)
        expected = pd.Series(expanded_values).rank(method="average").to_numpy()
        expanded_actual = np.repeat(actual[batch_index], weights)
        np.testing.assert_allclose(expanded_actual, expected)


def test_weighted_rerank_ci_matches_explicit_cluster_expansion():
    joined = (
        _association_frame()
        .loc[lambda frame: frame["compression_profile"].eq("pca_128")]
        .copy()
    )
    joined["saliency_entropy"] = [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0]
    joined["angular_error_rad"] = [4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0, 2.0]
    repeats = 40
    batch_size = 7
    seed = 1701

    weighted = saliency_compression_associations(
        joined,
        saliency_features=("saliency_entropy",),
        sensitivity_metrics=("angular_error_rad",),
        bootstrap_repeats=repeats,
        seed=seed,
        bootstrap_rank_strategy=WEIGHTED_RERANK_STRATEGY,
        bootstrap_batch_size=batch_size,
    )

    identities = joined["identity_id"].astype(str).to_numpy()
    cluster_identities, cluster_codes = np.unique(
        identities,
        return_inverse=True,
    )
    cluster_count = len(cluster_identities)
    rng = np.random.default_rng(
        _bootstrap_seed(
            seed,
            ("lfw", "model-a", "pca", "pca_128"),
            cluster_identities,
        )
    )
    probabilities = np.full(cluster_count, 1.0 / cluster_count)
    reference: list[float] = []
    for start in range(0, repeats, batch_size):
        counts = rng.multinomial(
            cluster_count,
            probabilities,
            size=min(batch_size, repeats - start),
        )
        for cluster_weights in counts:
            row_weights = cluster_weights[cluster_codes]
            expanded = np.repeat(np.arange(len(joined)), row_weights)
            value = _spearman(
                pd.Series(
                    joined["saliency_entropy"].to_numpy()[expanded]
                ),
                pd.Series(
                    joined["angular_error_rad"].to_numpy()[expanded]
                ),
            )
            if np.isfinite(value):
                reference.append(float(value))
    expected_low, expected_high = np.quantile(reference, [0.025, 0.975])

    assert weighted.loc[0, "bootstrap_valid_repeats"] == len(reference)
    assert weighted.loc[0, "bootstrap_ci_low"] == pytest.approx(expected_low)
    assert weighted.loc[0, "bootstrap_ci_high"] == pytest.approx(expected_high)


def test_weighted_rerank_bootstrap_is_stable_when_other_pairs_are_added():
    joined = _association_frame().assign(
        saliency_spread=lambda frame: frame["saliency_entropy"] ** 2
    )
    common = {
        "sensitivity_metrics": ("angular_error_rad",),
        "bootstrap_repeats": 80,
        "seed": 1701,
        "bootstrap_rank_strategy": WEIGHTED_RERANK_STRATEGY,
    }
    subset = saliency_compression_associations(
        joined,
        saliency_features=("saliency_entropy",),
        **common,
    )
    expanded = saliency_compression_associations(
        joined,
        saliency_features=("saliency_entropy", "saliency_spread"),
        **common,
    )
    expanded_subset = expanded.loc[
        expanded["saliency_feature"].eq("saliency_entropy")
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(subset, expanded_subset)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        (
            "bootstrap_rank_strategy",
            "unknown",
            "bootstrap_rank_strategy",
        ),
        ("bootstrap_batch_size", 0, "bootstrap_batch_size"),
    ],
)
def test_association_rejects_invalid_scalable_bootstrap_options(
    keyword: str,
    value: object,
    message: str,
):
    kwargs = {
        "saliency_features": ("saliency_entropy",),
        "sensitivity_metrics": ("angular_error_rad",),
        keyword: value,
    }
    with pytest.raises(ValueError, match=message):
        saliency_compression_associations(_association_frame(), **kwargs)

# wrapper tests follow
