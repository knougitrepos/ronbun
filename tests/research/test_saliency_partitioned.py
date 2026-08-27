from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.evaluation.saliency_compression import (
    DEFAULT_RETRIEVAL_METRICS,
    DEFAULT_SALIENCY_FEATURES,
    WEIGHTED_RERANK_STRATEGY,
    join_population_saliency_with_retrieval,
    saliency_retrieval_associations,
    threshold_instability_associations,
    threshold_policy_event_comparisons,
    threshold_policy_saliency_rho_comparisons,
)
from research.evaluation.saliency_partitioned import (
    compute_partitioned_retrieval_associations,
    iter_partitioned_projection_batches,
    load_partitioned_projection_manifest,
    stream_partition_saliency_retrieval_projection,
)


def _saliency() -> pd.DataFrame:
    rows = []
    for index in range(8):
        row: dict[str, object] = {
            "extraction_uid": "extract-a",
            "dataset_id": "survface",
            "sample_id": f"q{index}",
            "model_uid": "model-a",
            "identity_id": f"identity-{index}",
            "origin_embedding_artifact_uid": "origin-a",
            "saliency_spec_uid": "saliency-a",
            "saliency_target_eligible": True,
            "heatmap_available": True,
        }
        for feature_index, feature in enumerate(DEFAULT_SALIENCY_FEATURES):
            row[feature] = (index + 1) * (feature_index + 1) / 100.0
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _retrieval() -> pd.DataFrame:
    rows = []
    for policy in ("frozen_origin", "recalibrated_compressed"):
        compressed_threshold = 0.58 if policy == "frozen_origin" else 0.52
        for index in range(8):
            is_mated = index < 4
            origin_score = 0.75 - index * 0.055
            compressed_score = origin_score + (-0.04 + index * 0.012)
            origin_accepted = origin_score >= 0.58
            compressed_accepted = compressed_score >= compressed_threshold
            rows.append(
                {
                    "extraction_uid": "extract-a",
                    "dataset_id": "survface",
                    "query_id": f"q{index}",
                    "model_uid": "model-a",
                    "compression_family": "pca",
                    "compression_profile": "pca_32",
                    "origin_embedding_artifact_uid": "origin-a",
                    "origin_fallback_used": False,
                    "search_mode": "pca_reconstruction_cosine",
                    "protocol_uid": "survface-official-v1",
                    "threshold_source_split": "calibration",
                    "evaluation_split": "official_test",
                    "target_fpir": 0.1,
                    "threshold_policy": policy,
                    "is_mated": is_mated,
                    "top_k": 20,
                    "origin_top1_score": origin_score,
                    "compressed_top1_score": compressed_score,
                    "compressed_score_at_origin_top1": compressed_score,
                    "top1_score_drift": compressed_score - origin_score,
                    "origin_winner_score_drift": compressed_score - origin_score,
                    "origin_decision_threshold": 0.58,
                    "compressed_decision_threshold": compressed_threshold,
                    "origin_accepted": origin_accepted,
                    "compressed_accepted": compressed_accepted,
                    "origin_true_identity_rank": 1.0 if is_mated else np.nan,
                    "compressed_true_identity_rank": 1.0 if is_mated else np.nan,
                    "origin_true_identity_score": (
                        origin_score if is_mated else np.nan
                    ),
                    "compressed_true_identity_score": (
                        compressed_score if is_mated else np.nan
                    ),
                    "origin_tpir_at_rank_k": is_mated and origin_accepted,
                    "compressed_tpir_at_rank_k": (
                        is_mated and compressed_accepted
                    ),
                    "score_spaces_comparable": True,
                    "agreement_with_origin": index not in {2, 6},
                    "threshold_crossing": origin_accepted != compressed_accepted,
                    "threshold_crossing_direction": (
                        "accept_to_reject"
                        if origin_accepted and not compressed_accepted
                        else "reject_to_accept"
                        if compressed_accepted and not origin_accepted
                        else "none"
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        column
        for column in (
            "dataset_id",
            "model_uid",
            "compression_family",
            "compression_profile",
            "search_mode",
            "protocol_uid",
            "threshold_source_split",
            "evaluation_split",
            "target_fpir",
            "threshold_policy",
            "is_mated",
            "saliency_feature",
            "instability_predictor",
            "sensitivity_metric",
            "event_metric",
        )
        if column in frame
    ]
    return frame.sort_values(keys, kind="stable").reset_index(drop=True)


def test_partitioned_associations_match_in_memory_contract(tmp_path: Path) -> None:
    retrieval_path = tmp_path / "retrieval.csv"
    manifest_path = tmp_path / "projection" / "manifest.json"
    retrieval = _retrieval()
    retrieval.to_csv(retrieval_path, index=False)

    streamed = stream_partition_saliency_retrieval_projection(
        _saliency(),
        retrieval_path,
        joined_output_path=None,
        projection_manifest_path=manifest_path,
        chunksize=3,
        expected_rows=len(retrieval),
    )
    manifest = load_partitioned_projection_manifest(manifest_path)
    assert streamed.row_count == len(retrieval)
    assert manifest["partition_count"] == 1
    assert manifest["feature_row_count"] == len(_saliency())

    rehydrated = pd.concat(
        iter_partitioned_projection_batches(manifest_path, chunksize=3),
        ignore_index=True,
    )
    assert len(rehydrated) == len(retrieval)
    assert set(DEFAULT_RETRIEVAL_METRICS).issubset(rehydrated)

    kwargs = {
        "bootstrap_repeats": 12,
        "seed": 8972,
        "bootstrap_rank_strategy": WEIGHTED_RERANK_STRATEGY,
        "bootstrap_batch_size": 4,
    }
    result = compute_partitioned_retrieval_associations(
        manifest_path,
        **kwargs,
        paired_saliency_features=("outside_face_attention", "saliency_entropy"),
        paired_event_metrics=("threshold_crossing", "false_accept_gain"),
        paired_minimum_event_count=1,
        paired_confidence_level=0.95,
        max_workers=1,
        max_in_flight=2,
    )
    parallel = compute_partitioned_retrieval_associations(
        manifest_path,
        **kwargs,
        paired_saliency_features=("outside_face_attention", "saliency_entropy"),
        paired_event_metrics=("threshold_crossing", "false_accept_gain"),
        paired_minimum_event_count=1,
        paired_confidence_level=0.95,
        max_workers=2,
        max_in_flight=2,
    )
    expected_join = join_population_saliency_with_retrieval(
        _saliency(),
        retrieval,
    )
    expected_retrieval = saliency_retrieval_associations(
        expected_join,
        **kwargs,
    )
    expected_instability = threshold_instability_associations(
        expected_join,
        **kwargs,
    )
    expected_policy = threshold_policy_event_comparisons(
        expected_join,
        event_metrics=("threshold_crossing", "false_accept_gain"),
        bootstrap_repeats=12,
        confidence_level=0.95,
        seed=8972,
    )
    expected_rho = threshold_policy_saliency_rho_comparisons(
        expected_join,
        saliency_features=("outside_face_attention", "saliency_entropy"),
        event_metrics=("threshold_crossing", "false_accept_gain"),
        minimum_event_count=1,
        bootstrap_repeats=12,
        confidence_level=0.95,
        seed=8972,
        bootstrap_batch_size=4,
    )
    for actual, expected in (
        (result.retrieval_associations, expected_retrieval),
        (result.threshold_instability_associations, expected_instability),
        (result.threshold_policy_comparisons, expected_policy),
        (result.threshold_policy_saliency_rho_comparisons, expected_rho),
    ):
        pd.testing.assert_frame_equal(
            _sorted(actual),
            _sorted(expected),
            check_exact=True,
        )
    for single, multi in (
        (result.retrieval_associations, parallel.retrieval_associations),
        (
            result.threshold_instability_associations,
            parallel.threshold_instability_associations,
        ),
        (
            result.threshold_policy_comparisons,
            parallel.threshold_policy_comparisons,
        ),
        (
            result.threshold_policy_saliency_rho_comparisons,
            parallel.threshold_policy_saliency_rho_comparisons,
        ),
    ):
        pd.testing.assert_frame_equal(
            _sorted(single),
            _sorted(multi),
            check_exact=True,
        )
