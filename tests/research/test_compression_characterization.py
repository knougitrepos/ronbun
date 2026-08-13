import numpy as np
import pytest

from research.evaluation import (
    ORIGIN_RETRIEVAL_COLUMNS,
    PAIRED_EMBEDDING_COLUMNS,
    RETRIEVAL_COMPARISON_COLUMNS,
    apply_retrieval_thresholds,
    compare_cosine_retrieval,
    compare_pq_adc_retrieval,
    origin_cosine_retrieval,
    paired_embedding_metrics,
)


def test_paired_embedding_metrics_uses_explicit_reconstruction_without_fallback():
    original = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    search = np.array([[1.0], [0.5]], dtype=np.float32)
    reconstructed = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    result = paired_embedding_metrics(
        original,
        search,
        reconstructed_embeddings=reconstructed,
        sample_ids=["a", "b"],
        compression_family="pca",
        compression_profile="pca_1",
    )

    assert tuple(result.columns) == PAIRED_EMBEDDING_COLUMNS
    assert result["sample_id"].tolist() == ["a", "b"]
    assert result["metric_vector_source"].tolist() == ["reconstruction", "reconstruction"]
    assert result["reconstruction_available"].all()
    assert not result["origin_fallback_used"].any()
    assert result["reconstruction_mse"].to_numpy() == pytest.approx([0.0, 0.5])
    assert result["cosine_to_origin"].to_numpy() == pytest.approx(
        [1.0, 1.0 / np.sqrt(2.0)]
    )
    assert result["angular_error_rad"].to_numpy() == pytest.approx([0.0, np.pi / 4.0])


def test_paired_embedding_metrics_does_not_invent_lower_dimensional_reconstruction():
    result = paired_embedding_metrics(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.array([[1.0], [0.5]], dtype=np.float32),
        compression_family="pca",
        compression_profile="pca_1",
    )

    assert result["metric_vector_source"].eq("unavailable").all()
    assert not result["reconstruction_available"].any()
    assert result["reconstruction_mse"].isna().all()
    assert result["angular_error_rad"].isna().all()
    assert result["cosine_to_origin"].isna().all()
    assert not result["origin_fallback_used"].any()


def test_origin_cosine_retrieval_searches_one_space_with_deterministic_ties():
    gallery = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    queries = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )

    result = origin_cosine_retrieval(
        queries,
        gallery,
        query_ids=["q-a", "q-b", "q-u"],
        gallery_ids=["g-a", "g-b-first", "g-b-second"],
        query_identity_ids=["a", "b", "unknown"],
        gallery_identity_ids=["a", "b", "b-duplicate"],
        top_k=2,
        query_batch_size=2,
        gallery_batch_size=1,
        max_pairwise_elements=6,
    )

    assert tuple(result.columns) == ORIGIN_RETRIEVAL_COLUMNS
    assert result["origin_top1_gallery_id"].tolist() == [
        "g-a",
        "g-b-first",
        "g-a",
    ]
    assert result["is_mated"].tolist() == [True, True, False]
    assert result["origin_rank1_correct"].tolist() == [True, True, False]
    assert result["origin_top_k_correct"].tolist() == [True, True, False]
    assert result["origin_top1_score"].to_numpy() == pytest.approx(
        [1.0, 1.0, 1.0 / np.sqrt(2.0)]
    )


def test_compare_cosine_retrieval_separates_truth_agreement_and_crossing():
    root_two = np.sqrt(2.0)
    original_gallery = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    compressed_gallery = original_gallery.copy()
    original_queries = np.array(
        [[1.0, 0.0], [1.0 / root_two, 1.0 / root_two]],
        dtype=np.float32,
    )
    compressed_queries = np.array([[0.6, 0.8], [1.0, 0.0]], dtype=np.float32)

    result = compare_cosine_retrieval(
        original_queries,
        original_gallery,
        compressed_queries,
        compressed_gallery,
        query_ids=["q-a", "q-u"],
        gallery_ids=["g-a", "g-b"],
        query_identity_ids=["a", "unknown"],
        gallery_identity_ids=["a", "b"],
        compression_family="pca",
        compression_profile="pca_2",
        top_k=2,
        threshold=0.9,
        query_batch_size=1,
        gallery_batch_size=1,
        max_pairwise_elements=3,
    )

    assert tuple(result.columns) == RETRIEVAL_COMPARISON_COLUMNS
    assert not result["origin_fallback_used"].any()

    mated = result.iloc[0]
    assert bool(mated["is_mated"])
    assert bool(mated["origin_rank1_correct"])
    assert not bool(mated["compressed_rank1_correct"])
    assert not bool(mated["agreement_with_origin"])
    assert mated["origin_top_k_identity_ids"] == ("a", "b")
    assert mated["compressed_top_k_identity_ids"] == ("b", "a")
    assert not bool(mated["top_k_identity_agreement"])
    assert mated["top_k_identity_jaccard"] == 1.0
    assert mated["compressed_score_at_origin_top1"] == pytest.approx(0.6)
    assert bool(mated["threshold_crossing"])
    assert mated["threshold_crossing_direction"] == "accept_to_reject"
    assert not bool(mated["compressed_decision_correct"])
    assert mated["decision_threshold"] == pytest.approx(0.9)
    assert mated["origin_decision_threshold"] == pytest.approx(0.9)
    assert mated["compressed_decision_threshold"] == pytest.approx(0.9)

    non_mated = result.iloc[1]
    assert not bool(non_mated["is_mated"])
    assert not bool(non_mated["origin_rank1_correct"])
    assert not bool(non_mated["compressed_rank1_correct"])
    assert bool(non_mated["agreement_with_origin"])
    assert bool(non_mated["threshold_crossing"])
    assert non_mated["threshold_crossing_direction"] == "reject_to_accept"
    assert bool(non_mated["origin_decision_correct"])
    assert not bool(non_mated["compressed_decision_correct"])


def test_compare_cosine_retrieval_supports_distinct_operating_thresholds():
    gallery = np.array([[1.0, 0.0]], dtype=np.float32)
    origin_query = np.array([[0.8, 0.6]], dtype=np.float32)
    compressed_query = np.array([[0.9, np.sqrt(0.19)]], dtype=np.float32)

    result = compare_cosine_retrieval(
        origin_query,
        gallery,
        compressed_query,
        gallery,
        query_ids=["q-a"],
        gallery_ids=["g-a"],
        query_identity_ids=["a"],
        gallery_identity_ids=["a"],
        compression_family="pca",
        compression_profile="pca_2",
        origin_threshold=0.75,
        compressed_threshold=0.95,
    )

    row = result.iloc[0]
    assert row["origin_decision_threshold"] == pytest.approx(0.75)
    assert row["compressed_decision_threshold"] == pytest.approx(0.95)
    assert row["decision_threshold"] is None
    assert bool(row["origin_accepted"])
    assert not bool(row["compressed_accepted"])
    assert bool(row["threshold_crossing"])
    assert row["threshold_crossing_direction"] == "accept_to_reject"
    assert bool(row["origin_decision_correct"])
    assert not bool(row["compressed_decision_correct"])

    with pytest.raises(ValueError, match="cannot be combined"):
        compare_cosine_retrieval(
            origin_query,
            gallery,
            compressed_query,
            gallery,
            query_ids=["q-a"],
            gallery_ids=["g-a"],
            query_identity_ids=["a"],
            gallery_identity_ids=["a"],
            compression_family="pca",
            compression_profile="pca_2",
            threshold=0.8,
            origin_threshold=0.8,
            compressed_threshold=0.8,
        )


def test_compare_pq_adc_retrieval_keeps_distance_space_separate() -> None:
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    origin = compare_cosine_retrieval(
        queries,
        gallery,
        queries,
        gallery,
        query_ids=["q-a", "q-u"],
        gallery_ids=["g-a", "g-b"],
        query_identity_ids=["a", "unknown"],
        gallery_identity_ids=["a", "b"],
        compression_family="pq",
        compression_profile="pq_512_m1_b2",
        top_k=2,
        search_mode="pq_reconstruction_cosine",
    )
    adc = compare_pq_adc_retrieval(
        origin,
        queries,
        gallery,
        np.array([[0.2, 0.9], [0.8, 1.1]], dtype=np.float32),
        np.array([[1, 0], [0, 1]], dtype=np.int64),
        query_ids=["q-a", "q-u"],
        gallery_ids=["g-a", "g-b"],
        query_identity_ids=["a", "unknown"],
        gallery_identity_ids=["a", "b"],
        compression_profile="pq_512_m1_b2",
    )

    assert adc["search_mode"].eq("pq_adc_exhaustive").all()
    assert adc["compressed_score_space"].eq("negative_squared_l2_adc").all()
    assert not adc["score_spaces_comparable"].any()
    assert not adc["frozen_origin_threshold_applicable"].any()
    assert adc["compressed_top1_score"].to_numpy() == pytest.approx([-0.2, -0.8])
    assert adc["top1_score_drift"].isna().all()
    assert adc["origin_winner_score_drift"].isna().all()
    assert not bool(adc.iloc[0]["agreement_with_origin"])

    thresholded = apply_retrieval_thresholds(
        adc,
        origin_threshold=0.9,
        compressed_threshold=-2.0,
    )
    assert thresholded["compressed_accepted"].all()


def test_threshold_policy_can_be_reapplied_without_repeating_search():
    gallery = np.array([[1.0, 0.0]], dtype=np.float32)
    origin_query = np.array([[0.8, 0.6]], dtype=np.float32)
    compressed_query = np.array([[0.9, np.sqrt(0.19)]], dtype=np.float32)
    progress_events = []
    base = compare_cosine_retrieval(
        origin_query,
        gallery,
        compressed_query,
        gallery,
        query_ids=["q-a"],
        gallery_ids=["g-a"],
        query_identity_ids=["a"],
        gallery_identity_ids=["a"],
        compression_family="pca",
        compression_profile="pca_2",
        progress=lambda message, details: progress_events.append(
            (message, details)
        ),
        progress_message="SurvFace compression retrieval",
        progress_offset=10,
        progress_total=12,
    )

    updated = apply_retrieval_thresholds(
        base,
        origin_threshold=0.75,
        compressed_threshold=0.95,
    )

    assert [event[1]["processed"] for event in progress_events] == [11, 12]
    row = updated.iloc[0]
    assert bool(row["origin_accepted"])
    assert not bool(row["compressed_accepted"])
    assert row["threshold_crossing_direction"] == "accept_to_reject"


def test_tpir_uses_true_identity_score_in_top_k_not_top1_maximum():
    gallery = np.eye(2, dtype=np.float32)
    base = compare_cosine_retrieval(
        np.array([[0.6, 0.8]], dtype=np.float32),
        gallery,
        np.array([[0.8, 0.6]], dtype=np.float32),
        gallery,
        query_ids=["q-a"],
        gallery_ids=["g-a", "g-b"],
        query_identity_ids=["a"],
        gallery_identity_ids=["a", "b"],
        compression_family="pca",
        compression_profile="pca_2",
        top_k=2,
    )

    thresholded = apply_retrieval_thresholds(
        base,
        origin_threshold=0.7,
        compressed_threshold=0.7,
    )
    row = thresholded.iloc[0]

    assert bool(row["origin_accepted"])
    assert row["origin_true_identity_rank"] == 2
    assert row["origin_true_identity_score"] == pytest.approx(0.6)
    assert not bool(row["origin_tpir_at_rank_k"])
    assert row["compressed_true_identity_rank"] == 1
    assert row["compressed_true_identity_score"] == pytest.approx(0.8)
    assert bool(row["compressed_tpir_at_rank_k"])


def test_compression_characterization_validates_alignment_and_memory_boundary():
    vectors = np.eye(2, dtype=np.float32)

    with pytest.raises(ValueError, match="same row count"):
        paired_embedding_metrics(
            vectors,
            np.ones((1, 1), dtype=np.float32),
            compression_family="pca",
            compression_profile="pca_1",
        )
    with pytest.raises(ValueError, match="zero-norm"):
        paired_embedding_metrics(
            vectors,
            np.zeros((2, 1), dtype=np.float32),
            compression_family="pca",
            compression_profile="pca_1",
        )
    with pytest.raises(ValueError, match="exceeds max_pairwise_elements"):
        compare_cosine_retrieval(
            vectors,
            vectors,
            vectors,
            vectors,
            query_ids=["q1", "q2"],
            gallery_ids=["g1", "g2"],
            query_identity_ids=["a", "b"],
            gallery_identity_ids=["a", "b"],
            compression_family="pca",
            compression_profile="pca_2",
            query_batch_size=2,
            gallery_batch_size=2,
            max_pairwise_elements=5,
        )
    with pytest.raises(ValueError, match="query_ids values must be unique"):
        compare_cosine_retrieval(
            vectors,
            vectors,
            vectors,
            vectors,
            query_ids=["q", "q"],
            gallery_ids=["g1", "g2"],
            query_identity_ids=["a", "b"],
            gallery_identity_ids=["a", "b"],
            compression_family="pca",
            compression_profile="pca_2",
        )
