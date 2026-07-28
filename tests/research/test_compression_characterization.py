import numpy as np
import pytest

from research.evaluation import (
    PAIRED_EMBEDDING_COLUMNS,
    RETRIEVAL_COMPARISON_COLUMNS,
    apply_retrieval_thresholds,
    compare_cosine_retrieval,
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
