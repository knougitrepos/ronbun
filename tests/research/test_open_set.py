import numpy as np
import pandas as pd
import pytest

from research.search.open_set import (
    CALIBRATION_FEATURE_COLUMNS,
    build_certified_search_features,
    build_search_features,
    summarize_certified_search_features,
)


def test_build_search_features_has_stable_schema_for_registered_and_unknown_probes():
    templates = pd.DataFrame(
        {
            "identity_id": ["a", "b"],
            "embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            "quality": [0.9, 0.8],
            "variance": [0.01, 0.02],
            "enrollment_count": [2, 1],
        }
    )
    probes = pd.DataFrame(
        {
            "image_id": ["qa", "qu"],
            "identity_id": ["a", "u"],
            "probe_type": ["registered", "known_unknown"],
            "embedding": [np.array([0.95, 0.05]), np.array([0.6, 0.4])],
            "quality": [0.7, 0.5],
            "reconstruction_error_norm": [0.0, 1.5],
        }
    )

    features = build_search_features(probes, templates, compression_profile="pca_2", top_k=2)

    assert list(features["query_id"]) == ["qa", "qu"]
    assert features.loc[0, "top1_identity"] == "a"
    assert bool(features.loc[0, "is_mated"]) is True
    assert bool(features.loc[0, "top1_correct"]) is True
    assert features.loc[0, "y_true_accept"] == 1
    assert bool(features.loc[1, "is_mated"]) is False
    assert bool(features.loc[1, "top1_correct"]) is False
    assert features.loc[1, "y_true_accept"] == 0
    assert features.loc[0, "score_margin"] > 0.0
    assert set(CALIBRATION_FEATURE_COLUMNS).issubset(features.columns)
    assert set(features["compression_profile"]) == {"pca_2"}


def test_build_certified_search_features_adds_bound_decisions_without_replacing_baseline_scores():
    templates = pd.DataFrame(
        {
            "identity_id": ["a", "b"],
            "embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            "quality": [0.9, 0.8],
            "variance": [0.01, 0.02],
            "enrollment_count": [2, 1],
            "angular_error": [0.05, 0.05],
        }
    )
    probes = pd.DataFrame(
        {
            "image_id": ["qa", "qu"],
            "identity_id": ["a", "u"],
            "probe_type": ["registered", "unknown_unknown"],
            "embedding": [np.array([1.0, 0.0]), np.array([-1.0, 0.0])],
            "angular_error": [0.02, 0.0],
            "quality": [0.7, 0.5],
            "reconstruction_error_norm": [0.0, 1.5],
        }
    )

    features = build_certified_search_features(
        probes,
        templates,
        compression_profile="pca_2",
        threshold=0.80,
        top_k=2,
    )

    assert list(features["top1_identity"]) == ["a", "b"]
    assert list(features["certified_decision"]) == ["accept", "reject"]
    assert list(features["certified_identity"]) == ["a", None]
    assert list(features["certified_fallback_required"]) == [False, False]
    assert list(features["fallback_used"]) == [False, False]
    assert list(features["final_decision"]) == ["accept", "reject"]
    assert list(features["final_identity"]) == ["a", None]
    assert list(features["final_decision_source"]) == ["certified_bound", "certified_bound"]
    assert bool(features.loc[0, "certified_rank"]) is True
    assert features.loc[0, "certified_top1_lower_bound"] <= features.loc[0, "top1_score"]
    assert features.loc[0, "certified_query_angular_error"] == pytest.approx(0.02)
    assert features.loc[0, "certified_top1_template_angular_error"] == pytest.approx(0.05)
    assert features.loc[0, "certified_top1_total_angular_error"] == pytest.approx(0.07)
    assert features.loc[0, "certified_top1_approximate_angle"] == pytest.approx(0.0)
    assert features.loc[0, "certified_top1_bound_width"] == pytest.approx(
        features.loc[0, "certified_top1_upper_bound"]
        - features.loc[0, "certified_top1_lower_bound"]
    )
    assert features.loc[0, "certified_top1_threshold_margin"] > 0.0
    assert features.loc[0, "certified_rank_margin"] > 0.0
    assert features.loc[0, "certified_decision_margin"] == pytest.approx(
        min(
            features.loc[0, "certified_top1_threshold_margin"],
            features.loc[0, "certified_rank_margin"],
        )
    )
    assert features.loc[0, "top1_score"] == 1.0
    assert features.loc[1, "certified_max_upper_bound"] < 0.80
    assert features.loc[1, "certified_reject_margin"] == pytest.approx(
        0.80 - features.loc[1, "certified_max_upper_bound"]
    )
    assert features.loc[1, "certified_decision_margin"] == pytest.approx(
        features.loc[1, "certified_reject_margin"]
    )


def test_build_certified_search_features_resolves_defer_with_exact_fallback_embeddings():
    templates = pd.DataFrame(
        {
            "identity_id": ["a", "b"],
            "embedding": [np.array([0.80, 0.60]), np.array([0.78, 0.62])],
            "fallback_embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            "quality": [0.9, 0.8],
            "variance": [0.01, 0.02],
            "enrollment_count": [2, 1],
            "angular_error": [0.20, 0.20],
        }
    )
    probes = pd.DataFrame(
        {
            "image_id": ["qa"],
            "identity_id": ["a"],
            "probe_type": ["registered"],
            "embedding": [np.array([1.0, 0.0])],
            "fallback_embedding": [np.array([1.0, 0.0])],
            "quality": [0.7],
            "reconstruction_error_norm": [0.0],
        }
    )

    features = build_certified_search_features(
        probes,
        templates,
        compression_profile="pca_2",
        threshold=0.70,
        top_k=2,
    )

    assert features.loc[0, "certified_decision"] == "defer"
    assert features.loc[0, "certified_identity"] is None
    assert bool(features.loc[0, "certified_fallback_required"]) is True
    assert bool(features.loc[0, "fallback_used"]) is True
    assert features.loc[0, "fallback_query_source"] == "fallback_embedding"
    assert features.loc[0, "fallback_template_source"] == "fallback_embedding"
    assert features.loc[0, "fallback_decision"] == "accept"
    assert features.loc[0, "fallback_identity"] == "a"
    assert features.loc[0, "fallback_top1_score"] == 1.0
    assert features.loc[0, "final_decision"] == "accept"
    assert features.loc[0, "final_identity"] == "a"
    assert features.loc[0, "final_decision_source"] == "exact_fallback"


def test_build_certified_search_features_does_not_call_fallback_exact_without_full_precision_query():
    templates = pd.DataFrame(
        {
            "identity_id": ["a", "b"],
            "embedding": [np.array([0.80, 0.60]), np.array([0.78, 0.62])],
            "fallback_embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            "quality": [0.9, 0.8],
            "variance": [0.01, 0.02],
            "enrollment_count": [2, 1],
            "angular_error": [0.20, 0.20],
        }
    )
    probes = pd.DataFrame(
        {
            "image_id": ["qa"],
            "identity_id": ["a"],
            "probe_type": ["registered"],
            "embedding": [np.array([1.0, 0.0])],
            "angular_error": [0.10],
            "quality": [0.7],
            "reconstruction_error_norm": [0.0],
        }
    )

    features = build_certified_search_features(
        probes,
        templates,
        compression_profile="pca_2",
        threshold=0.70,
        top_k=2,
    )

    assert features.loc[0, "certified_decision"] == "defer"
    assert bool(features.loc[0, "fallback_used"]) is False
    assert features.loc[0, "fallback_query_source"] is None
    assert features.loc[0, "fallback_template_source"] is None
    assert features.loc[0, "final_decision"] == "defer"
    assert features.loc[0, "final_decision_source"] == "defer_unresolved"


def test_build_certified_search_features_marks_candidate_set_scope_without_global_claim():
    templates = pd.DataFrame(
        {
            "identity_id": ["a", "b"],
            "embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            "quality": [0.9, 0.8],
            "variance": [0.01, 0.02],
            "enrollment_count": [2, 1],
            "angular_error": [0.05, 0.05],
        }
    )
    probes = pd.DataFrame(
        {
            "image_id": ["qa"],
            "identity_id": ["a"],
            "probe_type": ["registered"],
            "embedding": [np.array([1.0, 0.0])],
            "quality": [0.7],
            "reconstruction_error_norm": [0.0],
        }
    )

    features = build_certified_search_features(
        probes,
        templates,
        compression_profile="pca_2",
        threshold=0.80,
        top_k=2,
        candidate_scope="candidate_set",
        gallery_size=100,
    )

    assert features.loc[0, "certification_candidate_scope"] == "candidate_set"
    assert features.loc[0, "certification_candidate_count"] == 2
    assert features.loc[0, "certification_gallery_size"] == 100
    assert bool(features.loc[0, "certification_global_claim"]) is False


def test_build_certified_search_features_requires_gallery_size_for_candidate_set_scope():
    templates = pd.DataFrame(
        {
            "identity_id": ["a", "b"],
            "embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            "quality": [0.9, 0.8],
            "variance": [0.01, 0.02],
            "enrollment_count": [2, 1],
            "angular_error": [0.05, 0.05],
        }
    )
    probes = pd.DataFrame(
        {
            "image_id": ["qa"],
            "identity_id": ["a"],
            "probe_type": ["registered"],
            "embedding": [np.array([1.0, 0.0])],
            "quality": [0.7],
            "reconstruction_error_norm": [0.0],
        }
    )

    with pytest.raises(ValueError, match="gallery_size is required"):
        build_certified_search_features(
            probes,
            templates,
            compression_profile="pca_2",
            threshold=0.80,
            top_k=2,
            candidate_scope="candidate_set",
        )


def test_build_certified_search_features_rejects_fractional_gallery_size():
    templates = pd.DataFrame(
        {
            "identity_id": ["a", "b"],
            "embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            "quality": [0.9, 0.8],
            "variance": [0.01, 0.02],
            "enrollment_count": [2, 1],
            "angular_error": [0.05, 0.05],
        }
    )
    probes = pd.DataFrame(
        {
            "image_id": ["qa"],
            "identity_id": ["a"],
            "probe_type": ["registered"],
            "embedding": [np.array([1.0, 0.0])],
            "quality": [0.7],
            "reconstruction_error_norm": [0.0],
        }
    )

    with pytest.raises(ValueError, match="gallery_size must be a positive integer"):
        build_certified_search_features(
            probes,
            templates,
            compression_profile="pca_2",
            threshold=0.80,
            top_k=2,
            candidate_scope="candidate_set",
            gallery_size=100.5,
        )


def test_build_certified_search_features_rejects_exhaustive_scope_when_gallery_size_differs():
    templates = pd.DataFrame(
        {
            "identity_id": ["a", "b"],
            "embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            "quality": [0.9, 0.8],
            "variance": [0.01, 0.02],
            "enrollment_count": [2, 1],
            "angular_error": [0.05, 0.05],
        }
    )
    probes = pd.DataFrame(
        {
            "image_id": ["qa"],
            "identity_id": ["a"],
            "probe_type": ["registered"],
            "embedding": [np.array([1.0, 0.0])],
            "quality": [0.7],
            "reconstruction_error_norm": [0.0],
        }
    )

    with pytest.raises(ValueError, match="gallery_size must equal the supplied candidate count"):
        build_certified_search_features(
            probes,
            templates,
            compression_profile="pca_2",
            threshold=0.80,
            top_k=2,
            candidate_scope="exhaustive",
            gallery_size=100,
        )


def test_summarizes_certified_search_feature_frame_for_result_tables():
    features = pd.DataFrame(
        {
            "probe_type": [
                "registered",
                "known_unknown",
                "unknown_unknown",
                "registered",
                "unknown_unknown",
            ],
            "certified_decision": ["accept", "reject", "defer", "accept", "defer"],
            "certified_fallback_required": [False, False, True, False, True],
            "certification_candidate_scope": [
                "exhaustive",
                "exhaustive",
                "candidate_set",
                "exhaustive",
                "candidate_set",
            ],
            "final_decision": ["accept", "reject", "accept", "accept", "defer"],
            "final_decision_source": [
                "certified_bound",
                "certified_bound",
                "exact_fallback",
                "certified_bound",
                "defer_unresolved",
            ],
            "fallback_used": [False, False, True, False, False],
            "certified_top1_bound_width": [0.10, 0.20, 0.40, 0.30, 0.50],
            "certified_decision_margin": [0.05, 0.07, np.nan, 0.09, np.nan],
            "certified_query_angular_error": [0.01, 0.02, 0.03, 0.04, 0.05],
            "certified_top1_template_angular_error": [0.05, 0.06, 0.07, 0.08, 0.09],
            "certified_top1_total_angular_error": [0.06, 0.08, 0.10, 0.12, 0.14],
        }
    )

    summary = summarize_certified_search_features(features)

    assert summary["total"] == 5
    assert summary["decision_counts"] == {"accept": 2, "reject": 1, "defer": 2}
    assert summary["certification_coverage"] == 0.60
    assert summary["defer_rate"] == 0.40
    assert summary["fallback_rate"] == 0.40
    assert summary["exact_fallback_rate"] == 0.20
    assert summary["fallback_resolution_rate"] == 0.50
    assert summary["mean_top1_bound_width"] == pytest.approx(0.30)
    assert summary["max_top1_bound_width"] == pytest.approx(0.50)
    assert summary["mean_certified_decision_margin"] == pytest.approx(0.07)
    assert summary["mean_query_angular_error"] == pytest.approx(0.03)
    assert summary["mean_top1_template_angular_error"] == pytest.approx(0.07)
    assert summary["mean_top1_total_angular_error"] == pytest.approx(0.10)
    assert summary["candidate_scope_counts"] == {"candidate_set": 2, "exhaustive": 3}
    assert summary["final_decision_counts"] == {"accept": 3, "reject": 1, "defer": 1}
    assert summary["by_probe_type"]["registered"]["decision_counts"] == {"accept": 2, "reject": 0, "defer": 0}
    assert summary["by_probe_type"]["known_unknown"]["certification_coverage"] == 1.0
    assert summary["by_probe_type"]["unknown_unknown"]["final_decision_counts"] == {
        "accept": 1,
        "reject": 0,
        "defer": 1,
    }
    assert summary["by_probe_type"]["unknown_unknown"]["exact_fallback_rate"] == 0.5
    assert summary["by_probe_type"]["unknown_unknown"]["fallback_resolution_rate"] == 0.5


def test_exact_fallback_rate_uses_execution_flag_for_pgvector_source_name():
    features = pd.DataFrame(
        {
            "probe_type": ["registered", "known_unknown", "unknown_unknown"],
            "certified_decision": ["accept", "defer", "defer"],
            "certified_fallback_required": [False, True, True],
            "fallback_used": [False, True, True],
            "final_decision_source": [
                "candidate_certificate",
                "origin_512_db_exact_fallback",
                "origin_512_db_exact_fallback",
            ],
        }
    )

    summary = summarize_certified_search_features(features)

    assert summary["defer_rate"] == pytest.approx(2 / 3)
    assert summary["fallback_rate"] == pytest.approx(2 / 3)
    assert summary["exact_fallback_rate"] == pytest.approx(2 / 3)
    assert summary["fallback_resolution_rate"] == 1.0
