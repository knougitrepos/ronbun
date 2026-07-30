import numpy as np

from research.search.certification import (
    certify_open_set_decision,
    compute_similarity_bounds,
    exact_open_set_decision,
    summarize_certified_decisions,
)


def test_similarity_bounds_contain_true_scores():
    query = np.array([1.0, 0.0], dtype=np.float32)
    true_templates = np.array(
        [
            [0.99, 0.10],
            [0.10, 0.99],
        ],
        dtype=np.float32,
    )
    compressed_templates = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    template_errors = np.arccos(
        np.clip(
            np.sum(
                true_templates / np.linalg.norm(true_templates, axis=1, keepdims=True)
                * compressed_templates,
                axis=1,
            ),
            -1.0,
            1.0,
        )
    )

    bounds = compute_similarity_bounds(query, compressed_templates, template_errors)
    true_scores = (
        true_templates / np.linalg.norm(true_templates, axis=1, keepdims=True)
    ) @ query

    assert np.all(bounds.lower_bounds <= true_scores + 1e-6)
    assert np.all(true_scores <= bounds.upper_bounds + 1e-6)


def test_certifies_accept_when_rank_and_threshold_are_bounded():
    result = certify_open_set_decision(
        query=np.array([1.0, 0.0], dtype=np.float32),
        compressed_templates=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        template_identities=["a", "b"],
        template_angular_errors=np.array([0.05, 0.05], dtype=np.float32),
        threshold=0.80,
    )

    assert result.decision == "accept"
    assert result.selected_identity == "a"
    assert result.rank_certified is True


def test_certifies_reject_when_all_upper_bounds_are_below_threshold():
    result = certify_open_set_decision(
        query=np.array([1.0, 0.0], dtype=np.float32),
        compressed_templates=np.array(
            [
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        template_identities=["a", "b"],
        template_angular_errors=np.array([0.05, 0.05], dtype=np.float32),
        threshold=0.90,
    )

    assert result.decision == "reject"
    assert result.selected_identity is None
    assert result.rank_certified is False


def test_defers_when_rank_bounds_overlap():
    result = certify_open_set_decision(
        query=np.array([1.0, 0.0], dtype=np.float32),
        compressed_templates=np.array(
            [
                [0.80, 0.60],
                [0.78, 0.62],
            ],
            dtype=np.float32,
        ),
        template_identities=["a", "b"],
        template_angular_errors=np.array([0.20, 0.20], dtype=np.float32),
        threshold=0.70,
    )

    assert result.decision == "defer"
    assert result.selected_identity == "a"
    assert result.rank_certified is False


def test_summarizes_certification_coverage_and_defer_rate():
    decisions = [
        certify_open_set_decision(
            query=np.array([1.0, 0.0], dtype=np.float32),
            compressed_templates=np.array([[1.0, 0.0]], dtype=np.float32),
            template_identities=["a"],
            template_angular_errors=np.array([0.0], dtype=np.float32),
            threshold=0.8,
        ),
        certify_open_set_decision(
            query=np.array([1.0, 0.0], dtype=np.float32),
            compressed_templates=np.array([[0.0, 1.0]], dtype=np.float32),
            template_identities=["a"],
            template_angular_errors=np.array([0.0], dtype=np.float32),
            threshold=0.8,
        ),
        certify_open_set_decision(
            query=np.array([1.0, 0.0], dtype=np.float32),
            compressed_templates=np.array([[0.80, 0.60], [0.78, 0.62]], dtype=np.float32),
            template_identities=["a", "b"],
            template_angular_errors=np.array([0.20, 0.20], dtype=np.float32),
            threshold=0.70,
        ),
    ]

    summary = summarize_certified_decisions(decisions)

    assert summary.counts == {"accept": 1, "reject": 1, "defer": 1}
    assert summary.certification_coverage == 2 / 3
    assert summary.accept_coverage == 1 / 3
    assert summary.reject_coverage == 1 / 3
    assert summary.defer_rate == 1 / 3


def test_exact_open_set_decision_accepts_or_rejects_with_full_precision_scores():
    templates = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    accepted = exact_open_set_decision(
        query=np.array([1.0, 0.0], dtype=np.float32),
        templates=templates,
        template_identities=["a", "b"],
        threshold=0.80,
    )
    rejected = exact_open_set_decision(
        query=np.array([-1.0, 0.0], dtype=np.float32),
        templates=templates,
        template_identities=["a", "b"],
        threshold=0.80,
    )

    assert accepted.decision == "accept"
    assert accepted.selected_identity == "a"
    assert accepted.top1_identity == "a"
    assert accepted.top1_score == 1.0
    assert rejected.decision == "reject"
    assert rejected.selected_identity is None
    assert rejected.top1_score < 0.80
