from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.compression import PQCompressor
from research.evaluation.rfw_verification import (
    compute_rfw_pair_qualities,
    compute_rfw_pair_scores,
    empirical_equal_error_rate,
    evaluate_rfw_10fold,
    evaluate_rfw_continuous_fiqa_10fold,
)


def _fixture() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    image_ids: list[str] = []
    vectors: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    basis = np.eye(512, dtype=np.float64)
    for group_index, group in enumerate(("African", "Asian")):
        for fold in range(2):
            genuine_left = f"{group}-g{fold}-left"
            genuine_right = f"{group}-g{fold}-right"
            impostor_left = f"{group}-i{fold}-left"
            impostor_right = f"{group}-i{fold}-right"
            image_ids.extend(
                [genuine_left, genuine_right, impostor_left, impostor_right]
            )
            axis = group_index * 4 + fold * 2
            vectors.extend(
                [basis[axis], basis[axis], basis[axis], basis[axis + 1]]
            )
            rows.extend(
                [
                    {
                        "pair_id": f"{group}-{fold}-g",
                        "rfw_group": group,
                        "fold_index": fold,
                        "left_image_id": genuine_left,
                        "right_image_id": genuine_right,
                        "is_genuine": True,
                    },
                    {
                        "pair_id": f"{group}-{fold}-i",
                        "rfw_group": group,
                        "fold_index": fold,
                        "left_image_id": impostor_left,
                        "right_image_id": impostor_right,
                        "is_genuine": False,
                    },
                ]
            )
    return pd.DataFrame(rows), image_ids, np.vstack(vectors)


def test_rfw_10fold_diagnostic_uses_other_folds_and_reports_no_open_set_claim():
    pairs, image_ids, embeddings = _fixture()
    result = evaluate_rfw_10fold(
        pairs,
        image_ids=image_ids,
        embeddings=embeddings,
        thresholds=[-0.5, 0.5, 1.0],
        strict_official=False,
    )

    assert result.fold_metrics["accuracy"].eq(1.0).all()
    assert result.fold_metrics["train_pair_count"].eq(2).all()
    assert result.fold_metrics["test_pair_count"].eq(2).all()
    assert result.summary["macro_group_accuracy"] == pytest.approx(1.0)
    assert result.summary["macro_group_eer"] == pytest.approx(0.0)
    assert result.summary["group_eer_gap"] == pytest.approx(0.0)
    assert result.summary["eer_threshold_source"] == (
        "heldout_fold_scores_and_labels"
    )
    assert result.summary["eer_uses_internal_9fold_threshold"] is False
    assert result.fold_metrics["eer"].eq(0.0).all()
    assert result.fold_metrics["eer_threshold_source"].eq(
        "heldout_fold_scores_and_labels"
    ).all()
    assert result.group_summary["mean_eer"].eq(0.0).all()
    assert result.summary["group_accuracy_gap"] == pytest.approx(0.0)
    assert result.summary["open_set_protocol"] is False
    assert result.summary["codec_fit_on_rfw"] is False


def test_rfw_evaluation_rejects_missing_pair_embedding():
    pairs, image_ids, embeddings = _fixture()
    with pytest.raises(ValueError, match="missing embeddings"):
        evaluate_rfw_10fold(
            pairs,
            image_ids=image_ids[:-1],
            embeddings=embeddings[:-1],
            strict_official=False,
        )


def test_empirical_eer_uses_heldout_scores_and_preserves_ties():
    result = empirical_equal_error_rate(
        scores=[0.9, 0.7, 0.7, 0.2],
        labels=[True, True, False, False],
    )

    assert result.threshold == pytest.approx(0.9)
    assert result.far == pytest.approx(0.0)
    assert result.frr == pytest.approx(0.5)
    assert result.eer == pytest.approx(0.25)
    assert result.absolute_far_frr_gap == pytest.approx(0.5)
    assert result.method == "heldout_scores_minimum_absolute_far_frr_v1"


def test_compute_rfw_pair_scores_and_qualities():
    pairs, image_ids, embeddings = _fixture()
    cosine_scores = compute_rfw_pair_scores(
        pairs,
        image_ids=image_ids,
        embeddings=embeddings,
        codec=None,
    )
    assert len(cosine_scores) == len(pairs)
    assert np.all(np.isfinite(cosine_scores))
    assert np.all(cosine_scores[pairs["is_genuine"]] > cosine_scores[~pairs["is_genuine"]])

    # Test ADC with PQCompressor
    rng = np.random.default_rng(8972)
    dev_data = rng.normal(size=(256, 512)).astype(np.float32)
    dev_norms = np.linalg.norm(dev_data, axis=1, keepdims=True)
    dev_data /= np.where(dev_norms > 0, dev_norms, 1.0)
    codec = PQCompressor(source_dim=512, m=32, nbits=8, random_state=8972).fit(dev_data)
    adc_scores = compute_rfw_pair_scores(
        pairs,
        image_ids=image_ids,
        embeddings=embeddings,
        codec=codec,
    )
    assert len(adc_scores) == len(pairs)
    assert np.all(np.isfinite(adc_scores))

    # Test symmetric pair quality min(q_left, q_right)
    fiqa_dict = {img_id: float(i + 1) for i, img_id in enumerate(image_ids)}
    pair_q = compute_rfw_pair_qualities(pairs, fiqa_dict)
    assert len(pair_q) == len(pairs)
    for i, (_, row) in enumerate(pairs.iterrows()):
        expected = min(fiqa_dict[row["left_image_id"]], fiqa_dict[row["right_image_id"]])
        assert pair_q[i] == pytest.approx(expected)


def test_evaluate_rfw_continuous_fiqa_10fold():
    # Construct multi-fold fixture with sufficient non-mated samples for fit and safety partitions (>=20 each)
    rows = []
    rng = np.random.default_rng(8972)
    groups = ("African", "Asian")
    total_pairs = 0
    for group in groups:
        for fold in range(4):
            for i in range(250):
                is_genuine = (i % 2 == 0)
                pair_id = f"rfw:{group.lower()}:fold{fold:02d}:pair{i:03d}"
                rows.append({
                    "pair_id": pair_id,
                    "rfw_group": group,
                    "fold_index": fold,
                    "left_image_id": f"{group}-f{fold}-p{i}-L",
                    "right_image_id": f"{group}-f{fold}-p{i}-R",
                    "left_identity_id": f"id-{group}-f{fold}-p{i}-L",
                    "right_identity_id": f"id-{group}-f{fold}-p{i}-R",
                    "is_genuine": is_genuine,
                })
                total_pairs += 1
    pairs = pd.DataFrame(rows)
    # Synthetic scores and qualities
    scores = np.where(pairs["is_genuine"], rng.uniform(0.6, 0.9, size=total_pairs), rng.uniform(-0.2, 0.4, size=total_pairs))
    qualities = rng.uniform(20.0, 80.0, size=total_pairs)

    result = evaluate_rfw_continuous_fiqa_10fold(
        pairs,
        scores=scores,
        pair_qualities=qualities,
        target_fmrs=(0.1, 0.2),
        methods=("global_safe", "fiqa_5bin", "continuous_fiqa"),
        score_space="cosine",
        compression_profile="origin",
        strict_official=False,
        bootstrap_seed=8972,
        bootstrap_repeats=100,
        safety_fraction=0.3,
    )

    assert not result.fold_metrics.empty
    assert not result.group_summary.empty
    assert set(result.group_summary["method"]) == {"global_safe", "fiqa_5bin", "continuous_fiqa"}
    assert set(result.group_summary["rfw_group"]) == {"African", "Asian"}
    assert "mean_tar" in result.group_summary.columns
    assert "mean_realized_fmr" in result.group_summary.columns
    assert "accuracy_ci95_low" in result.group_summary.columns
    assert "cross_partition_pairs_excluded" in result.fold_metrics.columns
    assert "calibration_pairs_retained" in result.fold_metrics.columns
    assert result.summary["formal_fmr_guarantee"] is False
    assert result.summary["compression_profile"] == "origin"


def test_compute_rfw_pair_qualities_rejects_non_finite_array():
    pairs, _, _ = _fixture()
    with pytest.raises(ValueError, match="non-finite"):
        compute_rfw_pair_qualities(pairs, [50.0, np.nan] + [50.0] * (len(pairs) - 2))
    with pytest.raises(ValueError, match="aligned array"):
        compute_rfw_pair_qualities(pairs, [50.0, 50.0])


def test_evaluate_rfw_continuous_fiqa_handles_constant_quality():
    # Construct fixture where all pairs have constant FIQA score (zero variance)
    rows = []
    rng = np.random.default_rng(8972)
    groups = ("African", "Asian")
    total_pairs = 0
    for group in groups:
        for fold in range(4):
            for i in range(250):
                is_genuine = (i % 2 == 0)
                pair_id = f"rfw:{group.lower()}:fold{fold:02d}:pair{i:03d}"
                rows.append({
                    "pair_id": pair_id,
                    "rfw_group": group,
                    "fold_index": fold,
                    "left_image_id": f"{group}-f{fold}-p{i}-L",
                    "right_image_id": f"{group}-f{fold}-p{i}-R",
                    "left_identity_id": f"id-{group}-f{fold}-p{i}-L",
                    "right_identity_id": f"id-{group}-f{fold}-p{i}-R",
                    "is_genuine": is_genuine,
                })
                total_pairs += 1
    pairs = pd.DataFrame(rows)
    scores = np.where(pairs["is_genuine"], rng.uniform(0.6, 0.9, size=total_pairs), rng.uniform(-0.2, 0.4, size=total_pairs))
    constant_qualities = np.full(total_pairs, 50.0)

    result = evaluate_rfw_continuous_fiqa_10fold(
        pairs,
        scores=scores,
        pair_qualities=constant_qualities,
        target_fmrs=(0.1,),
        methods=("continuous_fiqa",),
        score_space="cosine",
        compression_profile="origin",
        strict_official=False,
        bootstrap_seed=8972,
        bootstrap_repeats=100,
        safety_fraction=0.3,
    )
    assert not result.fold_metrics.empty
    assert "mean_tar" in result.group_summary.columns


