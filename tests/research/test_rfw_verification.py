from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.evaluation.rfw_verification import (
    empirical_equal_error_rate,
    evaluate_rfw_10fold,
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
