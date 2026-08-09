from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.generate_step4_compact_summaries import summarize_retrieval


def _retrieval_rows(*, include_search_schema: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for search_mode, score_space, comparable, frozen_applicable in (
        (
            "pq_reconstruction_cosine",
            "cosine_similarity",
            True,
            True,
        ),
        (
            "pq_adc_exhaustive",
            "negative_squared_l2_adc",
            False,
            False,
        ),
    ):
        for query_id, is_mated in (("q1", True), ("q2", False)):
            row: dict[str, object] = {
                "query_id": query_id,
                "compression_family": "pq",
                "compression_profile": "pq_m128_b8",
                "threshold_policy": "recalibrated_compressed",
                "is_mated": is_mated,
                "origin_rank1_correct": is_mated,
                "compressed_rank1_correct": is_mated,
                "origin_top_k_correct": is_mated,
                "compressed_top_k_correct": is_mated,
                "origin_accepted": is_mated,
                "compressed_accepted": is_mated,
                "threshold_crossing": False,
                "origin_decision_correct": True,
                "compressed_decision_correct": True,
                "agreement_with_origin": True,
                "origin_fallback_used": False,
                "top_k_identity_jaccard": 1.0,
                "top1_score_drift": (
                    np.nan if search_mode == "pq_adc_exhaustive" else 0.01
                ),
                "top_k": 1,
                "protocol_uid": "protocol-1",
                "threshold_source_split": "calibration",
                "evaluation_split": "test",
                "storage_bytes_per_embedding": 128,
                "codebook_bytes": 4096,
                "codebook_bytes_source": "faiss",
                "extraction_uid": "extract-1",
                "dataset_id": "survface",
                "origin_embedding_artifact_uid": "origin-1",
                "decision_threshold": np.nan,
                "origin_decision_threshold": 0.5,
                "compressed_decision_threshold": (
                    -2.0 if search_mode == "pq_adc_exhaustive" else 0.4
                ),
            }
            if include_search_schema:
                row.update(
                    {
                        "search_mode": search_mode,
                        "origin_score_space": "cosine_similarity",
                        "compressed_score_space": score_space,
                        "score_spaces_comparable": comparable,
                        "frozen_origin_threshold_applicable": frozen_applicable,
                    }
                )
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def test_retrieval_summary_keeps_adc_and_reconstruction_separate(tmp_path) -> None:
    source = tmp_path / "retrieval_metrics.csv"
    _retrieval_rows(include_search_schema=True).to_csv(source, index=False)

    summary, row_count = summarize_retrieval(source, chunksize=1)

    assert row_count == 4
    assert len(summary) == 2
    assert set(summary["search_mode"]) == {
        "pq_reconstruction_cosine",
        "pq_adc_exhaustive",
    }
    adc = summary.loc[summary["search_mode"].eq("pq_adc_exhaustive")].iloc[0]
    assert adc["compressed_score_space"] == "negative_squared_l2_adc"
    assert bool(adc["score_spaces_comparable"]) is False
    assert bool(adc["frozen_origin_threshold_applicable"]) is False
    assert np.isnan(adc["mean_top1_score_drift"])
    assert adc["origin_fpir_denominator"] == 1
    assert adc["compressed_fpir_denominator"] == 1
    assert adc["origin_false_accept_count"] == 0
    assert adc["compressed_false_accept_count"] == 0
    assert adc["origin_fpir_wilson95_low"] <= adc["origin_fpir"]
    assert adc["origin_fpir"] <= adc["origin_fpir_wilson95_high"]
    assert adc["origin_realized_fpir"] == adc["origin_fpir"]
    assert adc["compressed_fpir_wilson95_low"] <= adc["compressed_fpir"]
    assert adc["compressed_fpir"] <= adc["compressed_fpir_wilson95_high"]
    assert adc["compressed_realized_fpir"] == adc["compressed_fpir"]
    assert adc["compressed_minus_origin_fpir"] == 0.0
    assert adc["confidence_interval_unit"] == "probe"


def test_legacy_retrieval_summary_infers_reconstruction_cosine(tmp_path) -> None:
    source = tmp_path / "legacy_retrieval_metrics.csv"
    legacy = _retrieval_rows(include_search_schema=False).iloc[:2]
    legacy.to_csv(source, index=False)

    summary, row_count = summarize_retrieval(source, chunksize=2)

    assert row_count == 2
    assert summary["search_mode"].tolist() == ["pq_reconstruction_cosine"]
    assert summary["compressed_score_space"].tolist() == [
        "cosine_similarity"
    ]
    assert summary["score_spaces_comparable"].tolist() == [True]
    assert summary["frozen_origin_threshold_applicable"].tolist() == [True]


def test_dataframe_summary_splits_threshold_crossing_directions() -> None:
    rows = _retrieval_rows(include_search_schema=True).iloc[:2].copy()
    rows.loc[rows["query_id"].eq("q1"), "origin_accepted"] = True
    rows.loc[rows["query_id"].eq("q1"), "compressed_accepted"] = False
    rows.loc[rows["query_id"].eq("q1"), "threshold_crossing"] = True
    rows.loc[rows["query_id"].eq("q2"), "origin_accepted"] = False
    rows.loc[rows["query_id"].eq("q2"), "compressed_accepted"] = True
    rows.loc[rows["query_id"].eq("q2"), "threshold_crossing"] = True

    summary, row_count = summarize_retrieval(
        None,
        chunksize=len(rows),
        source_frame=rows,
    )

    assert row_count == 2
    assert len(summary) == 1
    result = summary.iloc[0]
    assert result["threshold_crossing_count"] == 2
    assert result["accept_to_reject_count"] == 1
    assert result["reject_to_accept_count"] == 1
    assert result["accept_to_reject_rate"] == 0.5
    assert result["reject_to_accept_rate"] == 0.5


def test_retrieval_summary_keeps_fpir_targets_separate() -> None:
    base = _retrieval_rows(include_search_schema=True).iloc[:2].copy()
    first = base.assign(target_fpir=0.10)
    second = base.assign(target_fpir=0.01)
    rows = pd.concat([first, second], ignore_index=True)

    summary, row_count = summarize_retrieval(
        None,
        chunksize=len(rows),
        source_frame=rows,
    )

    assert row_count == 4
    assert len(summary) == 2
    assert set(summary["target_fpir"]) == {0.10, 0.01}
