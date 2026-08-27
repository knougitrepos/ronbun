from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.evaluation.retrieval_ledger import (
    RetrievalLedgerWriter,
    iter_retrieval_source_batches,
    load_retrieval_ledger_manifest,
)


def _batch(policy: str, threshold: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": ["q1", "q2"],
            "query_identity_id": ["i1", "i2"],
            "is_mated": [True, False],
            "dataset": ["lfw", "lfw"],
            "model_uid": ["model-a", "model-a"],
            "compression_family": ["pca", "pca"],
            "compression_profile": ["pca_32", "pca_32"],
            "search_mode": ["pca_reconstruction_cosine"] * 2,
            "protocol_uid": ["protocol-a", "protocol-a"],
            "threshold_source_split": ["calibration", "calibration"],
            "evaluation_split": ["test", "test"],
            "origin_fallback_used": [False, False],
            "origin_top1_score": [0.9, 0.2],
            "compressed_top1_score": [0.8, 0.3],
            "target_fpir": [0.1, 0.1],
            "threshold_policy": [policy, policy],
            "origin_decision_threshold": [0.5, 0.5],
            "compressed_decision_threshold": [threshold, threshold],
            "origin_accepted": [True, False],
            "compressed_accepted": [True, False],
            "threshold_crossing": [False, False],
            "origin_top_k_gallery_ids": [("g1",), ("g2",)],
            "compressed_top_k_gallery_ids": [("g1",), ("g2",)],
            "origin_top_k_identity_ids": [("i1",), ("i2",)],
            "compressed_top_k_identity_ids": [("i1",), ("i2",)],
            "origin_top_k_scores": [(0.9,), (0.2,)],
            "compressed_top_k_scores": [(0.8,), (0.3,)],
        }
    )


def test_retrieval_ledger_normalizes_core_and_round_trips(tmp_path: Path) -> None:
    manifest_path = tmp_path / "retrieval_ledger" / "manifest.json"
    with RetrievalLedgerWriter(
        manifest_path,
        lineage={
            "dataset_id": "lfw",
            "extraction_uid": "extract-a",
            "origin_embedding_artifact_uid": "origin-a",
        },
        include_topk_detail=True,
    ) as writer:
        writer.write(_batch("frozen_origin", 0.5))
        writer.write(_batch("recalibrated_compressed", 0.4))

    manifest = load_retrieval_ledger_manifest(manifest_path)
    assert manifest["logical_row_count"] == 4
    assert manifest["core_row_count"] == 2
    assert manifest["condition_count"] == 1
    assert manifest["decision_partition_count"] == 2
    assert manifest["topk_detail_retained"] is True

    restored = pd.concat(
        iter_retrieval_source_batches(manifest_path, chunksize=1),
        ignore_index=True,
    )
    assert len(restored) == 4
    assert set(restored["threshold_policy"]) == {
        "frozen_origin",
        "recalibrated_compressed",
    }
    assert restored["origin_top_k_gallery_ids"].map(tuple).tolist() == [
        ("g1",),
        ("g2",),
        ("g1",),
        ("g2",),
    ]


def test_results_only_ledger_omits_topk_and_rejects_core_drift(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "retrieval_ledger" / "manifest.json"
    with pytest.raises(ValueError, match="threshold-invariant"):
        with RetrievalLedgerWriter(
            manifest_path,
            include_topk_detail=False,
        ) as writer:
            writer.write(_batch("frozen_origin", 0.5))
            drifted = _batch("recalibrated_compressed", 0.4)
            drifted.loc[0, "compressed_top1_score"] = 0.7
            writer.write(drifted)
    assert not manifest_path.exists()

    with RetrievalLedgerWriter(
        manifest_path,
        include_topk_detail=False,
    ) as writer:
        writer.write(_batch("frozen_origin", 0.5))
    manifest = load_retrieval_ledger_manifest(manifest_path)
    assert manifest["topk_detail_retained"] is False
    assert "origin_top_k_gallery_ids" in manifest["omitted_columns"]
    with pytest.raises(ValueError, match="missing columns"):
        list(
            iter_retrieval_source_batches(
                manifest_path,
                columns=["query_id", "origin_top_k_gallery_ids"],
            )
        )


def test_retrieval_ledger_reader_rejects_artifact_hash_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "retrieval_ledger" / "manifest.json"
    with RetrievalLedgerWriter(manifest_path) as writer:
        writer.write(_batch("frozen_origin", 0.5))
    manifest = load_retrieval_ledger_manifest(manifest_path)
    core_entry = manifest["conditions"][0]["core"]
    core_path = manifest_path.parent / core_entry["path"]
    payload = bytearray(core_path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    core_path.write_bytes(payload)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        list(iter_retrieval_source_batches(manifest_path))
