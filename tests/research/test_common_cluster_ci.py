import json

import numpy as np
import pandas as pd
import pytest

from research.evaluation.cluster_bootstrap import TpirClusterAccumulator, cluster_rate_draws
from research.evaluation.retrieval_ledger import RetrievalLedgerWriter
from research.runtime.hashing import sha256_file
from scripts.generate_step4_compact_summaries import summarize_retrieval
from scripts.postprocess_common_ci import postprocess
from test_step4_compact_summary_search_modes import _retrieval_rows


def test_joint_query_weighting_order_and_chunk_invariance():
    ids = ["001"] * 9 + ["1"]
    events = np.array([[1, 1]] * 9 + [[0, 0]])
    draws = cluster_rate_draws(ids, events)
    assert set(np.unique(draws)) == {0, .9, 1}
    np.testing.assert_array_equal(draws[:, 0], draws[:, 1])
    np.testing.assert_array_equal(draws, cluster_rate_draws(ids[::-1], events[::-1]))
    whole, chunks = TpirClusterAccumulator(), TpirClusterAccumulator()
    whole.update(ids, events)
    for i in range(10):
        chunks.update(ids[i:i+1], events[i:i+1])
    assert whole.summary() == chunks.summary()
    assert whole.summary()["compressed_minus_origin_tpir_at_rank_k_identity_cluster95_high"] == 0


@pytest.mark.parametrize("ids,events", [
    (["a", None], [[1, 0], [0, 1]]),
    (["a", " "], [[1, 0], [0, 1]]),
    (["a", "b"], [[1, 2], [0, 1]]),
    (["a", "a"], [[1, 0], [0, 1]]),
    (["a", "b"], [[], []]),
])
def test_invalid_cluster_inputs(ids, events):
    with pytest.raises(ValueError):
        cluster_rate_draws(ids, events)


@pytest.mark.parametrize("resamples", [True, 100.5, 99])
def test_invalid_resamples(resamples):
    with pytest.raises(ValueError):
        cluster_rate_draws(["a", "b"], [[1], [0]], resamples=resamples)


def test_unavailable_intervals_are_not_zero():
    acc = TpirClusterAccumulator()
    assert acc.summary()["tpir_cluster_ci_status"] == "no_mated_queries"
    acc.update(None, [[1, 0]])
    assert acc.summary()["tpir_cluster_ci_status"] == "identity_labels_unavailable"
    assert np.isnan(acc.summary()["origin_tpir_at_rank_k_identity_cluster95_low"])
    acc = TpirClusterAccumulator()
    acc.update(["a"], [[1, 0]])
    assert acc.summary()["tpir_cluster_ci_status"] == "insufficient_identity_clusters"


def _rows():
    base = _retrieval_rows(include_search_schema=True).iloc[:2].copy()
    first = base.assign(query_identity_id=["001", "unknown"], target_fpir=.1)
    second = base.assign(query_identity_id=["1", "unknown"], target_fpir=.1)
    second["query_id"] = ["q3", "q4"]
    second.loc[second.is_mated, "compressed_tpir_at_rank_k"] = False
    return pd.concat([first, second], ignore_index=True).assign(model_uid="model-a")


def test_streamed_csv_and_ledger_ci_match_frame_and_preserve_legacy(tmp_path):
    rows = _rows()
    expected, _ = summarize_retrieval(None, chunksize=1, source_frame=rows)
    legacy, _ = summarize_retrieval(None, chunksize=1, source_frame=rows.drop(columns="query_identity_id"))
    old_columns = [c for c in legacy if "cluster" not in c]
    pd.testing.assert_frame_equal(expected[old_columns], legacy[old_columns], check_exact=True)
    assert expected.tpir_cluster_ci_identity_count.tolist() == [2]
    csv = tmp_path / "rows.csv"
    rows.to_csv(csv, index=False)
    for size in (1, 3, 100):
        actual, _ = summarize_retrieval(csv, chunksize=size)
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    ledger = tmp_path / "ledger" / "manifest.json"
    with RetrievalLedgerWriter(ledger) as writer:
        writer.write(rows)
    actual, _ = summarize_retrieval(ledger, chunksize=1)
    pd.testing.assert_frame_equal(expected, actual, check_exact=True)


def test_postprocess_is_explicit_hash_verified_and_immutable(tmp_path):
    ledger = tmp_path / "ledger" / "manifest.json"
    with RetrievalLedgerWriter(ledger) as writer:
        writer.write(_rows())
    source = json.loads(ledger.read_text())
    args = dict(expected_sha256=sha256_file(ledger),
                condition_id=source["conditions"][0]["condition_id"],
                output_root=tmp_path / "derived", chunksize=1)
    output = postprocess(ledger, **args)
    assert json.loads((output / "manifest.json").read_text())["status"] == "completed"
    assert sha256_file(ledger) == args["expected_sha256"]
    with pytest.raises(FileExistsError):
        postprocess(ledger, **args)
    with pytest.raises(ValueError, match="SHA-256"):
        postprocess(ledger, **{**args, "expected_sha256": "0" * 64})
    with pytest.raises(ValueError, match="condition_id"):
        postprocess(ledger, **{**args, "condition_id": "missing"})
    artifact = ledger.parent / source["conditions"][0]["core"]["path"]
    original = artifact.read_bytes()
    artifact.write_bytes(b"X" + original[1:])
    with pytest.raises(ValueError, match="hash|SHA|sha"):
        postprocess(ledger, **{**args, "output_root": tmp_path / "tampered"})
