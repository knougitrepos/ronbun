from contextlib import contextmanager

import numpy as np
import pandas as pd

from research.experiments.lfw_certification import LFWCertificationInputs
from research.experiments.lfw_pgvector import (
    LFWTemplateScope,
    calibrate_lfw_pgvector_threshold,
    materialize_lfw_templates,
    run_lfw_pgvector_search,
)


def _templates():
    return pd.DataFrame.from_records(
        [
            {
                "identity_id": "a",
                "retrieval_embedding": np.array([1.0, 0.0], dtype=np.float32),
                "embedding": np.array([0.999, 0.01], dtype=np.float32),
                "fallback_embedding": np.array([1.0, 0.0], dtype=np.float32),
                "source_image_ids": ["ga"],
                "enrollment_count": 1,
                "quality": 0.0,
                "variance": 0.0,
                "angular_error": 0.02,
                "reconstruction_error_norm": 0.1,
            },
            {
                "identity_id": "b",
                "retrieval_embedding": np.array([0.0, 1.0], dtype=np.float32),
                "embedding": np.array([0.01, 0.999], dtype=np.float32),
                "fallback_embedding": np.array([0.0, 1.0], dtype=np.float32),
                "source_image_ids": ["gb"],
                "enrollment_count": 1,
                "quality": 0.0,
                "variance": 0.0,
                "angular_error": 0.02,
                "reconstruction_error_norm": 0.1,
            },
            {
                "identity_id": "c",
                "retrieval_embedding": np.array([-1.0, 0.0], dtype=np.float32),
                "embedding": np.array([-0.999, 0.01], dtype=np.float32),
                "fallback_embedding": np.array([-1.0, 0.0], dtype=np.float32),
                "source_image_ids": ["gc"],
                "enrollment_count": 1,
                "quality": 0.0,
                "variance": 0.0,
                "angular_error": 0.02,
                "reconstruction_error_norm": 0.1,
            },
        ]
    )


def _probes():
    return pd.DataFrame.from_records(
        [
            {
                "image_id": "qa",
                "identity_id": "a",
                "probe_type": "registered",
                "retrieval_embedding": np.array([1.0, 0.0], dtype=np.float32),
                "fallback_embedding": np.array([1.0, 0.0], dtype=np.float32),
                "quality": 0.0,
                "reconstruction_error_norm": 0.1,
            },
            {
                "image_id": "qu",
                "identity_id": "u",
                "probe_type": "known_unknown",
                "retrieval_embedding": np.array([0.0, -1.0], dtype=np.float32),
                "fallback_embedding": np.array([0.0, -1.0], dtype=np.float32),
                "quality": 0.0,
                "reconstruction_error_norm": 0.1,
            },
        ]
    )


class _FakeRepository:
    template_calls = []

    def __init__(self, session):
        self.session = session

    def upsert_template_512(self, **values):
        self.template_calls.append((512, values))
        return object(), "inserted"

    def upsert_template_256(self, **values):
        self.template_calls.append((256, values))
        return object(), "inserted"

    def upsert_pca_template(self, dimension, **values):
        self.template_calls.append((int(dimension), values))
        return object(), "inserted"

    def find_similar_templates_512(self, query, **kwargs):
        return [
            self._result("a", 0.95, kwargs),
            self._result("b", 0.10, kwargs),
        ]

    def find_similar_templates_256(self, query, **kwargs):
        query = np.asarray(query)
        if query[1] < 0:
            return [
                self._result("b", 0.30, kwargs),
                self._result("c", 0.20, kwargs),
            ]
        return [
            self._result("a", 0.90, kwargs),
            self._result("b", 0.20, kwargs),
        ]

    def find_similar_pca_templates(self, dimension, query, **kwargs):
        assert len(np.asarray(query)) == int(dimension) or int(dimension) == 256
        return self.find_similar_templates_256(query, **kwargs)

    @staticmethod
    def _result(identity, similarity, kwargs):
        return {
            "identity_id": identity,
            "similarity": similarity,
            "quality": 0.0,
            "variance": 0.0,
            "enrollment_count": 1,
            "query_elapsed_ms": 0.5,
            "search_mode": kwargs["search_mode"],
        }


@contextmanager
def _fake_session_scope(engine):
    yield object()


def _patch_database(monkeypatch):
    import research.experiments.lfw_pgvector as module

    _FakeRepository.template_calls = []
    monkeypatch.setattr(module, "session_scope", _fake_session_scope)
    monkeypatch.setattr(module, "VectorRepository", _FakeRepository)
    monkeypatch.setattr(module, "ensure_vector_indexes", lambda engine: None)


def _scope():
    return LFWTemplateScope(
        run_uid="run-1",
        protocol_name="lfw-test",
        model_uid="pca-model-1",
    )


def test_materialize_lfw_templates_writes_paired_512_and_256_rows(monkeypatch):
    _patch_database(monkeypatch)
    bundle = LFWCertificationInputs(
        probes=_probes(),
        templates=_templates(),
        coverage={},
        certificate_space="pca_reconstructed_512",
    )

    summary = materialize_lfw_templates(object(), bundle=bundle, scope=_scope())

    assert summary["template_count"] == 3
    assert summary["actions"]["template_512_inserted"] == 3
    assert summary["actions"]["template_256_inserted"] == 3
    assert len(_FakeRepository.template_calls) == 6


def test_calibration_uses_pgvector_exact_scores_and_target_fpir(monkeypatch):
    _patch_database(monkeypatch)

    threshold, features, summary = calibrate_lfw_pgvector_threshold(
        object(),
        probes=_probes(),
        scope=_scope(),
        target_fpir=0.0,
        compression_profile="pca_256",
    )

    assert threshold == 0.9
    assert features["accepted"].tolist() == [True, False]
    assert summary["achieved_fpir"] == 0.0
    assert summary["achieved_dir_rank1"] == 1.0


def test_pgvector_search_separates_candidates_certificate_and_exact_baseline(monkeypatch):
    _patch_database(monkeypatch)
    probe = _probes().iloc[[0]].reset_index(drop=True)

    features, summary = run_lfw_pgvector_search(
        object(),
        probes=probe,
        templates=_templates(),
        scope=_scope(),
        origin_threshold=0.5,
        pca_threshold=0.95,
        candidate_k=2,
        ef_search=20,
    )

    row = features.iloc[0]
    assert row["certified_query_angular_error"] == 0.0
    assert bool(row["candidate_contains_origin_top1"])
    assert bool(row["candidate_contains_true_identity"])
    assert row["pca_hnsw_top1_identity"] == "a"
    assert row["origin_exact_top1_identity"] == "a"
    assert row["origin_calibrated_threshold"] == 0.5
    assert row["pca_calibrated_threshold"] == 0.95
    assert bool(row["threshold_crossing"])
    assert summary["candidate_contains_origin_top1_rate"] == 1.0
    assert summary["candidate_contains_true_identity_rate_registered"] == 1.0
    assert summary["registered_compressed_rank_inversion_rate"] == 0.0
    assert summary["registered_identity_loss_rate"] == 0.0
    assert summary["registered_identity_gain_rate"] == 0.0
    assert summary["candidate_miss_caused_by_compression_rate"] == 0.0
    assert summary["candidate_miss_caused_by_hnsw_rate"] == 0.0
    assert summary["origin_open_set"]["dir_rank1"] == 1.0
    assert summary["compressed_open_set"]["dir_rank1"] == 0.0
    assert summary["final_open_set"]["dir_rank1"] == 1.0
    assert summary["certification"]["exact_fallback_rate"] == summary["certification"]["defer_rate"]
    assert summary["by_probe_type"]["registered"]["exact_fallback_rate"] == (
        summary["certification"]["by_probe_type"]["registered"]["exact_fallback_rate"]
    )
