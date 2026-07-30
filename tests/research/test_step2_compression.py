from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.experiments import step2_compression as module
from research.explainability.gradcam.extraction import PreparedPopulationInputs
from research.explainability.gradcam.templates import (
    build_leave_one_out_identity_templates,
)


class _FakePQ:
    def __init__(
        self,
        source_dim: int,
        m: int,
        nbits: int,
        *,
        source_profile: str,
    ):
        self.m = m
        self.nbits = nbits

    def fit(self, development_vectors: np.ndarray):
        return self

    def transform_profile(self, vectors: np.ndarray):
        restored = np.asarray(vectors, dtype=np.float32).copy()
        return SimpleNamespace(
            vectors=restored,
            reconstructed_vectors=restored,
            metadata={
                "code_bytes": self.m * self.nbits,
                "codebook_bytes": 128,
                "codebook_bytes_source": "test",
            },
        )


class _FakePCA:
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return np.asarray(vectors, dtype=np.float32)[:, :2]

    def transform_profile(self, vectors: np.ndarray):
        source = np.asarray(vectors, dtype=np.float32)
        return SimpleNamespace(
            vectors=source[:, :2],
            reconstructed_vectors=source.copy(),
            metadata={"storage_bytes_per_vector": 8},
        )


def _prepared() -> tuple[PreparedPopulationInputs, pd.DataFrame]:
    rows = [
        ("d1", "dev-1", "development", 0.0),
        ("d2", "dev-2", "development", 0.5),
        ("d3", "dev-3", "development", 1.0),
        ("d4", "dev-4", "development", 1.5),
        ("ca1", "cal-a", "calibration", 0.2),
        ("ca2", "cal-a", "calibration", 0.2),
        ("cb1", "cal-b", "calibration", 2.2),
        ("tg1", "test-gallery", "test", 0.4),
        ("tg2", "test-gallery", "test", 0.4),
        ("tk1", "test-known-unknown", "test", 1.8),
        ("tu1", "test-unknown-unknown", "test", 2.6),
    ]
    vectors = np.zeros((len(rows), 512), dtype=np.float32)
    for index, (_, _, _, angle) in enumerate(rows):
        vectors[index, 0] = np.cos(angle)
        vectors[index, 1] = np.sin(angle)
    sample_ids = np.asarray([row[0] for row in rows])
    identity_ids = np.asarray([row[1] for row in rows])
    splits = np.asarray([row[2] for row in rows])
    loo = build_leave_one_out_identity_templates(
        sample_ids,
        identity_ids,
        vectors,
        model_uid="arcface-test",
        scope_ids=splits,
    )
    raw = vectors * 2.0
    prepared = PreparedPopulationInputs(
        extraction_uid="extract-test",
        dataset_id="lfw",
        sample_ids=sample_ids,
        identity_ids=identity_ids,
        scope_ids=splits,
        raw_embeddings=raw,
        raw_norms=np.linalg.norm(raw, axis=1),
        normalized_embeddings=vectors,
        loo_templates=loo,
        model_uid="arcface-test",
        checkpoint_sha256="a" * 64,
        preprocess_hash="b" * 64,
        origin_embedding_artifact_uid="origin-test",
    )
    selected = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "identity_id": identity_ids,
            "split": splits,
            "image_path": [f"{sample}.jpg" for sample in sample_ids],
        }
    )
    return prepared, selected


def test_step2_runner_fits_calibrates_and_evaluates_one_lineage(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "PQCompressor", _FakePQ)
    monkeypatch.setattr(
        module,
        "fit_pca_family",
        lambda development_vectors, dimensions, random_state: {
            "pca_2": _FakePCA()
        },
    )
    prepared, selected = _prepared()

    result = module.characterize_step2_compression(
        prepared,
        selected,
        gallery_identities=["test-gallery"],
        unknown_unknown_identities=["test-unknown-unknown"],
        pca_dimensions=[2],
        pq_settings=[(8, 1)],
        seed=42,
        target_fpir=1.0,
        enrollment_count=1,
        calibration_gallery_identities=1,
        top_k=1,
    )

    assert set(result.paired_metrics["compression_family"]) == {"pca", "pq"}
    assert set(result.retrieval_metrics["threshold_policy"]) == {
        "frozen_origin",
        "recalibrated_compressed",
    }
    assert not result.paired_metrics["origin_fallback_used"].any()
    assert not result.retrieval_metrics["origin_fallback_used"].any()
    assert set(result.retrieval_metrics["evaluation_split"]) == {"test"}
    assert set(result.retrieval_metrics["threshold_source_split"]) == {
        "calibration"
    }
    assert len(result.summary) == 4


def test_step2_runner_rejects_manifest_order_mismatch() -> None:
    prepared, selected = _prepared()
    selected = selected.iloc[::-1].reset_index(drop=True)

    try:
        module.characterize_step2_compression(
            prepared,
            selected,
            gallery_identities=["test-gallery"],
            unknown_unknown_identities=["test-unknown-unknown"],
            pca_dimensions=[2],
            pq_settings=[(8, 1)],
        )
    except ValueError as exc:
        assert "row order" in str(exc)
    else:
        raise AssertionError("manifest order mismatch was not rejected")


def test_survface_runner_preserves_official_protocol_and_training_boundary(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "PQCompressor", _FakePQ)
    monkeypatch.setattr(
        module,
        "fit_pca_family",
        lambda development_vectors, dimensions, random_state: {
            "pca_2": _FakePCA()
        },
    )
    rows = [
        ("d1", "dev-1", "development", 0.0, None, None),
        ("d2", "dev-2", "development", 0.5, None, None),
        ("d3", "dev-3", "development", 1.0, None, None),
        ("d4", "dev-4", "development", 1.5, None, None),
        ("ca1", "cal-a", "calibration", 0.2, None, None),
        ("ca2", "cal-a", "calibration", 0.2, None, None),
        ("cb1", "cal-b", "calibration", 2.2, None, None),
        ("cb2", "cal-b", "calibration", 2.2, None, None),
        ("g1", "official-a", "test", 0.4, "gallery", 5),
        ("g2", "official-a", "test", 0.4, "gallery", 9),
        ("p1", "official-a", "test", 0.4, "registered_probe", 7),
        (
            "u1",
            "official-unknown",
            "test",
            2.6,
            "unknown_unknown_probe",
            11,
        ),
    ]
    vectors = np.zeros((len(rows), 512), dtype=np.float32)
    for index, row in enumerate(rows):
        vectors[index, 0] = np.cos(row[3])
        vectors[index, 1] = np.sin(row[3])
    sample_ids = np.asarray([row[0] for row in rows])
    identity_ids = np.asarray([row[1] for row in rows])
    splits = np.asarray([row[2] for row in rows])
    loo = build_leave_one_out_identity_templates(
        sample_ids,
        identity_ids,
        vectors,
        model_uid="arcface-test",
        scope_ids=splits,
    )
    prepared = PreparedPopulationInputs(
        extraction_uid="extract-survface-test",
        dataset_id="survface",
        sample_ids=sample_ids,
        identity_ids=identity_ids,
        scope_ids=splits,
        raw_embeddings=vectors * 2.0,
        raw_norms=np.linalg.norm(vectors * 2.0, axis=1),
        normalized_embeddings=vectors,
        loo_templates=loo,
        model_uid="arcface-test",
        checkpoint_sha256="a" * 64,
        preprocess_hash="b" * 64,
        origin_embedding_artifact_uid="origin-survface-test",
    )
    selected = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "identity_id": identity_ids,
            "split": splits,
            "image_path": [f"{value}.jpg" for value in sample_ids],
            "protocol_role": [row[4] for row in rows],
            "probe_type": [
                (
                    "registered"
                    if row[4] == "registered_probe"
                    else "unknown_unknown"
                    if row[4] == "unknown_unknown_probe"
                    else "not_applicable"
                    if row[4] == "gallery"
                    else None
                )
                for row in rows
            ],
            "protocol_index": [row[5] for row in rows],
        }
    )

    result = module.characterize_step2_survface_compression(
        prepared,
        selected,
        pca_dimensions=[2],
        pq_settings=[(8, 1)],
        target_fpir=1.0,
        calibration_gallery_identities=1,
        top_k=1,
    )

    assert set(result.retrieval_metrics["protocol_uid"]) == {
        "qmul-survface-v1-official-open-set-identification"
    }
    assert set(result.retrieval_metrics["query_id"]) == {"p1", "u1"}
    assert set(result.retrieval_metrics["threshold_policy"]) == {
        "frozen_origin",
        "recalibrated_compressed",
    }
