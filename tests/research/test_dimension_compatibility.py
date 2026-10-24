"""1024D artifacts keep their source space through compression and replay."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from research.compression import (
    PQCompressor,
    fit_pca_family,
    original_profile,
    origin_profile_name,
    pq_profile_name,
)
from research.database.repository import VectorRepository
from research.experiments.fiqa_threshold_calibration import _frozen_pq_codec
from research.explainability.gradcam.artifacts import (
    read_prepared_population_artifact,
    write_prepared_population_artifact,
)
from research.explainability.gradcam.extraction import PreparedPopulationInputs
from research.explainability.gradcam.templates import build_leave_one_out_identity_templates
from research.runtime.hashing import sha256_file


def _vectors(rows: int, dimension: int) -> np.ndarray:
    values = np.random.default_rng(8972).normal(size=(rows, dimension)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _prepared(dimension: int) -> PreparedPopulationInputs:
    normalized = _vectors(6, dimension)
    sample_ids = np.asarray([f"sample-{i}" for i in range(6)])
    identity_ids = np.asarray(["alpha", "alpha", "beta", "beta", "gamma", "gamma"])
    scope_ids = np.asarray(["development"] * 6)
    templates = build_leave_one_out_identity_templates(
        sample_ids, identity_ids, normalized,
        model_uid="facex-test", scope_ids=scope_ids,
    )
    return PreparedPopulationInputs(
        extraction_uid="extraction-test", dataset_id="synthetic",
        sample_ids=sample_ids, identity_ids=identity_ids, scope_ids=scope_ids,
        raw_embeddings=normalized * 2.0, raw_norms=np.full(6, 2.0),
        normalized_embeddings=normalized, loo_templates=templates,
        model_uid="facex-test", checkpoint_sha256="a" * 64,
        preprocess_hash="b" * 64, origin_embedding_artifact_uid="origin-test",
    )


@pytest.mark.parametrize("dimension", [512, 1024])
def test_prepared_artifact_roundtrip_and_dimension_contract(tmp_path: Path, dimension: int):
    prepared = _prepared(dimension)
    artifact = write_prepared_population_artifact(prepared, tmp_path / "prepared", shard_size=2)
    restored = read_prepared_population_artifact(artifact)
    assert restored.embedding_dim == dimension
    np.testing.assert_array_equal(restored.raw_embeddings, prepared.raw_embeddings)
    np.testing.assert_array_equal(restored.loo_templates.templates, prepared.loo_templates.templates)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["embedding_dimension"] == dimension
    manifest["embedding_dimension"] = dimension // 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="embedding_dimension"):
        read_prepared_population_artifact(artifact)


def test_prepared_inputs_reject_mixed_template_dimensions():
    prepared = _prepared(1024)
    templates = replace(prepared.loo_templates, templates=prepared.loo_templates.templates[:, :512])
    with pytest.raises(ValueError, match="template and embedding dimensions"):
        replace(prepared, loo_templates=templates)


def test_origin_and_pca_metadata_preserve_1024d_source_space():
    vectors = _vectors(40, 1024)
    origin = original_profile(vectors)
    assert origin.profile_name == "origin_1024"
    assert origin.pgvector_searchable
    assert origin.metadata["source_dim"] == 1024
    family = fit_pca_family(vectors, dimensions=(32,), source_dim=1024, random_state=8972)
    profile = family["pca_32"].transform_profile(vectors)
    assert profile.metadata["source_profile"] == "origin_1024"
    assert profile.metadata["source_dim"] == 1024
    assert profile.reconstructed_vectors.shape == (40, 1024)
    with pytest.raises(ValueError, match="requires original 1024D"):
        fit_pca_family(vectors[:, :512], dimensions=(32,), source_dim=1024)


@pytest.mark.parametrize("dimension", [512, 1024])
def test_frozen_pq_roundtrip_replays_actual_source_dimension(tmp_path: Path, dimension: int):
    faiss = pytest.importorskip("faiss")
    old_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        vectors = _vectors(300, dimension)
        compressor = PQCompressor(
            source_dim=dimension, m=32, nbits=8,
            source_profile=origin_profile_name(dimension), random_state=8972,
        ).fit(vectors)
        profile = compressor.transform_profile(vectors)
        expected_profile = pq_profile_name(32, 8, source_dim=dimension)
        assert profile.profile_name == expected_profile
        assert profile.metadata["source_profile"] == f"origin_{dimension}"
        assert profile.metadata["family_profile"] == f"pq_origin_{dimension}"
        assert profile.metadata["code_bytes"] == 32
        assert profile.metadata["codebook_bytes"] == dimension * 256 * 4
        workflow = tmp_path / "workflow"
        workflow.mkdir()
        codec_path = workflow / "codec.faiss"
        compressor.save(codec_path)
        manifest = {
            "artifact_type": "frozen_compression_codec_bundle",
            "codecs": [{
                "family": "pq", "profile_name": expected_profile,
                "artifact": "workflow/codec.faiss", "artifact_sha256": sha256_file(codec_path),
                "fit_count": len(vectors),
            }],
        }
        manifest_path = workflow / "frozen_codec_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        restored, _, _ = _frozen_pq_codec(tmp_path, workflow, compression_profile=expected_profile)
        assert restored.source_dim == dimension
        assert restored.source_profile == f"origin_{dimension}"
        codes = restored.encode(vectors[10:20])
        distances, indices = restored.search_adc(vectors[:3], codes, top_k=3)
        decoded = restored.decode(codes)
        reference = ((vectors[:3, None, :] - decoded[None, :, :]) ** 2).sum(axis=2)
        np.testing.assert_array_equal(indices, np.argsort(reference, axis=1)[:, :3])
        np.testing.assert_allclose(distances, np.take_along_axis(reference, indices, axis=1), atol=2e-6)
        wrong_profile = pq_profile_name(32, 8, source_dim=dimension * 2)
        manifest["codecs"][0]["profile_name"] = wrong_profile
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="dimensions/settings differ"):
            _frozen_pq_codec(tmp_path, workflow, compression_profile=wrong_profile)
    finally:
        faiss.omp_set_num_threads(old_threads)


def test_legacy_db_rejects_1024d_before_accessing_database(monkeypatch):
    repository = VectorRepository(None)
    with pytest.raises(ValueError, match="1024D FaceX-Zoo"):
        repository.add_embedding_512(1, "origin_1024", {}, np.ones(1024))
    with pytest.raises(ValueError, match="1024D FaceX-Zoo"):
        repository.upsert_template_512(embedding=np.ones(1024))

    monkeypatch.setattr(repository, "_embedding_for_run", lambda *args: None)
    monkeypatch.setattr(repository, "_persist", lambda record: record)
    record = repository.add_embedding_512(1, "origin_512", {}, np.ones(512))
    assert len(record.embedding) == 512
    monkeypatch.setattr(repository, "_upsert_template", lambda model, **values: values)
    assert len(repository.upsert_template_512(embedding=np.ones(512))["embedding"]) == 512
