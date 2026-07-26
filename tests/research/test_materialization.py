import pytest

import research.experiments.materialization as materialization
from research.compression import PCA_256, PQ_AUXILIARY, PCACompressor, pca_profile_name


def _frozen_stats():
    return {
        PCA_256: {"mean": 0.1, "std": 0.02, "fit_count": 100},
        PQ_AUXILIARY: {"mean": 0.2, "std": 0.03, "fit_count": 100},
    }


def test_transfer_materialization_passes_only_frozen_stats_to_core(monkeypatch, tmp_path):
    captured = {}

    def fake_core(engine, **kwargs):
        captured.update(kwargs)
        return {"error_normalization": kwargs["frozen_error_normalization"]}

    monkeypatch.setattr(materialization, "_materialize_compressed_embeddings", fake_core)

    result = materialization.materialize_compressed_embeddings_with_frozen_stats(
        object(),
        run_uid="survface-run",
        pca=object(),
        pq=object(),
        pca_artifact_path="pca.joblib",
        pca_artifact_sha256="a" * 64,
        pq_artifact_path="pq.faiss",
        pq_artifact_sha256="b" * 64,
        error_normalization=_frozen_stats(),
        measurements_path=tmp_path / "measurements.csv",
    )

    assert captured["development_image_paths"] is None
    assert captured["frozen_error_normalization"] == _frozen_stats()
    assert result["error_normalization"] == _frozen_stats()


@pytest.mark.parametrize(
    "stats",
    [
        {PCA_256: {"mean": 0.0, "std": 1.0, "fit_count": 10}},
        {
            PCA_256: {"mean": 0.0, "std": -1.0, "fit_count": 10},
            PQ_AUXILIARY: {"mean": 0.0, "std": 1.0, "fit_count": 10},
        },
        {
            PCA_256: {"mean": 0.0, "std": 1.0, "fit_count": 0},
            PQ_AUXILIARY: {"mean": 0.0, "std": 1.0, "fit_count": 10},
        },
    ],
)
def test_frozen_error_normalization_requires_development_provenance(stats):
    with pytest.raises(ValueError, match="frozen error normalization"):
        materialization._validate_error_normalization(stats)


def test_db_materialization_accepts_all_step1_pca_dimensions(monkeypatch, tmp_path):
    monkeypatch.setattr(materialization, "ensure_database_schema", lambda _engine: None)

    def stop_after_dimension_validation(*_args, **_kwargs):
        raise RuntimeError("dimension validation passed")

    monkeypatch.setattr(materialization, "_source_batches", stop_after_dimension_validation)
    pcas = {
        f"pca_{dimension}": PCACompressor(dimension)
        for dimension in (384, 256, 128, 64, 32)
    }
    with pytest.raises(RuntimeError, match="dimension validation passed"):
        materialization.materialize_pca_sweep_embeddings(
            object(),
            run_uid="step1",
            pcas=pcas,
            pca_artifact_paths={
                profile: f"{profile}.joblib" for profile in pcas
            },
            pca_artifact_sha256={profile: "a" * 64 for profile in pcas},
            development_image_paths={"image.jpg"},
            measurements_path=tmp_path / "measurements.csv",
        )


def test_db_materialization_keeps_legacy_pca_448_profile_resolvable():
    assert pca_profile_name(448, allow_legacy=True) == "pca_448"
