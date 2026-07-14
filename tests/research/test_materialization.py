import pytest

import research.experiments.materialization as materialization
from research.compression import PCA_256, PQ_AUXILIARY


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

