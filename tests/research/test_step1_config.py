from pathlib import Path

import yaml

from research.compression import ORIGIN_512, PCA_SWEEP_DIMENSIONS
from research.experiments.scope import ExperimentScope


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "step1_embedding_compression.yaml"


def test_step1_config_separates_pca_and_direct_origin_pq_families():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    compression = config["compression"]
    pca = compression["families"]["pca"]
    pq = compression["families"]["pq"]

    assert compression["baseline_profile"] == ORIGIN_512
    assert tuple(pca["dimensions"]) == PCA_SWEEP_DIMENSIONS
    assert pca["source_profile"] == ORIGIN_512
    assert pca["source_dimension"] == 512
    assert pq["source_profile"] == ORIGIN_512
    assert pq["source_dimension"] == 512
    assert "pca_pq" not in compression["families"]
    assert all("input_profile" not in setting for setting in pq["settings"])
    settings = [(setting["m"], setting["nbits"]) for setting in pq["settings"]]
    assert settings == [(8, 8), (16, 8), (32, 8), (64, 8), (128, 8)]
    assert [m * nbits // 8 for m, nbits in settings] == [8, 16, 32, 64, 128]
    assert all(512 % m == 0 for m, _ in settings)


def test_step1_config_has_explicit_scope_and_disables_old_fallback_path():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    scope = ExperimentScope.from_config(config)

    assert scope.mode == "real"
    assert scope.data_fraction == 1.0
    assert config["evaluation"]["full_precision_baseline"] is True
    assert config["evaluation"]["exact_fallback"] is False
    assert config["evaluation"]["certification"] is False
    assert config["models"]["enabled"] == ["arcface"]
    assert set(config["models"]["planned"]) == {"adaface", "magface"}
    assert set(config["models"]["enabled"]).isdisjoint(config["models"]["planned"])
    assert {"lfw", "survface"}.issubset(config["datasets"])
    survface = config["datasets"]["survface"]
    assert survface["training_manifest_path"].endswith("training_manifest.csv")
    assert survface["fit_source"] == "training_3000_watchlist_enrollment_only"
    assert survface["official_manifest_role"] == "evaluation_only"
