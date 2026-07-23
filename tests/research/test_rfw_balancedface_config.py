from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = PROJECT_ROOT / "configs" / "datasets" / "rfw_balancedface.yaml"
EXPERIMENT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "rfw_balancedface_quantization.yaml"
)
STEP2_CONFIG = (
    PROJECT_ROOT / "configs" / "experiments" / "step2_pytorch_gradcam.yaml"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_source_config_freezes_local_integrity_and_alternative_formats():
    config = _load(SOURCE_CONFIG)
    rfw = config["datasets"]["rfw"]
    balanced = config["datasets"]["balancedface"]

    assert rfw["role"] == "evaluation_test_only"
    assert rfw["task"] == "official_10fold_pair_verification"
    assert rfw["official_open_set_protocol"] is False
    assert rfw["compressor_fit_allowed"] is False
    assert (
        rfw["artifacts"]["jpg_archive"]["integrity"]
        == "verified_readable_to_eof"
    )
    assert (
        balanced["artifacts"]["jpg_archive"]["integrity"]
        == "corrupt_truncated"
    )
    assert balanced["artifacts"]["jpg_archive"]["usable"] is False
    assert (
        balanced["artifacts"]["recordio_archive"]["integrity"]
        == "verified_readable_to_eof"
    )
    assert balanced["observed_rfw_overlap"]["source_identity_count"] == 3
    assert balanced["observed_rfw_overlap"]["balancedface_image_count"] == 177


def test_experiment_config_keeps_rfw_conditional_and_balancedface_non_test():
    config = _load(EXPERIMENT_CONFIG)
    balanced = config["datasets"]["balancedface"]
    rfw = config["datasets"]["rfw"]

    assert config["execution"]["execute_stage"] is False
    assert config["execution"]["write_outputs"] is False
    assert balanced["role"] == "development_and_calibration_only"
    assert balanced["test_split"] is None
    assert balanced["enabled_for_embedding_extraction"] is False
    assert rfw["role"] == "evaluation_test_only"
    assert rfw["enabled_for_headline_evaluation"] is False
    assert rfw["compressor_fit"] is False
    assert config["thresholds"]["fit_on_all_rfw_pairs"] is False
    assert config["compression"]["exact_fallback"] is False
    assert "dir_at_fpir" in config["evaluation"]["forbidden_official_metrics"]


def test_step2_does_not_silently_promote_rfw_into_current_quantitative_set():
    config = _load(STEP2_CONFIG)
    datasets = config["datasets"]

    assert datasets["quantitative"] == ["lfw", "survface"]
    assert datasets["conditional_quantitative"] == ["rfw"]
    assert datasets["additional_development_sources"] == ["balancedface"]
    assert datasets["rfw"]["status"] == "blocked_checkpoint_overlap"
    assert datasets["balancedface"]["status"] == "source_index_only"
