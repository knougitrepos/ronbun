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
        == "verified_readable_to_eof"
    )
    assert balanced["artifacts"]["jpg_archive"]["usable"] is True
    assert balanced["observed_jpg_archive"]["image_count"] == 1_251_430
    assert balanced["observed_jpg_archive"]["identity_count"] == 28_000
    assert balanced["observed_jpg_archive"]["differs_from_recordio_list"] is True
    assert (
        balanced["artifacts"]["recordio_archive"]["integrity"]
        == "verified_readable_to_eof"
    )
    assert balanced["observed_rfw_overlap"]["source_identity_count"] == 3
    assert balanced["observed_rfw_overlap"]["balancedface_image_count"] == 177


def test_experiment_config_uses_rfw_as_frozen_codec_diagnostic_only():
    config = _load(EXPERIMENT_CONFIG)
    balanced = config["datasets"]["balancedface"]
    rfw = config["datasets"]["rfw"]

    assert config["execution"]["execute_stage"] is False
    assert config["execution"]["write_outputs"] is False
    assert balanced["status"] == "deferred_by_step7_scope"
    assert balanced["role"] == "deferred_not_used"
    assert balanced["enabled_for_source_index"] is False
    assert balanced["test_split"] is None
    assert balanced["enabled_for_embedding_extraction"] is False
    assert (
        balanced["jpg_archive_alternative"]["blocked_reason"]
        == "alignment_and_group_coverage_audit_not_implemented"
    )
    assert rfw["role"] == "evaluation_test_only"
    assert rfw["enabled_for_headline_evaluation"] is False
    assert rfw["compressor_fit"] is False
    assert config["compression"]["fit_dataset"] is None
    assert config["compression"]["fit_in_this_workflow"] is False
    assert config["compression"]["application_mode"] == (
        "frozen_codec_transfer_only"
    )
    assert config["compression"]["frozen_codec_sources"] == ["lfw", "survface"]
    assert config["thresholds"]["fit_on_all_rfw_pairs"] is False
    assert config["compression"]["exact_fallback"] is False
    assert "dir_at_fpir" in config["evaluation"]["forbidden_official_metrics"]
    assert "empirical_eer_from_heldout_fold_scores" in config["evaluation"][
        "metrics"
    ]
    assert "eer" not in config["evaluation"][
        "planned_metrics_not_yet_implemented"
    ]


def test_step2_separates_rfw_custom_open_set_from_rfw_official_verification():
    config = _load(STEP2_CONFIG)
    datasets = config["datasets"]

    assert datasets["quantitative"] == ["lfw", "survface", "rfw_custom"]
    assert datasets["conditional_quantitative"] == ["rfw"]
    assert datasets["additional_development_sources"] == []
    assert datasets["rfw_custom"]["official_protocol_claim"] is False
    assert datasets["rfw_custom"]["protocol_adapter"] == (
        "rfw_custom_identity_disjoint_v1"
    )
    assert datasets["rfw"]["status"] == "additional_verification_diagnostic"
    assert datasets["rfw"]["official_open_set_protocol"] is False
    assert datasets["rfw"]["headline_evaluation"] is False
    assert datasets["balancedface"]["status"] == "deferred_by_step7_scope"
