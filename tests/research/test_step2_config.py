from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "experiments" / "step2_pytorch_gradcam.yaml"
)


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_step2_config_is_checkpoint_level_and_fails_closed() -> None:
    config = _load_config()

    assert config["study_boundary"]["comparison_unit"] == "pretrained_checkpoint"
    assert config["study_boundary"]["causal_loss_claim"] is False
    assert config["study_boundary"]["train_face_recognition_models"] is False
    assert config["study_boundary"]["exact_origin_fallback"] is False
    assert config["execution"]["execute_stage"] is False
    assert config["execution"]["write_outputs"] is False

    models = config["models"]
    assert models["selected"] == ["arcface", "adaface", "magface"]
    assert models["allow_unverified_metadata"] is False
    for name in models["selected"]:
        candidate = models["candidates"][name]
        assert candidate["status"] == "checkpoint_required"
        assert candidate["framework"] == "pytorch"
        assert candidate["embedding_dim"] == 512
        assert candidate["checkpoint_path"] is None
        assert candidate["loader_factory"] is None
        assert candidate["target_layer"] is None
        assert None in candidate["preprocessing"].values()


def test_step2_config_preserves_independent_pca_and_pq_families() -> None:
    config = _load_config()
    compression = config["compression"]
    pca = compression["families"]["pca"]
    pq = compression["families"]["pq"]

    assert compression["baseline_profile"] == "origin_512"
    assert pca["source_profile"] == "origin_512"
    assert pca["dimensions"] == [384, 256, 128, 64, 32]
    assert pq["source_profile"] == "origin_512"
    assert pq["source_dimension"] == 512
    assert "pca_pq" not in compression["families"]
    assert config["evaluation"]["exact_fallback"] is False


def test_step2_gradcam_is_separate_and_does_not_differentiate_pq() -> None:
    config = _load_config()
    gradcam = config["gradcam"]

    assert config["datasets"]["quantitative"] == ["lfw", "survface"]
    assert config["datasets"]["gradcam_initial"] == ["lfw"]
    assert gradcam["enabled"] is False
    assert gradcam["target"]["name"] == "origin_pair_cosine"
    assert gradcam["target"]["gallery_branch_detached"] is True
    assert gradcam["target"]["query_branch_only"] is True
    assert gradcam["target"]["differentiate_hard_pq"] is False
    assert gradcam["case_selection"]["require_paired_probe_profile_rows"] is True
