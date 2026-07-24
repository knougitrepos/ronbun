from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "step2_pytorch_gradcam.yaml"


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
    assert config["aligned_crops"]["source_color_order"] == "rgb"
    assert config["aligned_crops"]["dtype"] == "uint8"
    assert config["aligned_crops"]["layout"] == "nhwc"

    models = config["models"]
    assert models["selected"] == ["arcface", "adaface", "magface"]
    assert models["allow_unverified_metadata"] is False
    for name in models["selected"]:
        candidate = models["candidates"][name]
        assert candidate["status"] == "implementation_ready_checkpoint_required"
        assert candidate["framework"] == "pytorch"
        assert candidate["embedding_dim"] == 512
        assert candidate["checkpoint_path"] is None
        assert candidate["loader_factory"].startswith(
            "research.embeddings.pytorch.official_loaders:"
        )
        assert candidate["target_layer"]
        assert candidate["implementation_repository"].startswith("https://github.com/")
        assert candidate["checkpoint_source_page"].startswith("https://github.com/")
        assert None not in candidate["preprocessing"].values()

    assert models["candidates"]["arcface"]["preprocessing"] == {
        "input_size": [112, 112],
        "model_color_order": "rgb",
        "input_range": [-1.0, 1.0],
        "mean": [127.5, 127.5, 127.5],
        "std": [127.5, 127.5, 127.5],
    }
    assert models["candidates"]["adaface"]["preprocessing"][
        "model_color_order"
    ] == "bgr"
    assert models["candidates"]["magface"]["preprocessing"] == {
        "input_size": [112, 112],
        "model_color_order": "bgr",
        "input_range": [0.0, 1.0],
        "mean": [0.0, 0.0, 0.0],
        "std": [255.0, 255.0, 255.0],
    }


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


def test_step2_gradcam_extracts_origin_population_before_compression() -> None:
    config = _load_config()
    gradcam = config["gradcam"]

    assert config["datasets"]["quantitative"] == ["lfw", "survface"]
    assert config["datasets"]["saliency_population_initial"] == ["lfw"]
    assert gradcam["study_enabled"] is True
    assert gradcam["execution_ready"] is False
    assert gradcam["role"] == "origin_population_feature"
    assert gradcam["stage_order"] == [
        "pass_a_origin_embedding",
        "leave_one_out_identity_template",
        "pass_b_population_gradcam",
        "origin_embedding_compression",
        "strict_saliency_compression_join",
        "representative_case_visualization",
    ]

    population = gradcam["population"]
    assert population["pass_a_origin_embedding_coverage"] == "all_selected_samples"
    assert population["pass_b_saliency_coverage"] == "all_target_eligible_samples"
    assert population["retain_all_selected_sample_rows"] is True
    assert population["singleton_identity_policy"] == "record_ineligible"
    assert population["unlabeled_sample_policy"] == "record_ineligible"
    assert population["ineligible_rows_must_include_reason"] is True

    two_pass = gradcam["two_pass_extraction"]
    assert two_pass["pass_a"]["output"] == "origin_embedding_512"
    assert two_pass["pass_a"]["backward"] is False
    assert two_pass["template_build"]["leave_query_out"] is True
    assert two_pass["template_build"]["minimum_same_identity_samples"] == 2
    assert two_pass["pass_b"]["coverage"] == "all_target_eligible_samples"
    assert two_pass["pass_b"]["require_pass_a_embedding_match"] is True

    target = gradcam["target"]
    assert target["name"] == "origin_leave_one_out_identity_cosine"
    assert target["embedding_space"] == "origin_512"
    assert target["reference_branch_detached"] is True
    assert gradcam["target"]["query_branch_only"] is True
    assert gradcam["target"]["differentiate_hard_pq"] is False
    assert target["leave_query_out"] is True
    assert target["template_scope_keys"] == [
        "dataset_id",
        "split",
        "identity_id",
        "model_uid",
        "selected_manifest_sha256",
    ]


def test_step2_gradcam_persists_bounded_shards_and_defers_cases() -> None:
    gradcam = _load_config()["gradcam"]
    persistence = gradcam["persistence"]

    assert gradcam["extraction"]["shard_size"] > 0
    assert persistence["format"] == "immutable_shards"
    assert persistence["heatmap_resolution"] == "native_target_layer"
    assert persistence["persist_normalized_heatmap"] is True
    assert persistence["persist_scalar_features"] is True
    assert persistence["persist_full_activations"] is False
    assert persistence["persist_full_gradients"] is False
    assert (
        persistence["full_activation_gradient_policy"]
        == "transient_or_explicit_debug_subset_only"
    )
    assert gradcam["regions"]["require_explicit_landmark_or_face_masks"] is True
    assert gradcam["regions"]["infer_missing_semantic_masks"] is False
    assert gradcam["faithfulness"]["coverage"] == "all_target_eligible_samples"
    assert gradcam["faithfulness"]["random_seed_unit"] == "sample_id"

    representative = gradcam["representative_case_visualization"]
    assert representative["role"] == "visualization_only"
    assert representative["run_after_population_join"] is True
    assert representative["regenerate_gradcam"] is False


def test_step2_joint_analysis_requires_strict_keys_and_origin_lineage() -> None:
    analysis = _load_config()["joint_analysis"]

    assert analysis["join_stage"] == "after_origin_embedding_compression"
    assert analysis["strict_join"] is True
    assert analysis["join_keys"] == [
        "extraction_uid",
        "dataset_id",
        "sample_id",
        "model_uid",
    ]
    assert analysis["lineage"]["required_key"] == "origin_embedding_artifact_uid"
    assert analysis["lineage"]["compression_source_must_match_saliency_origin"] is True
    assert analysis["lineage"]["reject_missing_or_mismatched_lineage"] is True
    assert analysis["prohibit_saliency_embedding_concatenation"] is True

    association = analysis["association"]
    assert association["stratify_by"] == [
        "dataset_id",
        "model_uid",
        "compression_family",
        "compression_profile",
    ]
    assert association["bootstrap_unit"] == "identity_id"
    assert association["bootstrap_method"] == "identity_cluster"
    assert association["pool_models"] is False
    assert association["pool_compression_profiles"] is False
