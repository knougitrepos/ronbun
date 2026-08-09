from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from research.experiments import pipeline_runner
from research.experiments.pipeline_runner import (
    DEFAULT_MODEL_PROFILES,
    DEFAULT_MODEL_WEIGHT_PATHS,
    FULL_DATA_FRACTION,
    QUICK_DATA_FRACTIONS,
    build_common_experiment_plan,
    inspect_common_experiment_plan,
    load_evaluation_contract,
    materialize_effective_step4_config,
    prepare_common_model_checkpoint,
    reuse_completed_run_for_plan,
    run_common_step4_experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "configs" / "experiments" / "evaluation_contract_v1.yaml"


def _write_plan_inputs(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "step4.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "root": "runs",
                    "dataset_date_dir_template": "{dataset_id}_{date}",
                },
                "execution": {
                    "model_profile": "arcface_registered",
                    "mode": "real",
                    "data_fraction": 1.0,
                    "seed": 42,
                    "execute_stage": True,
                    "write_outputs": True,
                    "overwrite": False,
                    "allow_dirty": False,
                },
                "models": {
                    "profiles": {
                        "arcface_registered": {
                            "family": "arcface",
                        }
                    }
                },
                "evaluation": {
                    "reported_target_fpirs": [0.10, 0.01],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    contract_path = tmp_path / "evaluation_contract_v1.yaml"
    contract_path.write_text(
        CONTRACT_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return config_path, contract_path


@pytest.mark.parametrize(
    ("dataset_id", "run_tier", "expected_fraction", "expected_selected"),
    [
        ("lfw", "quick", 0.10, 10),
        ("survface", "quick", 0.02, 2),
        ("rfw_custom", "quick", 0.10, 10),
        ("lfw", "full", 1.00, 100),
        ("survface", "full", 1.00, 100),
        ("rfw_custom", "full", 1.00, 100),
    ],
)
def test_common_plan_uses_only_confirmed_quick_full_fractions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_id: str,
    run_tier: str,
    expected_fraction: float,
    expected_selected: int,
) -> None:
    config_path, contract_path = _write_plan_inputs(tmp_path)
    source = pd.DataFrame(
        {
            "image_id": [f"image-{index:03d}" for index in range(100)],
            "identity_id": [f"identity-{index:03d}" for index in range(100)],
            "split": ["test"] * 100,
            "protocol_role": ["registered_probe"] * 100,
        }
    )
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_step4_dataset_spec",
        lambda *args, **kwargs: SimpleNamespace(
            dataset_id=dataset_id,
            manifest_paths=(),
        ),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "load_step4_source_manifest",
        lambda spec: source,
    )

    def select(frame, *, dataset_id, scope):
        assert scope.mode == "real"
        assert scope.data_fraction == expected_fraction
        return frame.iloc[:expected_selected].copy()

    monkeypatch.setattr(
        pipeline_runner,
        "select_step4_source_manifest",
        select,
    )
    plan = build_common_experiment_plan(
        project_root=tmp_path,
        dataset_id=dataset_id,
        run_tier=run_tier,
        step4_config_path=config_path,
        evaluation_contract_path=contract_path,
    )

    assert plan.data_fraction == expected_fraction
    assert plan.quick_data_fractions == {
        "lfw": 0.10,
        "survface": 0.02,
        "rfw_custom": 0.10,
    }
    assert plan.quick_fraction_override is False
    assert plan.selected_source_rows == expected_selected
    assert len(plan.selected_image_ids_sha256) == 64
    assert plan.model_name == "arc"
    assert plan.scope_paper_eligible is (run_tier == "full")
    assert plan.comparison_paper_eligible is False
    assert plan.effective_step4_config["execution"]["mode"] == "real"
    assert (
        plan.effective_step4_config["execution"]["data_fraction"] == expected_fraction
    )
    assert plan.effective_step4_config["execution"]["allow_dirty"] is (
        run_tier == "quick"
    )
    assert plan.effective_step4_config["orchestration"]["run_tier"] == run_tier
    source_snapshot = plan.effective_step4_config["orchestration"][
        "source_snapshot"
    ]
    assert set(source_snapshot) == {
        "commit",
        "branch",
        "dirty",
        "working_tree_diff_sha256",
        "untracked_content_sha256",
    }


def test_contract_constants_match_user_confirmed_values() -> None:
    contract = load_evaluation_contract(CONTRACT_PATH)

    assert dict(QUICK_DATA_FRACTIONS) == {
        "lfw": 0.10,
        "survface": 0.02,
        "rfw_custom": 0.10,
    }
    assert FULL_DATA_FRACTION == 1.0
    assert dict(DEFAULT_MODEL_PROFILES) == {
        "arc": "arcface_ms1mv3_r100",
        "ada": "adaface_ms1mv3_r100",
        "mag": "magface_ms1mv2_iresnet100",
        "edge": "edgeface_webface12m_xs_gamma_06",
    }
    assert dict(DEFAULT_MODEL_WEIGHT_PATHS) == {
        "arc": "models/arcface/ms1mv3_r100_backbone.pth",
        "ada": "models/adaface/adaface_ir101_ms1mv3.ckpt",
        "mag": "models/magface/magface_ms1mv2.pth",
        "edge": "models/edgeface/edgeface_xs_gamma_06.pt",
    }
    assert contract["comparison"]["main_profiles"] == [
        "origin_512",
        "pca_256",
        "pca_128",
        "pq_origin_512_m128_nbits8",
    ]
    assert contract["pq_search"]["primary"] == "exhaustive_adc"
    assert contract["calibration"]["enrollment_identity_counts"] == [
        100,
        500,
        1000,
    ]
    assert contract["calibration"]["paper_operating_points"] == [0.10, 0.01]
    assert contract["calibration"][
        "reuse_search_scores_across_operating_points"
    ] is True
    assert contract["calibration"]["appendix_datasets"] == ["lfw", "survface"]
    assert contract["evaluation"]["confidence_intervals"] == {
        "binomial_rates": "wilson_95",
        "compressed_minus_origin": "paired_query_bootstrap_95",
        "bootstrap_seed": 42,
        "bootstrap_repeats": 2000,
    }
    assert contract["rfw"]["edgeface_training_identity_overlap_status"] == (
        "UNKNOWN"
    )
    assert contract["rfw"]["strict_unseen_identity_evidence_allowed"] is False


def test_model_preparation_rejects_profile_checkpoint_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/experiments/step2_pytorch_gradcam.yaml").read_text(
            encoding="utf-8"
        )
    )
    profile = config["models"]["profiles"]["edgeface_webface12m_xs_gamma_06"]
    profile["expected_checkpoint_sha256"] = "0" * 64
    monkeypatch.setattr(pipeline_runner, "load_step4_config", lambda path: config)
    checkpoint = tmp_path / "edgeface.pt"
    checkpoint.write_bytes(b"wrong-checkpoint")

    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        prepare_common_model_checkpoint(
            project_root=PROJECT_ROOT,
            model_name="edge",
            checkpoint_path=checkpoint,
            run_smoke_validation=False,
        )


def test_contract_rejects_unconfirmed_quick_fraction(tmp_path: Path) -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["execution"]["quick_data_fractions"]["survface"] = 0.05
    path = tmp_path / "invalid.yaml"
    path.write_text(
        yaml.safe_dump(contract, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="user-confirmed contract"):
        load_evaluation_contract(path)


def test_notebook_quick_fraction_override_is_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, contract_path = _write_plan_inputs(tmp_path)
    source = pd.DataFrame(
        {
            "image_id": [f"image-{index}" for index in range(10)],
            "identity_id": [f"identity-{index}" for index in range(10)],
            "split": ["test"] * 10,
        }
    )
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_step4_dataset_spec",
        lambda *args, **kwargs: SimpleNamespace(
            dataset_id="survface",
            manifest_paths=(),
        ),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "load_step4_source_manifest",
        lambda spec: source,
    )
    monkeypatch.setattr(
        pipeline_runner,
        "select_step4_source_manifest",
        lambda frame, **kwargs: frame.iloc[:3].copy(),
    )

    plan = build_common_experiment_plan(
        project_root=tmp_path,
        dataset_id="survface",
        run_tier="quick",
        quick_data_fractions={
            "lfw": 0.25,
            "survface": 0.03,
            "rfw_custom": 0.15,
        },
        step4_config_path=config_path,
        evaluation_contract_path=contract_path,
    )
    full_plan = build_common_experiment_plan(
        project_root=tmp_path,
        dataset_id="survface",
        run_tier="full",
        quick_data_fractions={
            "lfw": 0.25,
            "survface": 0.03,
            "rfw_custom": 0.15,
        },
        step4_config_path=config_path,
        evaluation_contract_path=contract_path,
    )

    assert plan.data_fraction == 0.03
    assert plan.quick_fraction_override is True
    assert plan.effective_step4_config["orchestration"]["quick_data_fractions"] == {
        "lfw": 0.25,
        "survface": 0.03,
        "rfw_custom": 0.15,
    }
    assert (
        plan.effective_step4_config["orchestration"]["quick_fraction_override"]
        is True
    )
    assert full_plan.data_fraction == 1.0
    assert full_plan.effective_step4_config["execution"]["data_fraction"] == 1.0


def test_results_only_plan_disables_duplicate_join_and_raw_cam_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, contract_path = _write_plan_inputs(tmp_path)
    source = pd.DataFrame(
        {
            "image_id": ["image-1"],
            "identity_id": ["identity-1"],
            "split": ["test"],
        }
    )
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_step4_dataset_spec",
        lambda *args, **kwargs: SimpleNamespace(
            dataset_id="lfw",
            manifest_paths=(),
        ),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "load_step4_source_manifest",
        lambda spec: source,
    )
    monkeypatch.setattr(
        pipeline_runner,
        "select_step4_source_manifest",
        lambda frame, **kwargs: frame.copy(),
    )

    plan = build_common_experiment_plan(
        project_root=tmp_path,
        dataset_id="lfw",
        run_tier="quick",
        artifact_storage_mode="results_only",
        step4_config_path=config_path,
        evaluation_contract_path=contract_path,
    )

    config = plan.effective_step4_config
    assert config["workflow"]["artifact_storage_mode"] == "results_only"
    assert config["workflow"]["persist_large_join_artifacts"] is False
    persistence = config["gradcam"]["persistence"]
    assert persistence["persist_normalized_heatmap"] is True
    assert persistence["persist_raw_cam"] is False
    assert persistence["persist_relu_cam"] is False
    assert persistence["persist_channel_weights"] is False
    assert persistence["persist_pass_b_embeddings"] is False


def test_prepare_common_model_checkpoint_uses_alias_profile_and_weight(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "models" / "adaface" / "custom.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic-adaface-checkpoint")
    config_path = tmp_path / "step4.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "aligned_crops": {"source_color_order": "rgb"},
                "models": {
                    "registry_root": "runs/step2/model_registry",
                    "validation_root": "runs/step2/model_validation",
                    "blocked_profiles": [],
                    "profiles": {
                        "adaface_test": {
                            "family": "adaface",
                            "architecture": "ir_101",
                            "training_dataset": "ms1mv3",
                            "implementation_repository": (
                                "https://example.invalid/adaface"
                            ),
                            "checkpoint_source_url": (
                                "https://example.invalid/adaface.ckpt"
                            ),
                            "loader_factory": (
                                "research.embeddings.pytorch."
                                "official_loaders:load_adaface_checkpoint"
                            ),
                            "target_layer": "body.48.res_layer.4",
                            "embedding_dim": 512,
                            "run_gradcam": True,
                            "preprocessing": {
                                "input_size": [112, 112],
                                "model_color_order": "bgr",
                                "mean": [127.5, 127.5, 127.5],
                                "std": [127.5, 127.5, 127.5],
                            },
                        }
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    prepared = prepare_common_model_checkpoint(
        project_root=tmp_path,
        model_name="ada",
        model_profile="adaface_test",
        checkpoint_path=checkpoint,
        step4_config_path=config_path,
        run_smoke_validation=False,
    )
    repeated = prepare_common_model_checkpoint(
        project_root=tmp_path,
        model_name="ada",
        model_profile="adaface_test",
        checkpoint_path=checkpoint,
        step4_config_path=config_path,
        run_smoke_validation=False,
    )

    assert prepared == repeated
    assert prepared.family == "adaface"
    assert prepared.checkpoint_path == checkpoint.resolve()
    assert prepared.model_spec_path.is_file()
    assert prepared.smoke_validation_status == "not_requested"

    with pytest.raises(ValueError, match="requires family"):
        prepare_common_model_checkpoint(
            project_root=tmp_path,
            model_name="arc",
            model_profile="adaface_test",
            checkpoint_path=checkpoint,
            step4_config_path=config_path,
            run_smoke_validation=False,
        )


def test_materialized_effective_config_is_restart_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, contract_path = _write_plan_inputs(tmp_path)
    source = pd.DataFrame(
        {
            "image_id": ["image-1"],
            "identity_id": ["identity-1"],
            "split": ["test"],
        }
    )
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_step4_dataset_spec",
        lambda *args, **kwargs: SimpleNamespace(
            dataset_id="lfw",
            manifest_paths=(),
        ),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "load_step4_source_manifest",
        lambda spec: source,
    )
    monkeypatch.setattr(
        pipeline_runner,
        "select_step4_source_manifest",
        lambda frame, **kwargs: frame.copy(),
    )
    plan = build_common_experiment_plan(
        project_root=tmp_path,
        dataset_id="lfw",
        run_tier="quick",
        step4_config_path=config_path,
        evaluation_contract_path=contract_path,
    )

    first = materialize_effective_step4_config(plan)
    second = materialize_effective_step4_config(plan)

    assert first == second
    assert yaml.safe_load(first.read_text(encoding="utf-8")) == (
        plan.effective_step4_config
    )


def test_preflight_reports_partial_evaluation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, contract_path = _write_plan_inputs(tmp_path)
    source = pd.DataFrame(
        {
            "image_id": ["image-1"],
            "identity_id": ["identity-1"],
            "split": ["test"],
        }
    )
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_step4_dataset_spec",
        lambda *args, **kwargs: SimpleNamespace(
            dataset_id="lfw",
            manifest_paths=(),
        ),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "load_step4_source_manifest",
        lambda spec: source,
    )
    monkeypatch.setattr(
        pipeline_runner,
        "select_step4_source_manifest",
        lambda frame, **kwargs: frame.copy(),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "inspect_step4_readiness",
        lambda *args, **kwargs: {
            "ready_to_materialize": True,
            "ready_to_run_experiment": False,
        },
    )
    plan = build_common_experiment_plan(
        project_root=tmp_path,
        dataset_id="lfw",
        run_tier="quick",
        step4_config_path=config_path,
        evaluation_contract_path=contract_path,
    )

    report = inspect_common_experiment_plan(plan)

    assert report["ready_to_execute_pipeline"] is True
    assert report["evaluation_contract_coverage"]["pq_exhaustive_adc"] == (
        "implemented"
    )
    assert report["evaluation_contract_coverage"][
        "search_space_clean_full_rerun"
    ] == "validation_required"
    assert report["evaluation_contract_coverage"][
        "survface_gradcam_faithfulness_controls"
    ] == "implemented"
    assert report["evaluation_contract_coverage"][
        "survface_gradcam_faithfulness_three_models"
    ] == "implemented"
    assert report["evaluation_contract_coverage"][
        "gradcam_faithfulness_clean_promotion"
    ] == "validation_required"
    assert plan.comparison_paper_eligible is False


def test_execution_requires_explicit_user_acknowledgement(
    tmp_path: Path,
) -> None:
    plan = SimpleNamespace()

    with pytest.raises(RuntimeError, match="explicit local acknowledgement"):
        run_common_step4_experiment(plan)


class _SyntheticResumeRun:
    def __init__(
        self,
        run_dir: Path,
        *,
        frozen_config: dict[str, object],
        frozen_config_path: Path,
    ) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True)
        self.config = {"step4": frozen_config}
        self.events: list[tuple[str, dict[str, object]]] = []
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "inputs": [
                        {
                            "role": "step4_config",
                            "path": str(frozen_config_path),
                            "sha256": pipeline_runner.sha256_file(
                                frozen_config_path
                            ),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    def record_event(self, event: str, **details: object) -> None:
        self.events.append((event, details))


def _write_phase_attempt(
    run_dir: Path,
    phase_name: str,
    *,
    status: str,
    message: str | None = None,
) -> None:
    attempt_dir = run_dir / "phases" / phase_name / "attempts" / "A001"
    attempt_dir.mkdir(parents=True)
    payload: dict[str, object] = {
        "phase": phase_name,
        "attempt": 1,
        "status": status,
    }
    if message is not None:
        payload["failure"] = {"type": "ValueError", "message": message}
    (attempt_dir / "phase_manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_known_survface_quick_protocol_failure_resumes_with_frozen_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_snapshot = {
        "commit": "a" * 40,
        "branch": "step5",
        "dirty": True,
        "working_tree_diff_sha256": "b" * 64,
        "untracked_content_sha256": "c" * 64,
    }
    resume_snapshot = {
        **frozen_snapshot,
        "working_tree_diff_sha256": "d" * 64,
    }
    frozen_config: dict[str, object] = {
        "execution": {"data_fraction": 0.3},
        "orchestration": {
            "dataset_id": "survface",
            "run_tier": "quick",
            "source_snapshot": frozen_snapshot,
        },
    }
    current_config = {
        **frozen_config,
        "orchestration": {
            **frozen_config["orchestration"],
            "source_snapshot": resume_snapshot,
        },
    }
    frozen_config_path = tmp_path / "frozen.yaml"
    frozen_config_path.write_text(
        yaml.safe_dump(frozen_config, sort_keys=False),
        encoding="utf-8",
    )
    current_config_path = tmp_path / "current.yaml"
    current_config_path.write_text(
        yaml.safe_dump(current_config, sort_keys=False),
        encoding="utf-8",
    )
    run = _SyntheticResumeRun(
        tmp_path / "active-run",
        frozen_config=frozen_config,
        frozen_config_path=frozen_config_path,
    )
    for phase_name in (
        "00_source_and_model_freeze",
        "01_origin_embedding_and_target_templates",
        "02_population_gradcam_extraction",
        "03_saliency_feature_validation",
    ):
        _write_phase_attempt(run.run_dir, phase_name, status="completed")
    _write_phase_attempt(
        run.run_dir,
        "04_step2_compression_characterization",
        status="failed",
        message="gallery protocol_index must be unique and contiguous from 0",
    )
    plan = SimpleNamespace(
        dataset_id="survface",
        run_tier="quick",
        effective_step4_config=current_config,
        plan_id="resume-plan",
    )
    monkeypatch.setattr(
        pipeline_runner,
        "_active_dataset_run",
        lambda selected_plan: run,
    )

    resolved_run, resolved_path, context = (
        pipeline_runner._resolve_execution_run(
            plan,
            current_config_path=current_config_path,
        )
    )

    assert resolved_run is run
    assert resolved_path == frozen_config_path.resolve()
    assert context is not None
    assert context["correction_id"] == (
        "survface_quick_protocol_index_rebase_v1"
    )
    assert context["frozen_source_snapshot"] == frozen_snapshot
    assert context["resume_source_snapshot"] == resume_snapshot
    assert run.events[-1][0] == "source_correction_resume_authorized"


def test_known_retrieval_search_mode_join_failure_resumes_completed_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_snapshot = {
        "commit": "a" * 40,
        "branch": "step6",
        "dirty": False,
        "working_tree_diff_sha256": "b" * 64,
        "untracked_content_sha256": "c" * 64,
    }
    resume_snapshot = {
        **frozen_snapshot,
        "commit": "d" * 40,
    }
    frozen_config: dict[str, object] = {
        "execution": {"data_fraction": 1.0},
        "orchestration": {
            "dataset_id": "lfw",
            "run_tier": "full",
            "source_snapshot": frozen_snapshot,
        },
    }
    current_config = {
        **frozen_config,
        "orchestration": {
            **frozen_config["orchestration"],
            "source_snapshot": resume_snapshot,
        },
    }
    frozen_config_path = tmp_path / "frozen.yaml"
    frozen_config_path.write_text(
        yaml.safe_dump(frozen_config, sort_keys=False),
        encoding="utf-8",
    )
    current_config_path = tmp_path / "current.yaml"
    current_config_path.write_text(
        yaml.safe_dump(current_config, sort_keys=False),
        encoding="utf-8",
    )
    run = _SyntheticResumeRun(
        tmp_path / "active-run",
        frozen_config=frozen_config,
        frozen_config_path=frozen_config_path,
    )
    for phase_name in (
        "00_source_and_model_freeze",
        "01_origin_embedding_and_target_templates",
        "02_population_gradcam_extraction",
        "03_saliency_feature_validation",
        "04_step2_compression_characterization",
    ):
        _write_phase_attempt(run.run_dir, phase_name, status="completed")
    _write_phase_attempt(
        run.run_dir,
        "05_saliency_compression_join",
        status="failed",
        message=(
            "retrieval_sensitivity rows are not unique by "
            "['sample_id', 'compression_profile']"
        ),
    )
    retrieval_path = (
        run.run_dir
        / "artifacts"
        / "step2_workflow"
        / "retrieval_metrics.csv"
    )
    retrieval_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "query_id": ["sample-1", "sample-1"],
            "search_mode": [
                "pca_direct_cosine",
                "pca_reconstruction_cosine",
            ],
        }
    ).to_csv(retrieval_path, index=False)
    plan = SimpleNamespace(
        dataset_id="lfw",
        run_tier="full",
        effective_step4_config=current_config,
        plan_id="resume-retrieval-plan",
    )
    monkeypatch.setattr(
        pipeline_runner,
        "_active_dataset_run",
        lambda selected_plan: run,
    )

    resolved_run, resolved_path, context = (
        pipeline_runner._resolve_execution_run(
            plan,
            current_config_path=current_config_path,
        )
    )

    assert resolved_run is run
    assert resolved_path == frozen_config_path.resolve()
    assert context is not None
    assert context["correction_id"] == (
        "retrieval_search_mode_join_grain_v1"
    )
    assert context["frozen_source_snapshot"] == frozen_snapshot
    assert context["resume_source_snapshot"] == resume_snapshot
    assert run.events[-1][0] == "source_correction_resume_authorized"


def test_known_representative_case_memory_failure_resumes_phase06_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_snapshot = {"commit": "a" * 40, "branch": "step6", "dirty": False}
    resume_snapshot = {**frozen_snapshot, "commit": "b" * 40}
    frozen_config: dict[str, object] = {
        "execution": {"data_fraction": 1.0},
        "workflow": {"persist_large_join_artifacts": True},
        "gradcam": {"persistence": {"persist_raw_cam": True}},
        "orchestration": {
            "dataset_id": "survface",
            "run_tier": "full",
            "artifact_storage_mode": "full",
            "source_snapshot": frozen_snapshot,
        },
    }
    current_config: dict[str, object] = {
        **frozen_config,
        "workflow": {
            "artifact_storage_mode": "results_only",
            "persist_large_join_artifacts": False,
            "representative_case_candidates_path": (
                "representative_case_candidates.csv"
            ),
        },
        "gradcam": {"persistence": {"persist_raw_cam": False}},
        "orchestration": {
            **frozen_config["orchestration"],
            "artifact_storage_mode": "results_only",
            "source_snapshot": resume_snapshot,
        },
    }
    frozen_config_path = tmp_path / "frozen.yaml"
    frozen_config_path.write_text(
        yaml.safe_dump(frozen_config, sort_keys=False),
        encoding="utf-8",
    )
    current_config_path = tmp_path / "current.yaml"
    current_config_path.write_text(
        yaml.safe_dump(current_config, sort_keys=False),
        encoding="utf-8",
    )
    run = _SyntheticResumeRun(
        tmp_path / "active-run",
        frozen_config=frozen_config,
        frozen_config_path=frozen_config_path,
    )
    for phase_name in (
        "00_source_and_model_freeze",
        "01_origin_embedding_and_target_templates",
        "02_population_gradcam_extraction",
        "03_saliency_feature_validation",
        "04_step2_compression_characterization",
        "05_saliency_compression_join",
    ):
        _write_phase_attempt(run.run_dir, phase_name, status="completed")
    _write_phase_attempt(
        run.run_dir,
        "06_representative_case_visualization",
        status="failed",
        message="Error tokenizing data. C error: out of memory",
    )
    workflow = run.run_dir / "artifacts" / "step2_workflow"
    workflow.mkdir(parents=True)
    for name in ("saliency_geometry_join.csv", "saliency_retrieval_join.csv"):
        (workflow / name).write_text("sample_id\n", encoding="utf-8")
    plan = SimpleNamespace(
        dataset_id="survface",
        run_tier="full",
        effective_step4_config=current_config,
        plan_id="resume-representative-plan",
    )
    monkeypatch.setattr(
        pipeline_runner,
        "_active_dataset_run",
        lambda selected_plan: run,
    )

    resolved_run, resolved_path, context = pipeline_runner._resolve_execution_run(
        plan,
        current_config_path=current_config_path,
    )

    assert resolved_run is run
    assert resolved_path == frozen_config_path.resolve()
    assert context is not None
    assert context["correction_id"] == "representative_case_streaming_memory_v1"
    assert run.events[-1][0] == "source_correction_resume_authorized"


def test_explicit_completed_override_allows_only_source_snapshot_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    selected = root / "runs" / "lfw_20260803" / "completed-run"
    selected.mkdir(parents=True)
    frozen_config = {
        "run": {"root": "runs"},
        "execution": {"data_fraction": 1.0},
        "orchestration": {
            "dataset_id": "lfw",
            "source_snapshot": {"commit": "a" * 40},
        },
    }
    current_config = {
        **frozen_config,
        "orchestration": {
            **frozen_config["orchestration"],
            "source_snapshot": {"commit": "b" * 40},
        },
    }
    (selected / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "20260803-R001-test",
                "status": "completed",
                "config": {"step4": frozen_config},
            }
        ),
        encoding="utf-8",
    )
    (selected / "COMPLETED").write_text("done\n", encoding="utf-8")

    class _VerifiedCompletedRun:
        def __init__(self) -> None:
            self.verified: list[str] = []

        def verify_phase_artifacts(self, phase_name: str) -> None:
            self.verified.append(phase_name)

    verified = _VerifiedCompletedRun()
    monkeypatch.setattr(
        pipeline_runner.RunStore,
        "open",
        lambda path: verified,
    )
    monkeypatch.setattr(
        pipeline_runner,
        "_completed_phase",
        lambda run_dir, phase_name: True,
    )
    plan = SimpleNamespace(
        project_root=root,
        effective_step4_config=current_config,
        plan_id="current-plan",
    )

    result = reuse_completed_run_for_plan(plan, selected)

    assert result["status"] == "already_completed"
    assert result["explicit_completed_override"] is True
    assert result["run_id"] == "20260803-R001-test"
    assert len(verified.verified) == 7

    changed = {
        **current_config,
        "execution": {"data_fraction": 0.5},
    }
    changed_plan = SimpleNamespace(
        project_root=root,
        effective_step4_config=changed,
        plan_id="changed-plan",
    )
    with pytest.raises(ValueError, match="science config"):
        reuse_completed_run_for_plan(changed_plan, selected)


def test_protocol_failure_resume_rejects_any_non_snapshot_config_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_config = {
        "execution": {"data_fraction": 0.3},
        "orchestration": {
            "source_snapshot": {"working_tree_diff_sha256": "a" * 64}
        },
    }
    changed_config = {
        "execution": {"data_fraction": 0.2},
        "orchestration": {
            "source_snapshot": {"working_tree_diff_sha256": "b" * 64}
        },
    }
    frozen_path = tmp_path / "frozen.yaml"
    frozen_path.write_text(
        yaml.safe_dump(frozen_config, sort_keys=False),
        encoding="utf-8",
    )
    run = _SyntheticResumeRun(
        tmp_path / "active-run",
        frozen_config=frozen_config,
        frozen_config_path=frozen_path,
    )
    plan = SimpleNamespace(
        dataset_id="survface",
        run_tier="quick",
        effective_step4_config=changed_config,
        plan_id="changed-plan",
    )
    monkeypatch.setattr(
        pipeline_runner,
        "_active_dataset_run",
        lambda selected_plan: run,
    )

    with pytest.raises(RuntimeError, match="different incomplete dataset run"):
        pipeline_runner._resolve_execution_run(
            plan,
            current_config_path=tmp_path / "current.yaml",
        )
