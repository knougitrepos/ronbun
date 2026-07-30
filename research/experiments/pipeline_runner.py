from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import pandas as pd
import yaml

from research.experiments.scope import ExperimentScope
from research.experiments.step4_datasets import (
    load_step4_source_manifest,
    resolve_step4_dataset_spec,
)
from research.experiments.step4_workflow import (
    analyze_step4_saliency_compression,
    characterize_step4_compression,
    extract_step4_origin_embeddings,
    extract_step4_population_gradcam,
    finalize_step4_representative_cases,
    freeze_step4_source_and_model,
    inspect_step4_readiness,
    load_step4_config,
    materialize_step4_aligned_crops,
    materialize_step4_landmark_regions,
    select_step4_source_manifest,
    validate_step4_saliency,
)
from research.runtime import (
    RunStore,
    resolve_active_dataset_run,
)
from research.runtime.hashing import canonical_sha256, sha256_file


SUPPORTED_COMMON_DATASETS = ("lfw", "survface")
SUPPORTED_RUN_TIERS = ("quick", "full")
QUICK_DATA_FRACTIONS: Mapping[str, float] = {
    "lfw": 0.10,
    "survface": 0.02,
}
FULL_DATA_FRACTION = 1.00
DEFAULT_STEP4_CONFIG_PATH = Path("configs/experiments/step2_pytorch_gradcam.yaml")
DEFAULT_EVALUATION_CONTRACT_PATH = Path(
    "configs/experiments/evaluation_contract_v1.yaml"
)

MAIN_COMPARISON_PROFILES = (
    "origin_512",
    "pca_256",
    "pca_128",
    "pq_origin_512_m128_nbits8",
)
FORMAL_FPIR_TARGETS = (0.30, 0.20, 0.10, 0.01, 0.001)
EXPLORATORY_FPIR_TARGETS = (0.0001,)
CALIBRATION_IDENTITY_COUNTS = (100, 500, 1000)


@dataclass(frozen=True)
class PipelineStage:
    stage_id: str
    phase_name: str | None
    description: str

    def as_dict(self) -> dict[str, str | None]:
        return {
            "stage_id": self.stage_id,
            "phase_name": self.phase_name,
            "description": self.description,
        }


STEP4_PIPELINE_STAGES = (
    PipelineStage(
        "aligned_crops",
        None,
        "Validate or materialize the canonical aligned RGB uint8 NHWC bundle.",
    ),
    PipelineStage(
        "landmark_regions",
        None,
        "Validate or materialize the canonical landmark-region bundle.",
    ),
    PipelineStage(
        "source_model_freeze",
        "00_source_and_model_freeze",
        "Freeze the selected real-data scope and registered model identity.",
    ),
    PipelineStage(
        "origin_embeddings",
        "01_origin_embedding_and_target_templates",
        "Extract origin embeddings and saliency target templates.",
    ),
    PipelineStage(
        "population_gradcam",
        "02_population_gradcam_extraction",
        "Extract population Grad-CAM features for the deterministic scope.",
    ),
    PipelineStage(
        "saliency_validation",
        "03_saliency_feature_validation",
        "Validate saliency coverage and stored feature contracts.",
    ),
    PipelineStage(
        "compression_characterization",
        "04_step2_compression_characterization",
        "Run the currently implemented PCA/PQ reconstruction and retrieval study.",
    ),
    PipelineStage(
        "saliency_compression_join",
        "05_saliency_compression_join",
        "Join saliency, geometry, and retrieval results and compute associations.",
    ),
    PipelineStage(
        "representative_cases",
        "06_representative_case_visualization",
        "Create representative figures and complete the immutable run.",
    ),
)


@dataclass(frozen=True)
class CommonExperimentPlan:
    project_root: Path
    dataset_id: str
    run_tier: str
    data_fraction: float
    seed: int
    model_profile: str
    pipeline_id: str
    evaluation_contract_id: str
    base_step4_config_path: Path
    evaluation_contract_path: Path
    evaluation_contract_sha256: str
    effective_step4_config: dict[str, Any]
    plan_id: str
    source_rows: int
    selected_source_rows: int
    source_manifest_sha256: dict[str, str]
    selected_image_ids_sha256: str
    selected_split_counts: dict[str, int]
    selected_role_counts: dict[str, int]
    scope_paper_eligible: bool
    comparison_paper_eligible: bool
    stages: tuple[PipelineStage, ...] = STEP4_PIPELINE_STAGES

    def as_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "pipeline_id": self.pipeline_id,
            "dataset_id": self.dataset_id,
            "run_tier": self.run_tier,
            "data_fraction": self.data_fraction,
            "seed": self.seed,
            "model_profile": self.model_profile,
            "evaluation_contract_id": self.evaluation_contract_id,
            "evaluation_contract_path": str(self.evaluation_contract_path),
            "evaluation_contract_sha256": self.evaluation_contract_sha256,
            "base_step4_config_path": str(self.base_step4_config_path),
            "source_rows": self.source_rows,
            "selected_source_rows": self.selected_source_rows,
            "source_manifest_sha256": self.source_manifest_sha256,
            "selected_image_ids_sha256": self.selected_image_ids_sha256,
            "selected_split_counts": self.selected_split_counts,
            "selected_role_counts": self.selected_role_counts,
            "scope_paper_eligible": self.scope_paper_eligible,
            "comparison_paper_eligible": self.comparison_paper_eligible,
            "stages": [stage.as_dict() for stage in self.stages],
        }


def _resolve_project_file(
    project_root: Path,
    path: str | Path,
    *,
    label: str,
) -> Path:
    candidate = Path(path)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (project_root / candidate).resolve()
    )
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the project root") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def load_evaluation_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path).resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    if not isinstance(contract, dict):
        raise ValueError("evaluation contract root must be a mapping")
    _validate_evaluation_contract(contract)
    return contract


def _numeric_tuple(values: object, *, label: str) -> tuple[float, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numbers") from exc


def _validate_evaluation_contract(contract: Mapping[str, Any]) -> None:
    contract_id = str(contract.get("contract_id", "")).strip()
    if not contract_id:
        raise ValueError("evaluation contract_id must not be empty")

    execution = contract.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("evaluation execution section must be a mapping")
    if tuple(execution.get("tiers", ())) != SUPPORTED_RUN_TIERS:
        raise ValueError(f"execution tiers must be exactly {SUPPORTED_RUN_TIERS}")
    quick = execution.get("quick_data_fractions")
    if not isinstance(quick, Mapping):
        raise ValueError("quick_data_fractions must be a mapping")
    actual_quick = {
        str(dataset): float(fraction) for dataset, fraction in quick.items()
    }
    if actual_quick != dict(QUICK_DATA_FRACTIONS):
        raise ValueError("quick_data_fractions differ from the user-confirmed contract")
    if float(execution.get("full_data_fraction", -1.0)) != FULL_DATA_FRACTION:
        raise ValueError("full_data_fraction must be 1.0")

    comparison = contract.get("comparison")
    if not isinstance(comparison, Mapping):
        raise ValueError("comparison section must be a mapping")
    if tuple(comparison.get("main_profiles", ())) != MAIN_COMPARISON_PROFILES:
        raise ValueError("main_profiles must be Origin-512/PCA-256/PCA-128/PQ-m128")

    pq_search = contract.get("pq_search")
    if not isinstance(pq_search, Mapping):
        raise ValueError("pq_search section must be a mapping")
    if pq_search.get("primary") != "exhaustive_adc":
        raise ValueError("PQ primary search must be exhaustive_adc")
    if pq_search.get("system_ablation") != "ivf_pq_selected_profiles":
        raise ValueError("PQ system ablation must be ivf_pq_selected_profiles")
    if pq_search.get("decoded_vector_cosine_is_adc") is not False:
        raise ValueError("decoded-vector cosine must not be labeled ADC")

    calibration = contract.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("calibration section must be a mapping")
    actual_counts = tuple(
        int(value) for value in calibration.get("enrollment_identity_counts", ())
    )
    if actual_counts != CALIBRATION_IDENTITY_COUNTS:
        raise ValueError(f"calibration counts must be {CALIBRATION_IDENTITY_COUNTS}")

    evaluation = contract.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation section must be a mapping")
    formal = _numeric_tuple(
        evaluation.get("formal_fpir_targets"),
        label="formal_fpir_targets",
    )
    exploratory = _numeric_tuple(
        evaluation.get("exploratory_fpir_targets"),
        label="exploratory_fpir_targets",
    )
    if formal != FORMAL_FPIR_TARGETS:
        raise ValueError(f"formal FPIR targets must be {FORMAL_FPIR_TARGETS}")
    if exploratory != EXPLORATORY_FPIR_TARGETS:
        raise ValueError(f"exploratory FPIR targets must be {EXPLORATORY_FPIR_TARGETS}")


def _effective_step4_config(
    base_config: Mapping[str, Any],
    *,
    dataset_id: str,
    run_tier: str,
    data_fraction: float,
    seed: int,
    model_profile: str | None,
    contract_id: str,
    contract_sha256: str,
) -> dict[str, Any]:
    config = deepcopy(dict(base_config))
    execution = config.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("Step 4 execution config must be a mapping")
    profiles = config.get("models", {}).get("profiles", {})
    selected_profile = str(model_profile or execution.get("model_profile", "")).strip()
    if selected_profile not in profiles:
        raise ValueError(
            f"unknown model_profile {selected_profile!r}; available={sorted(profiles)}"
        )
    execution.update(
        {
            "model_profile": selected_profile,
            "mode": "real",
            "data_fraction": data_fraction,
            "seed": seed,
            "execute_stage": True,
            "write_outputs": True,
            "overwrite": False,
            "allow_dirty": False,
        }
    )
    config["orchestration"] = {
        "pipeline_id": "common_step4_gradcam_v1",
        "dataset_id": dataset_id,
        "run_tier": run_tier,
        "evaluation_contract_id": contract_id,
        "evaluation_contract_sha256": contract_sha256,
        "comparison_contract_coverage": "partial",
    }
    return config


def _count_values(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    counts = frame[column].astype(str).value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _display_path(path: Path, *, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)


def build_common_experiment_plan(
    *,
    project_root: str | Path,
    dataset_id: str,
    run_tier: str,
    seed: int = 42,
    model_profile: str | None = None,
    step4_config_path: str | Path = DEFAULT_STEP4_CONFIG_PATH,
    evaluation_contract_path: str | Path = (DEFAULT_EVALUATION_CONTRACT_PATH),
) -> CommonExperimentPlan:
    """Build one deterministic real-data Step 4 plan without starting a run."""

    root = Path(project_root).resolve()
    dataset = str(dataset_id).strip().lower()
    tier = str(run_tier).strip().lower()
    if dataset not in SUPPORTED_COMMON_DATASETS:
        raise ValueError(f"dataset_id must be one of {SUPPORTED_COMMON_DATASETS}")
    if tier not in SUPPORTED_RUN_TIERS:
        raise ValueError(f"run_tier must be one of {SUPPORTED_RUN_TIERS}")
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    resolved_seed = int(seed)
    data_fraction = (
        float(QUICK_DATA_FRACTIONS[dataset]) if tier == "quick" else FULL_DATA_FRACTION
    )
    if tier == "full" and data_fraction != 1.0:
        raise RuntimeError("full runs must use data_fraction=1.0")

    base_path = _resolve_project_file(
        root,
        step4_config_path,
        label="Step 4 config",
    )
    contract_path = _resolve_project_file(
        root,
        evaluation_contract_path,
        label="evaluation contract",
    )
    base_config = load_step4_config(base_path)
    contract = load_evaluation_contract(contract_path)
    contract_id = str(contract["contract_id"])
    contract_sha256 = sha256_file(contract_path)
    effective_config = _effective_step4_config(
        base_config,
        dataset_id=dataset,
        run_tier=tier,
        data_fraction=data_fraction,
        seed=resolved_seed,
        model_profile=model_profile,
        contract_id=contract_id,
        contract_sha256=contract_sha256,
    )

    spec = resolve_step4_dataset_spec(
        effective_config,
        project_root=root,
        dataset_id=dataset,
    )
    source = load_step4_source_manifest(spec)
    scope = ExperimentScope(
        mode="real",
        data_fraction=data_fraction,
        seed=resolved_seed,
    )
    selected = select_step4_source_manifest(
        source,
        dataset_id=dataset,
        scope=scope,
    )
    if "image_id" not in selected:
        raise ValueError("selected source manifest is missing image_id")
    selected_image_ids_sha256 = canonical_sha256(
        sorted(selected["image_id"].astype(str).tolist())
    )
    manifest_paths = tuple(getattr(spec, "manifest_paths", ()))
    source_manifest_sha256 = {
        _display_path(Path(path), project_root=root): sha256_file(path)
        for path in manifest_paths
    }
    fingerprint_payload = {
        "pipeline_id": "common_step4_gradcam_v1",
        "dataset_id": dataset,
        "run_tier": tier,
        "evaluation_contract_sha256": contract_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "selected_image_ids_sha256": selected_image_ids_sha256,
        "effective_step4_config": effective_config,
    }
    plan_id = canonical_sha256(fingerprint_payload)[:16]
    selected_profile = str(effective_config["execution"]["model_profile"])
    return CommonExperimentPlan(
        project_root=root,
        dataset_id=dataset,
        run_tier=tier,
        data_fraction=data_fraction,
        seed=resolved_seed,
        model_profile=selected_profile,
        pipeline_id="common_step4_gradcam_v1",
        evaluation_contract_id=contract_id,
        base_step4_config_path=base_path,
        evaluation_contract_path=contract_path,
        evaluation_contract_sha256=contract_sha256,
        effective_step4_config=effective_config,
        plan_id=plan_id,
        source_rows=int(len(source)),
        selected_source_rows=int(len(selected)),
        source_manifest_sha256=source_manifest_sha256,
        selected_image_ids_sha256=selected_image_ids_sha256,
        selected_split_counts=_count_values(selected, "split"),
        selected_role_counts=_count_values(selected, "protocol_role"),
        scope_paper_eligible=scope.is_paper_run,
        comparison_paper_eligible=False,
    )


def _write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    text = yaml.safe_dump(
        dict(payload),
        sort_keys=False,
        allow_unicode=True,
    )
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if existing != payload:
            raise RuntimeError(
                f"effective config path contains different content: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def materialize_effective_step4_config(
    plan: CommonExperimentPlan,
) -> Path:
    """Persist the restart-stable effective config under the ignored run root."""

    path = (
        plan.project_root
        / "runs"
        / "orchestration"
        / "effective_configs"
        / f"{plan.plan_id}.yaml"
    )
    _write_yaml_atomic(path, plan.effective_step4_config)
    return path


def inspect_common_experiment_plan(
    plan: CommonExperimentPlan,
) -> dict[str, object]:
    """Run read-only local readiness checks for a common experiment plan."""

    with tempfile.TemporaryDirectory(prefix="ronbun-plan-") as directory:
        config_path = Path(directory) / f"{plan.plan_id}.yaml"
        config_path.write_text(
            yaml.safe_dump(
                plan.effective_step4_config,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        readiness = inspect_step4_readiness(
            config_path,
            project_root=plan.project_root,
            dataset_id=plan.dataset_id,
        )
    coverage = {
        "quick_full_scope": "implemented",
        "common_step4_phase_dispatch": "implemented",
        "identity_aware_role_preserving_selection": "implemented",
        "origin_pca_reconstruction_study": "implemented",
        "same_model_lfw_survface_full_rerun": "validation_required",
        "four_profile_main_table": "proposed",
        "pq_exhaustive_adc": "proposed",
        "ivf_pq_system_ablation": "proposed",
        "official_and_db_baseline_matrix": "proposed",
        "calibration_100_500_1000": "proposed",
        "full_fpir_contract": "proposed",
        "gradcam_promotion_gates": "proposed",
    }
    return {
        "plan": plan.as_dict(),
        "readiness": readiness,
        "ready_to_execute_pipeline": bool(readiness["ready_to_materialize"]),
        "evaluation_contract_coverage": coverage,
        "paper_result_warning": (
            "A full-data scope alone is not a paper-final comparison. "
            "The proposed evaluation-contract stages above must also be "
            "implemented and rerun from one clean source commit."
        ),
    }


def _completed_phase(run_dir: Path, phase_name: str) -> bool:
    phase_slug = phase_name.replace(" ", "-")
    attempts = run_dir / "phases" / phase_slug / "attempts"
    for path in sorted(attempts.glob("A*/phase_manifest.json"), reverse=True):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            return True
    return False


def _completed_matching_runs(plan: CommonExperimentPlan) -> list[Path]:
    base_root = (
        plan.project_root / plan.effective_step4_config["run"]["root"]
    ).resolve()
    matches: list[tuple[str, Path]] = []
    for manifest_path in base_root.rglob("run_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        completed = (
            manifest.get("status") == "completed"
            or (manifest_path.parent / "COMPLETED").is_file()
        )
        if not completed:
            continue
        config = manifest.get("config")
        if not isinstance(config, Mapping):
            continue
        if config.get("step4") != plan.effective_step4_config:
            continue
        matches.append((str(manifest.get("created_at_utc", "")), manifest_path.parent))
    return [path for _, path in sorted(matches, key=lambda item: item[0])]


def _active_matching_run(
    plan: CommonExperimentPlan,
) -> RunStore | None:
    config = plan.effective_step4_config
    try:
        run_dir = resolve_active_dataset_run(
            plan.project_root / config["run"]["root"],
            dataset_id=plan.dataset_id,
            directory_template=config["run"]["dataset_date_dir_template"],
        )
    except FileNotFoundError:
        return None
    run = RunStore.open(run_dir)
    if run.config.get("step4") != config:
        raise RuntimeError(
            "a different incomplete dataset run is active. Complete or "
            "selectively clean that run before executing this plan: "
            f"{run.run_dir}"
        )
    return run


def _call_step4_stage(
    stage_id: str,
    *,
    config_path: Path,
    plan: CommonExperimentPlan,
    progress: Callable[[str, dict[str, object]], None] | None,
) -> dict[str, object]:
    common = {
        "project_root": plan.project_root,
        "dataset_id": plan.dataset_id,
        "execution_acknowledged": True,
    }
    if stage_id == "origin_embeddings":
        return extract_step4_origin_embeddings(
            config_path,
            progress=progress,
            **common,
        )
    if stage_id == "population_gradcam":
        return extract_step4_population_gradcam(
            config_path,
            progress=progress,
            **common,
        )
    if stage_id == "saliency_validation":
        return validate_step4_saliency(config_path, **common)
    if stage_id == "compression_characterization":
        return characterize_step4_compression(
            config_path,
            progress=progress,
            **common,
        )
    if stage_id == "saliency_compression_join":
        return analyze_step4_saliency_compression(
            config_path,
            progress=progress,
            **common,
        )
    if stage_id == "representative_cases":
        return finalize_step4_representative_cases(
            config_path,
            **common,
        )
    raise ValueError(f"unsupported dispatched stage: {stage_id}")


def run_common_step4_experiment(
    plan: CommonExperimentPlan,
    *,
    execution_acknowledged: bool = False,
    start_new_run: bool = False,
    progress: Callable[[str, dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run or resume the canonical Step 4 phases without running notebooks.

    This dispatcher covers the existing Step 4 Grad-CAM workflow.  It does not
    claim that the newly proposed exhaustive-ADC and calibration-matrix stages
    are implemented.
    """

    if execution_acknowledged is not True:
        raise RuntimeError("common execution requires explicit local acknowledgement")
    preflight = inspect_common_experiment_plan(plan)
    if not preflight["ready_to_execute_pipeline"]:
        raise RuntimeError("local preflight failed; inspect readiness before execution")
    config_path = materialize_effective_step4_config(plan)
    run = _active_matching_run(plan)
    if run is None:
        completed = _completed_matching_runs(plan)
        if completed and not start_new_run:
            latest = completed[-1]
            manifest = json.loads(
                (latest / "run_manifest.json").read_text(encoding="utf-8")
            )
            return {
                "status": "already_completed",
                "run_id": manifest["run_id"],
                "run_dir": str(latest),
                "plan_id": plan.plan_id,
                "materialization": [],
                "message": (
                    "A matching completed run exists. Set start_new_run=True "
                    "only when an intentional independent rerun is required."
                ),
            }

    materialized: list[dict[str, object]] = []
    materialized.append(
        materialize_step4_aligned_crops(
            config_path,
            project_root=plan.project_root,
            dataset_id=plan.dataset_id,
            execution_acknowledged=True,
            progress=progress,
        )
    )
    materialized.append(
        materialize_step4_landmark_regions(
            config_path,
            project_root=plan.project_root,
            dataset_id=plan.dataset_id,
            execution_acknowledged=True,
            progress=progress,
        )
    )

    if run is None:
        freeze_result = freeze_step4_source_and_model(
            config_path,
            project_root=plan.project_root,
            dataset_id=plan.dataset_id,
            execution_acknowledged=True,
            execution_source="common_orchestration_notebook",
        )
        run = _active_matching_run(plan)
        if run is None:
            raise RuntimeError("source/model freeze did not create an active run")
        stage_results: list[dict[str, object]] = [
            {
                "stage_id": "source_model_freeze",
                "status": "completed",
                "result": freeze_result,
            }
        ]
    else:
        stage_results = []
        if _completed_phase(run.run_dir, "00_source_and_model_freeze"):
            stage_results.append(
                {
                    "stage_id": "source_model_freeze",
                    "status": "skipped_completed",
                }
            )
        else:
            freeze_result = freeze_step4_source_and_model(
                config_path,
                project_root=plan.project_root,
                dataset_id=plan.dataset_id,
                execution_acknowledged=True,
                execution_source="common_orchestration_notebook",
            )
            stage_results.append(
                {
                    "stage_id": "source_model_freeze",
                    "status": "completed",
                    "result": freeze_result,
                }
            )
            run = _active_matching_run(plan)
            if run is None:
                raise RuntimeError("active run disappeared after source freeze")

    for stage in STEP4_PIPELINE_STAGES[3:]:
        assert stage.phase_name is not None
        if _completed_phase(run.run_dir, stage.phase_name):
            stage_results.append(
                {
                    "stage_id": stage.stage_id,
                    "status": "skipped_completed",
                }
            )
            continue
        result = _call_step4_stage(
            stage.stage_id,
            config_path=config_path,
            plan=plan,
            progress=progress,
        )
        stage_results.append(
            {
                "stage_id": stage.stage_id,
                "status": "completed",
                "result": result,
            }
        )

    final_manifest = json.loads(
        (run.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "status": str(final_manifest.get("status", "unknown")),
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "plan_id": plan.plan_id,
        "materialization": materialized,
        "stages": stage_results,
        "comparison_paper_eligible": False,
        "comparison_limitation": (
            "The common Step 4 dispatcher is implemented, but exhaustive ADC, "
            "the complete calibration/FPIR matrix, and same-commit dual-dataset "
            "full reruns remain outstanding."
        ),
    }
