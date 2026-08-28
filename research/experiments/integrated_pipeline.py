from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

from research.experiments.pipeline_runner import (
    DEFAULT_RUN_TIER,
    QUICK_DATA_FRACTIONS,
    SUPPORTED_COMMON_DATASETS,
    SUPPORTED_ORCHESTRATION_DATASETS,
    CommonExperimentPlan,
    CommonModelPreparation,
    build_common_experiment_plan,
    inspect_common_experiment_plan,
    reuse_completed_run_for_plan,
    run_common_step4_experiment,
)
from research.experiments.tinyface_pipeline import (
    TinyFaceExperimentPlan,
    build_tinyface_experiment_plan,
    inspect_tinyface_experiment_plan,
    reuse_completed_tinyface_run,
    run_tinyface_experiment,
)


OPEN_SET_DATASET_IDS = tuple(SUPPORTED_COMMON_DATASETS)
INTEGRATED_DATASET_IDS = tuple(SUPPORTED_ORCHESTRATION_DATASETS)
IntegratedExperimentPlan: TypeAlias = CommonExperimentPlan | TinyFaceExperimentPlan
ProgressCallback: TypeAlias = Callable[[str, dict[str, object]], None]


def validate_integrated_dataset_ids(dataset_ids: Sequence[str]) -> tuple[str, ...]:
    resolved = tuple(str(value).strip().lower() for value in dataset_ids)
    if not resolved:
        raise ValueError("DATASET_IDS must contain at least one dataset")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"DATASET_IDS contains duplicates: {resolved}")
    unsupported = sorted(set(resolved) - set(INTEGRATED_DATASET_IDS))
    if unsupported:
        raise ValueError(
            f"unsupported DATASET_IDS: {unsupported}; expected a subset of "
            f"{INTEGRATED_DATASET_IDS}"
        )
    return resolved


def validate_integrated_quick_data_fractions(
    values: Mapping[str, float],
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError("QUICK_DATA_FRACTIONS must be a mapping")
    if set(values) != set(INTEGRATED_DATASET_IDS):
        raise ValueError(
            "QUICK_DATA_FRACTIONS keys must be exactly "
            f"{INTEGRATED_DATASET_IDS}"
        )
    resolved: dict[str, float] = {}
    for dataset_id in INTEGRATED_DATASET_IDS:
        value = values[dataset_id]
        if isinstance(value, bool):
            raise ValueError("quick fractions must be numeric values in (0, 1]")
        fraction = float(value)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("quick fractions must be in (0, 1]")
        resolved[dataset_id] = fraction
    return resolved


def validate_completed_run_overrides(
    overrides: Mapping[str, str | Path],
    *,
    selected_dataset_ids: Sequence[str],
) -> dict[str, str | Path]:
    if not isinstance(overrides, Mapping):
        raise TypeError("COMPLETED_RUN_OVERRIDES must be a mapping")
    selected = set(validate_integrated_dataset_ids(selected_dataset_ids))
    unsupported = sorted(set(overrides) - selected)
    if unsupported:
        raise ValueError(
            "COMPLETED_RUN_OVERRIDES contains datasets not selected in "
            f"DATASET_IDS: {unsupported}"
        )
    return {str(dataset): value for dataset, value in overrides.items()}


def build_integrated_experiment_plans(
    *,
    project_root: str | Path,
    dataset_ids: Sequence[str],
    run_tier: str = DEFAULT_RUN_TIER,
    quick_data_fractions: Mapping[str, float],
    seed: int,
    model_preparation: CommonModelPreparation,
    model_name: str,
    artifact_storage_mode: str = "results_only",
    device: str = "cuda",
    pq_sdc_settings: Sequence[tuple[int, int]] = ((128, 8),),
) -> dict[str, IntegratedExperimentPlan]:
    """Build every selected dataset plan from one visible notebook contract."""

    selected = validate_integrated_dataset_ids(dataset_ids)
    fractions = validate_integrated_quick_data_fractions(quick_data_fractions)
    plans: dict[str, IntegratedExperimentPlan] = {}
    for dataset_id in selected:
        if dataset_id == "tinyface":
            plan: IntegratedExperimentPlan = build_tinyface_experiment_plan(
                project_root=project_root,
                run_tier=run_tier,
                seed=seed,
                model_name=model_name,
                model_profile=model_preparation.model_profile,
                model_uid=model_preparation.model_uid,
                model_spec_path=model_preparation.model_spec_path,
                device=device,
                quick_data_fraction=fractions[dataset_id],
                pq_sdc_settings=pq_sdc_settings,
            )
        else:
            plan = build_common_experiment_plan(
                project_root=project_root,
                dataset_id=dataset_id,
                run_tier=run_tier,
                seed=seed,
                model_name=model_name,
                model_profile=model_preparation.model_profile,
                model_uid=model_preparation.model_uid,
                model_checkpoint_path=model_preparation.checkpoint_path,
                quick_data_fractions=fractions,
                artifact_storage_mode=artifact_storage_mode,
                pq_sdc_settings=pq_sdc_settings,
            )
        if plan.dataset_id != dataset_id:
            raise RuntimeError(
                f"integrated plan dataset mismatch: {dataset_id} != {plan.dataset_id}"
            )
        plans[dataset_id] = plan
    return plans


def inspect_integrated_experiment_plan(
    plan: IntegratedExperimentPlan,
) -> dict[str, Any]:
    if isinstance(plan, TinyFaceExperimentPlan):
        return inspect_tinyface_experiment_plan(plan)
    if isinstance(plan, CommonExperimentPlan):
        return inspect_common_experiment_plan(plan)
    raise TypeError(f"unsupported integrated plan type: {type(plan).__name__}")


def inspect_integrated_experiment_plans(
    plans: Mapping[str, IntegratedExperimentPlan],
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for dataset_id, plan in plans.items():
        if plan.dataset_id != dataset_id:
            raise ValueError(
                f"integrated plan key mismatch: {dataset_id} != {plan.dataset_id}"
            )
        reports[dataset_id] = inspect_integrated_experiment_plan(plan)
    return reports


def run_or_reuse_integrated_experiment(
    plan: IntegratedExperimentPlan,
    *,
    execution_acknowledged: bool,
    start_new_run: bool,
    completed_run_override: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Dispatch one selected dataset to its protocol-specific canonical runner."""

    if isinstance(plan, TinyFaceExperimentPlan):
        if completed_run_override is not None:
            return reuse_completed_tinyface_run(plan, completed_run_override)
        return run_tinyface_experiment(
            plan,
            execution_acknowledged=execution_acknowledged,
            start_new_run=start_new_run,
            progress=progress,
        )
    if isinstance(plan, CommonExperimentPlan):
        if completed_run_override is not None:
            return reuse_completed_run_for_plan(plan, completed_run_override)
        return run_common_step4_experiment(
            plan,
            execution_acknowledged=execution_acknowledged,
            start_new_run=start_new_run,
            progress=progress,
        )
    raise TypeError(f"unsupported integrated plan type: {type(plan).__name__}")


def is_open_set_dataset(dataset_id: str) -> bool:
    resolved = str(dataset_id).strip().lower()
    if resolved not in INTEGRATED_DATASET_IDS:
        raise ValueError(f"unsupported dataset_id: {dataset_id!r}")
    return resolved in OPEN_SET_DATASET_IDS


def default_integrated_quick_data_fractions() -> dict[str, float]:
    return validate_integrated_quick_data_fractions(QUICK_DATA_FRACTIONS)
