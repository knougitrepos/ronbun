from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import yaml

from research.embeddings import (
    CheckpointProvenance,
    ModelSpec,
    PreprocessingSpec,
    create_pytorch_adapter_from_spec,
    resolve_smoke_input_batch,
    write_model_spec,
)
from research.embeddings.manifests import model_spec_registry_stem
from research.experiments.scope import ExperimentScope
from research.experiments.step4_datasets import (
    load_step4_source_manifest,
    resolve_step4_dataset_spec,
)
from research.experiments.step4_workflow import (
    SOURCE_SNAPSHOT_FIELDS,
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
    inspect_git_provenance,
    resolve_active_dataset_run,
)
from research.runtime.hashing import canonical_sha256, sha256_file


SUPPORTED_COMMON_DATASETS = ("lfw", "survface", "rfw_custom")
SUPPORTED_RUN_TIERS = ("quick", "full")
SUPPORTED_MODEL_NAMES = ("arc", "ada", "mag", "edge")
SUPPORTED_ARTIFACT_STORAGE_MODES = ("results_only", "full")
MODEL_NAME_TO_FAMILY: Mapping[str, str] = {
    "arc": "arcface",
    "ada": "adaface",
    "mag": "magface",
    "edge": "edgeface",
}
DEFAULT_MODEL_PROFILES: Mapping[str, str] = {
    "arc": "arcface_ms1mv3_r100",
    "ada": "adaface_ms1mv3_r100",
    "mag": "magface_ms1mv2_iresnet100",
    "edge": "edgeface_webface12m_xs_gamma_06",
}
DEFAULT_MODEL_WEIGHT_PATHS: Mapping[str, str] = {
    "arc": "models/arcface/ms1mv3_r100_backbone.pth",
    "ada": "models/adaface/adaface_ir101_ms1mv3.ckpt",
    "mag": "models/magface/magface_ms1mv2.pth",
    "edge": "models/edgeface/edgeface_xs_gamma_06.pt",
}
QUICK_DATA_FRACTIONS: Mapping[str, float] = {
    "lfw": 0.10,
    "survface": 0.02,
    "rfw_custom": 0.10,
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
FORMAL_FPIR_TARGETS = (0.30, 0.20, 0.10, 0.05, 0.01, 0.001)
PAPER_OPERATING_POINTS = (0.01, 0.05, 0.10, 0.20, 0.30)
EXPLORATORY_FPIR_TARGETS = (0.0001,)
CALIBRATION_IDENTITY_COUNTS = (100, 500, 1000)
SURVFACE_QUICK_PROTOCOL_REBASE_CORRECTION_ID = (
    "survface_quick_protocol_index_rebase_v1"
)
RETRIEVAL_JOIN_GRAIN_CORRECTION_ID = "retrieval_multi_fpir_join_grain_v2"
REPRESENTATIVE_CASE_STREAMING_CORRECTION_ID = (
    "representative_case_streaming_memory_v1"
)
SALIENCY_ATOMIC_PUBLISH_CORRECTION_ID = (
    "saliency_population_atomic_publish_retry_v1"
)


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
class CommonModelPreparation:
    model_name: str
    model_profile: str
    family: str
    checkpoint_path: Path
    checkpoint_sha256: str
    model_uid: str
    model_spec_path: Path
    smoke_validation_status: str
    smoke_validation_path: Path | None

    def as_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_profile": self.model_profile,
            "family": self.family,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_uid": self.model_uid,
            "model_spec_path": str(self.model_spec_path),
            "smoke_validation_status": self.smoke_validation_status,
            "smoke_validation_path": (
                None
                if self.smoke_validation_path is None
                else str(self.smoke_validation_path)
            ),
        }


@dataclass(frozen=True)
class CommonExperimentPlan:
    project_root: Path
    dataset_id: str
    run_tier: str
    data_fraction: float
    quick_data_fractions: dict[str, float]
    quick_fraction_override: bool
    seed: int
    model_name: str
    model_profile: str
    model_uid: str | None
    model_checkpoint_path: Path | None
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
            "quick_data_fractions": self.quick_data_fractions,
            "quick_fraction_override": self.quick_fraction_override,
            "seed": self.seed,
            "model_name": self.model_name,
            "model_profile": self.model_profile,
            "model_uid": self.model_uid,
            "model_checkpoint_path": (
                None
                if self.model_checkpoint_path is None
                else str(self.model_checkpoint_path)
            ),
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


def _resolve_checkpoint_file(
    project_root: Path,
    path: str | Path,
) -> Path:
    candidate = Path(path)
    resolved = (
        candidate.expanduser().resolve()
        if candidate.is_absolute()
        else (project_root / candidate).resolve()
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {resolved}")
    return resolved


def _validated_model_name(model_name: str) -> str:
    resolved = str(model_name).strip().lower()
    if resolved not in SUPPORTED_MODEL_NAMES:
        raise ValueError(f"model_name must be one of {SUPPORTED_MODEL_NAMES}")
    return resolved


def _validated_quick_data_fractions(
    values: Mapping[str, float] | None,
) -> dict[str, float]:
    source = QUICK_DATA_FRACTIONS if values is None else values
    if not isinstance(source, Mapping):
        raise ValueError("quick_data_fractions must be a mapping")
    if set(source) != set(SUPPORTED_COMMON_DATASETS):
        raise ValueError(
            f"quick_data_fractions must contain exactly {SUPPORTED_COMMON_DATASETS}"
        )
    resolved: dict[str, float] = {}
    for dataset_id in SUPPORTED_COMMON_DATASETS:
        value = source[dataset_id]
        if isinstance(value, bool):
            raise ValueError("quick data fractions must be numbers in (0, 1]")
        try:
            fraction = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("quick data fractions must be numbers in (0, 1]") from exc
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("quick data fractions must be in (0, 1]")
        resolved[dataset_id] = fraction
    return resolved


def _source_snapshot(provenance: Mapping[str, object]) -> dict[str, object]:
    return {
        field: provenance.get(field)
        for field in SOURCE_SNAPSHOT_FIELDS
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _embed_with_target_shape(
    adapter: Any,
    aligned_faces: np.ndarray,
) -> tuple[Any, list[int]]:
    """Run one embedding smoke pass and capture the selected feature shape."""

    captured_shapes: list[list[int]] = []

    def capture_shape(_module: Any, _inputs: Any, output: Any) -> None:
        shape = getattr(output, "shape", None)
        if shape is None:
            raise TypeError("target_layer output must expose a tensor shape")
        captured_shapes.append([int(value) for value in shape])

    handle = adapter.target_layer.register_forward_hook(capture_shape)
    try:
        output = adapter.embed(aligned_faces)
    finally:
        handle.remove()
    if len(captured_shapes) != 1:
        raise ValueError(
            "target_layer must run exactly once during the model smoke pass"
        )
    target_shape = captured_shapes[0]
    if len(target_shape) != 4:
        raise ValueError(
            "target_layer output must have shape [batch, channels, height, width]"
        )
    return output, target_shape


def prepare_common_model_checkpoint(
    *,
    project_root: str | Path,
    model_name: str,
    model_profile: str | None = None,
    checkpoint_path: str | Path | None = None,
    step4_config_path: str | Path = DEFAULT_STEP4_CONFIG_PATH,
    run_smoke_validation: bool = True,
    smoke_device: str = "cuda",
    max_smoke_images: int = 8,
    seed: int = 42,
) -> CommonModelPreparation:
    """Register and optionally smoke-test one explicitly selected checkpoint.

    The short notebook aliases ``arc``, ``ada``, ``mag`` and ``edge`` select a profile,
    while the checkpoint path remains an explicit user-controlled input.  The
    resulting model UID pins the exact checkpoint hash and preprocessing
    contract for the experiment plan.
    """

    root = Path(project_root).resolve()
    name = _validated_model_name(model_name)
    config_path = _resolve_project_file(
        root,
        step4_config_path,
        label="Step 4 config",
    )
    config = load_step4_config(config_path)
    profiles = config.get("models", {}).get("profiles", {})
    profile_id = str(model_profile or DEFAULT_MODEL_PROFILES[name]).strip()
    if profile_id not in profiles:
        raise ValueError(
            f"unknown model_profile {profile_id!r}; available={sorted(profiles)}"
        )
    if profile_id in set(config["models"].get("blocked_profiles", ())):
        raise RuntimeError(f"blocked model profile cannot be prepared: {profile_id}")
    profile = profiles[profile_id]
    expected_family = MODEL_NAME_TO_FAMILY[name]
    actual_family = str(profile["family"])
    if actual_family != expected_family:
        raise ValueError(
            f"model_name {name!r} requires family {expected_family!r}, "
            f"but profile {profile_id!r} declares {actual_family!r}"
        )
    if not bool(profile.get("run_gradcam", False)):
        raise ValueError(
            f"profile {profile_id!r} is not enabled for the common Grad-CAM "
            "pipeline; select a run_gradcam=true profile"
        )

    selected_checkpoint = _resolve_checkpoint_file(
        root,
        checkpoint_path or DEFAULT_MODEL_WEIGHT_PATHS[name],
    )
    source_url = str(
        profile.get("checkpoint_source_url")
        or profile.get("checkpoint_source_page")
        or profile["implementation_repository"]
    ).strip()
    checkpoint = CheckpointProvenance.from_file(
        selected_checkpoint,
        source_url=source_url,
    )
    expected_checkpoint_sha256 = str(
        profile.get("expected_checkpoint_sha256") or ""
    ).strip().lower()
    if (
        expected_checkpoint_sha256
        and checkpoint.sha256 != expected_checkpoint_sha256
    ):
        raise ValueError(
            f"checkpoint SHA-256 mismatch for profile {profile_id!r}: "
            f"expected={expected_checkpoint_sha256}, actual={checkpoint.sha256}"
        )
    preprocessing = PreprocessingSpec(
        input_height=int(profile["preprocessing"]["input_size"][0]),
        input_width=int(profile["preprocessing"]["input_size"][1]),
        source_color_order=str(config["aligned_crops"]["source_color_order"]),
        model_color_order=str(profile["preprocessing"]["model_color_order"]),
        channel_mean=tuple(float(value) for value in profile["preprocessing"]["mean"]),
        channel_std=tuple(float(value) for value in profile["preprocessing"]["std"]),
    )
    spec = ModelSpec(
        family=actual_family,
        architecture=str(profile["architecture"]),
        training_dataset=str(profile["training_dataset"]),
        implementation_repository=str(profile["implementation_repository"]),
        checkpoint=checkpoint,
        preprocessing=preprocessing,
        target_layer=str(profile["target_layer"]),
        embedding_dim=int(profile["embedding_dim"]),
        module_factory=str(profile["loader_factory"]),
    )
    registry_root = root / config["models"]["registry_root"]
    registry_path = registry_root / f"{spec.model_uid}.json"
    if registry_path.is_file():
        try:
            write_model_spec(registry_path, spec)
        except FileExistsError:
            registry_path = registry_root / f"{model_spec_registry_stem(spec)}.json"
            write_model_spec(registry_path, spec)
    else:
        write_model_spec(registry_path, spec)

    validation_path = (
        root
        / config["models"]["validation_root"]
        / registry_path.stem
        / "smoke_summary.json"
    )
    smoke_status = "not_requested"
    expected_heatmap_size: list[int] | None = None
    if validation_path.is_file() or run_smoke_validation:
        expected_heatmap_size = [
            int(value)
            for value in config["gradcam"]["extraction"][
                "expected_heatmap_size"
            ]
        ]
        if len(expected_heatmap_size) != 2 or any(
            value < 1 for value in expected_heatmap_size
        ):
            raise ValueError(
                "gradcam expected_heatmap_size must be [height, width]"
            )
    if validation_path.is_file():
        existing = json.loads(validation_path.read_text(encoding="utf-8"))
        expected = {
            "model_uid": spec.model_uid,
            "profile_id": profile_id,
            "family": spec.family,
            "architecture": spec.architecture,
            "training_dataset": spec.training_dataset,
            "checkpoint_sha256": spec.checkpoint.sha256,
            "preprocess_hash": spec.preprocessing.preprocess_hash,
            "target_layer": spec.target_layer,
            "status": "validated",
        }
        mismatches = {
            key: {"expected": value, "actual": existing.get(key)}
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if "target_feature_shape" in existing:
            assert expected_heatmap_size is not None
            actual_spatial = list(existing["target_feature_shape"][-2:])
            if actual_spatial != expected_heatmap_size:
                mismatches["target_feature_shape"] = {
                    "expected_spatial": expected_heatmap_size,
                    "actual": existing["target_feature_shape"],
                }
        if mismatches:
            raise RuntimeError(
                "existing smoke validation does not match the selected model: "
                f"{mismatches}"
            )
        smoke_status = "reused_validated"
    elif run_smoke_validation:
        assert expected_heatmap_size is not None
        if (
            isinstance(max_smoke_images, bool)
            or int(max_smoke_images) != max_smoke_images
            or int(max_smoke_images) < 1
        ):
            raise ValueError("max_smoke_images must be a positive integer")
        smoke_input = resolve_smoke_input_batch(
            root,
            source_color_order=spec.preprocessing.source_color_order,
            lfw_manifest_path=(root / config["datasets"]["lfw"]["manifest_path"]),
            max_images=int(max_smoke_images),
            seed=int(seed),
        )
        adapter = create_pytorch_adapter_from_spec(
            spec,
            device=str(smoke_device),
        )
        output, target_feature_shape = _embed_with_target_shape(
            adapter,
            smoke_input.aligned_faces,
        )
        if target_feature_shape[-2:] != expected_heatmap_size:
            raise ValueError(
                f"profile {profile_id!r} target_layer {spec.target_layer!r} "
                f"produces spatial shape {target_feature_shape[-2:]}, expected "
                f"the common Grad-CAM grid {expected_heatmap_size}"
            )
        unit_norms = np.linalg.norm(output.normalized_embedding, axis=1)
        smoke_summary = {
            "model_uid": spec.model_uid,
            "model_name": name,
            "profile_id": profile_id,
            "family": spec.family,
            "architecture": spec.architecture,
            "training_dataset": spec.training_dataset,
            "model_spec_path": str(registry_path),
            "smoke_input": smoke_input.metadata,
            "checkpoint_sha256": spec.checkpoint.sha256,
            "preprocess_hash": spec.preprocessing.preprocess_hash,
            "sample_count": int(len(smoke_input.aligned_faces)),
            "raw_shape": list(output.raw_embedding.shape),
            "raw_norm_min": float(output.raw_norm.min()),
            "raw_norm_max": float(output.raw_norm.max()),
            "maximum_unit_norm_error": float(np.max(np.abs(unit_norms - 1.0))),
            "target_layer": spec.target_layer,
            "target_feature_shape": target_feature_shape,
            "expected_heatmap_size": expected_heatmap_size,
            "status": "validated",
        }
        _write_json_atomic(validation_path, smoke_summary)
        smoke_status = "validated_now"

    return CommonModelPreparation(
        model_name=name,
        model_profile=profile_id,
        family=actual_family,
        checkpoint_path=selected_checkpoint,
        checkpoint_sha256=spec.checkpoint.sha256,
        model_uid=spec.model_uid,
        model_spec_path=registry_path,
        smoke_validation_status=smoke_status,
        smoke_validation_path=(validation_path if validation_path.is_file() else None),
    )


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
    paper_operating_points = _numeric_tuple(
        calibration.get("paper_operating_points"),
        label="paper_operating_points",
    )
    if paper_operating_points != PAPER_OPERATING_POINTS:
        raise ValueError(
            f"paper operating points must be exactly {PAPER_OPERATING_POINTS}"
        )
    if calibration.get("reuse_search_scores_across_operating_points") is not True:
        raise ValueError("multi-FPIR evaluation must reuse search scores")
    if tuple(calibration.get("appendix_datasets", ())) != ("lfw", "survface"):
        raise ValueError("the common FPIR appendix must cover LFW and SurvFace")
    appendix_targets = _numeric_tuple(
        calibration.get("appendix_operating_points"),
        label="appendix_operating_points",
    )
    if appendix_targets != paper_operating_points:
        raise ValueError("appendix operating points must match paper points")

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
    if evaluation.get("report_realized_fpir") is not True:
        raise ValueError("realized FPIR reporting must remain enabled")
    if evaluation.get("report_denominator_and_error_count") is not True:
        raise ValueError("FPIR denominator and false-accept reporting is required")
    if evaluation.get("report_confidence_interval") is not True:
        raise ValueError("open-set confidence intervals are required")
    confidence = evaluation.get("confidence_intervals")
    if not isinstance(confidence, Mapping):
        raise ValueError("confidence_intervals must be a mapping")
    expected_confidence = {
        "binomial_rates": "wilson_95",
        "compressed_minus_origin": "paired_query_bootstrap_95",
        "bootstrap_seed": 42,
        "bootstrap_repeats": 2000,
    }
    if dict(confidence) != expected_confidence:
        raise ValueError(
            "confidence interval contract differs from Wilson/paired bootstrap v1"
        )

    rfw = contract.get("rfw")
    if not isinstance(rfw, Mapping):
        raise ValueError("RFW evaluation boundary must be a mapping")
    custom = rfw.get("custom")
    official = rfw.get("official")
    if not isinstance(custom, Mapping) or not isinstance(official, Mapping):
        raise ValueError("RFW custom and official contracts are required")
    if custom.get("official_protocol_claim") is not False:
        raise ValueError("RFW-Custom must not claim official protocol status")
    if custom.get("score_statistic") != "maximum_gallery_score":
        raise ValueError("RFW-Custom must use maximum_gallery_score")
    if custom.get("calibration_gallery_policy") != "evaluation_group_matched":
        raise ValueError(
            "RFW-Custom calibration gallery must match the evaluation gallery"
        )
    if official.get("open_set_metrics_forbidden") is not True:
        raise ValueError("RFW-Official must forbid open-set metrics")
    if rfw.get("edgeface_training_identity_overlap_status") != "UNKNOWN":
        raise ValueError("EdgeFace-RFW overlap status must remain UNKNOWN")
    if rfw.get("strict_unseen_identity_evidence_allowed") is not False:
        raise ValueError("RFW cannot be strict unseen-identity evidence")


def _effective_step4_config(
    base_config: Mapping[str, Any],
    *,
    project_root: Path,
    dataset_id: str,
    run_tier: str,
    data_fraction: float,
    quick_data_fractions: Mapping[str, float],
    quick_fraction_override: bool,
    seed: int,
    model_name: str | None,
    model_profile: str | None,
    model_uid: str | None,
    model_checkpoint_path: str | Path | None,
    contract_id: str,
    contract_sha256: str,
    artifact_storage_mode: str,
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
    profile = profiles[selected_profile]
    profile_family = str(profile["family"])
    if profile.get("run_gradcam") is False:
        raise ValueError(
            f"profile {selected_profile!r} is not enabled for the common "
            "Grad-CAM pipeline"
        )
    selected_model_name = (
        next(
            name
            for name, family in MODEL_NAME_TO_FAMILY.items()
            if family == profile_family
        )
        if model_name is None
        else _validated_model_name(model_name)
    )
    if MODEL_NAME_TO_FAMILY[selected_model_name] != profile_family:
        raise ValueError(
            f"model_name {selected_model_name!r} does not match profile "
            f"{selected_profile!r} family {profile_family!r}"
        )
    resolved_checkpoint_path = (
        None
        if model_checkpoint_path is None
        else _resolve_checkpoint_file(
            project_root,
            model_checkpoint_path,
        )
    )
    if model_uid is not None:
        profile["model_uid"] = str(model_uid)
        config["models"]["model_uid"] = str(model_uid)
    if resolved_checkpoint_path is not None:
        profile["checkpoint_path"] = str(resolved_checkpoint_path)
    storage_mode = str(artifact_storage_mode).strip().lower()
    if storage_mode not in SUPPORTED_ARTIFACT_STORAGE_MODES:
        raise ValueError(
            "artifact_storage_mode must be one of "
            f"{SUPPORTED_ARTIFACT_STORAGE_MODES}"
        )
    workflow = config.setdefault("workflow", {})
    workflow["artifact_storage_mode"] = storage_mode
    workflow["persist_large_join_artifacts"] = (
        storage_mode == "full"
    )
    gradcam = config.setdefault("gradcam", {})
    persistence = gradcam.setdefault("persistence", {})
    if storage_mode == "results_only":
        persistence.update(
            {
                "persist_normalized_heatmap": True,
                "persist_raw_cam": False,
                "persist_relu_cam": False,
                "persist_channel_weights": False,
                "persist_pass_b_embeddings": False,
                "persist_scalar_features": True,
                "persist_full_activations": False,
                "persist_full_gradients": False,
            }
        )
    execution.update(
        {
            "model_name": selected_model_name,
            "model_profile": selected_profile,
            "model_uid": None if model_uid is None else str(model_uid),
            "model_checkpoint_path": (
                None
                if resolved_checkpoint_path is None
                else str(resolved_checkpoint_path)
            ),
            "mode": "real",
            "data_fraction": data_fraction,
            "seed": seed,
            "execute_stage": True,
            "write_outputs": True,
            "overwrite": False,
            "allow_dirty": run_tier == "quick",
        }
    )
    config["orchestration"] = {
        "pipeline_id": "common_step4_gradcam_v1",
        "dataset_id": dataset_id,
        "run_tier": run_tier,
        "quick_data_fractions": dict(quick_data_fractions),
        "quick_fraction_override": quick_fraction_override,
        "evaluation_contract_id": contract_id,
        "evaluation_contract_sha256": contract_sha256,
        "comparison_contract_coverage": "partial",
        "artifact_storage_mode": storage_mode,
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
    model_name: str | None = None,
    model_profile: str | None = None,
    model_uid: str | None = None,
    model_checkpoint_path: str | Path | None = None,
    quick_data_fractions: Mapping[str, float] | None = None,
    artifact_storage_mode: str = "full",
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
    resolved_quick_fractions = _validated_quick_data_fractions(quick_data_fractions)
    quick_fraction_override = resolved_quick_fractions != dict(QUICK_DATA_FRACTIONS)
    data_fraction = (
        resolved_quick_fractions[dataset] if tier == "quick" else FULL_DATA_FRACTION
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
    configured_targets = tuple(
        float(value)
        for value in base_config.get("evaluation", {}).get(
            "reported_target_fpirs",
            (),
        )
    )
    paper_targets = tuple(
        float(value)
        for value in contract["calibration"]["paper_operating_points"]
    )
    if configured_targets != paper_targets:
        raise ValueError(
            "Step 4 reported_target_fpirs differ from evaluation contract: "
            f"{configured_targets} != {paper_targets}"
        )
    configured_rfw_policy = str(
        base_config.get("evaluation", {}).get(
            "rfw_custom_calibration_gallery_policy",
            "",
        )
    )
    contract_rfw_policy = str(
        contract.get("rfw", {})
        .get("custom", {})
        .get("calibration_gallery_policy", "")
    )
    if configured_rfw_policy != contract_rfw_policy:
        raise ValueError(
            "Step 4 RFW-Custom calibration gallery policy differs from the "
            f"evaluation contract: {configured_rfw_policy!r} != "
            f"{contract_rfw_policy!r}"
        )
    contract_id = str(contract["contract_id"])
    contract_sha256 = sha256_file(contract_path)
    source_provenance = inspect_git_provenance(
        root,
        run_root=root / base_config["run"]["root"],
    )
    effective_config = _effective_step4_config(
        base_config,
        project_root=root,
        dataset_id=dataset,
        run_tier=tier,
        data_fraction=data_fraction,
        quick_data_fractions=resolved_quick_fractions,
        quick_fraction_override=quick_fraction_override,
        seed=resolved_seed,
        model_name=model_name,
        model_profile=model_profile,
        model_uid=model_uid,
        model_checkpoint_path=model_checkpoint_path,
        contract_id=contract_id,
        contract_sha256=contract_sha256,
        artifact_storage_mode=artifact_storage_mode,
    )
    effective_config["orchestration"]["source_snapshot"] = (
        _source_snapshot(source_provenance)
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
    selected_model_name = str(effective_config["execution"]["model_name"])
    selected_profile = str(effective_config["execution"]["model_profile"])
    selected_model_uid = effective_config["execution"]["model_uid"]
    selected_checkpoint_value = effective_config["execution"]["model_checkpoint_path"]
    return CommonExperimentPlan(
        project_root=root,
        dataset_id=dataset,
        run_tier=tier,
        data_fraction=data_fraction,
        quick_data_fractions=resolved_quick_fractions,
        quick_fraction_override=quick_fraction_override,
        seed=resolved_seed,
        model_name=selected_model_name,
        model_profile=selected_profile,
        model_uid=(None if selected_model_uid is None else str(selected_model_uid)),
        model_checkpoint_path=(
            None
            if selected_checkpoint_value is None
            else Path(str(selected_checkpoint_value)).resolve()
        ),
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
        "four_search_space_main_table": "implemented",
        "pq_exhaustive_adc": "implemented",
        "search_space_clean_full_rerun": "validation_required",
        "survface_gradcam_faithfulness_controls": "implemented",
        "survface_gradcam_faithfulness_three_models": "implemented",
        "rfw_custom_open_set_protocol": (
            "implemented_gallery_matched_calibration_v2_pending_full_run"
        ),
        "three_open_set_dataset_comparison": "validation_required",
        "edgeface_rfw_identity_overlap": "unknown",
        "checkpoint_level_generalization_only": "implemented",
        "multi_fpir_score_reuse": "implemented",
        "probe_level_wilson_and_paired_delta_ci": "implemented",
        "lfw_survface_fpir_appendix": "implemented",
        "cross_dataset_calibration_transfer": "implemented_unvalidated_artifacts",
        "rfw_official_tar_far_eer": "implemented_unvalidated_full_run",
        "gradcam_faithfulness_clean_promotion": "validation_required",
        "repeated_latency_benchmark": "validation_required",
        "ivf_pq_system_ablation": "deferred_step7",
        "pgvector_ivfflat": "deferred_step7",
        "ann_parameter_sweep": "deferred_step7",
        "balancedface": "deferred_step7",
        "uncertainty_defer": "deferred_step7",
        "risk_stratified_query_experiments": "deferred_step7",
        "official_and_db_baseline_matrix": "deferred_step7",
        "calibration_100_500_1000": "proposed",
        "full_fpir_contract": "implemented",
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


def reuse_completed_run_for_plan(
    plan: CommonExperimentPlan,
    run_dir: str | Path,
) -> dict[str, object]:
    """Explicitly reuse one completed run after strict science-config checks.

    This is intentionally opt-in. It permits source-snapshot drift only when
    every non-snapshot Step 4 setting is unchanged and all phase artifacts are
    complete and checksum-valid.
    """

    selected = Path(run_dir).expanduser().resolve()
    base_root = (
        plan.project_root / plan.effective_step4_config["run"]["root"]
    ).resolve()
    try:
        selected.relative_to(base_root)
    except ValueError as exc:
        raise ValueError(
            f"completed run override must stay under {base_root}: {selected}"
        ) from exc
    manifest_path = selected / "run_manifest.json"
    completed_path = selected / "COMPLETED"
    if not manifest_path.is_file() or not completed_path.is_file():
        raise ValueError(
            "completed run override requires run_manifest.json and COMPLETED: "
            f"{selected}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(
            f"completed run override has non-completed status: {selected}"
        )
    config = manifest.get("config")
    if not isinstance(config, Mapping) or not isinstance(
        config.get("step4"),
        Mapping,
    ):
        raise ValueError("completed run override is missing the Step 4 config")
    frozen_step4 = config["step4"]
    if _config_without_storage_policy(frozen_step4) != (
        _config_without_storage_policy(plan.effective_step4_config)
    ):
        raise ValueError(
            "completed run override differs from the selected science config "
            "outside source snapshot and artifact storage policy"
        )
    run = RunStore.open(selected)
    reused_stages: list[dict[str, object]] = []
    for stage in STEP4_PIPELINE_STAGES[2:]:
        if stage.phase_name is None:
            continue
        if not _completed_phase(selected, stage.phase_name):
            raise ValueError(
                "completed run override is missing phase: "
                f"{stage.phase_name}"
            )
        run.verify_phase_artifacts(stage.phase_name)
        reused_stages.append(
            {
                "stage_id": stage.stage_id,
                "status": "reused_completed_override",
            }
        )
    frozen_snapshot = frozen_step4.get("orchestration", {}).get(
        "source_snapshot"
    )
    current_snapshot = plan.effective_step4_config.get(
        "orchestration",
        {},
    ).get("source_snapshot")
    return {
        "status": "already_completed",
        "run_id": str(manifest["run_id"]),
        "run_dir": str(selected),
        "plan_id": plan.plan_id,
        "materialization": [],
        "stages": reused_stages,
        "explicit_completed_override": True,
        "frozen_source_snapshot": frozen_snapshot,
        "current_source_snapshot": current_snapshot,
        "message": (
            "Explicit completed-run override verified; non-snapshot science "
            "config and all phase artifact checks passed."
        ),
    }


def _active_matching_run(
    plan: CommonExperimentPlan,
) -> RunStore | None:
    run = _active_dataset_run(plan)
    if run is None:
        return None
    if run.config.get("step4") != plan.effective_step4_config:
        raise RuntimeError(
            "a different incomplete dataset run is active. Complete or "
            "selectively clean that run before executing this plan: "
            f"{run.run_dir}"
        )
    return run


def _active_dataset_run(
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
    return RunStore.open(run_dir)


def _config_without_source_snapshot(
    config: Mapping[str, object],
) -> dict[str, object]:
    comparable = deepcopy(dict(config))
    orchestration = comparable.get("orchestration")
    if isinstance(orchestration, dict):
        orchestration.pop("source_snapshot", None)
    return comparable


def _latest_phase_attempt_manifest(
    run: RunStore,
    phase_name: str,
) -> dict[str, object] | None:
    attempts_dir = run.run_dir / "phases" / phase_name / "attempts"
    candidates: list[tuple[int, dict[str, object]]] = []
    for path in attempts_dir.glob("A*/phase_manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            attempt = int(payload["attempt"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        candidates.append((attempt, payload))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _is_known_survface_quick_protocol_index_failure(
    run: RunStore,
    plan: CommonExperimentPlan,
) -> bool:
    existing_config = run.config.get("step4")
    if not isinstance(existing_config, Mapping):
        return False
    if plan.dataset_id != "survface" or plan.run_tier != "quick":
        return False
    if (
        _config_without_source_snapshot(existing_config)
        != _config_without_source_snapshot(plan.effective_step4_config)
    ):
        return False
    required_completed = (
        "00_source_and_model_freeze",
        "01_origin_embedding_and_target_templates",
        "02_population_gradcam_extraction",
        "03_saliency_feature_validation",
    )
    if not all(
        _completed_phase(run.run_dir, phase_name)
        for phase_name in required_completed
    ):
        return False
    latest = _latest_phase_attempt_manifest(
        run,
        "04_step2_compression_characterization",
    )
    if latest is None or latest.get("status") != "failed":
        return False
    failure = latest.get("failure")
    if not isinstance(failure, Mapping):
        return False
    return str(failure.get("message", "")).endswith(
        "protocol_index must be unique and contiguous from 0"
    )


def _is_known_retrieval_join_grain_failure(
    run: RunStore,
    plan: CommonExperimentPlan,
) -> bool:
    existing_config = run.config.get("step4")
    if not isinstance(existing_config, Mapping):
        return False
    if (
        _config_without_source_snapshot(existing_config)
        != _config_without_source_snapshot(plan.effective_step4_config)
    ):
        return False
    required_completed = (
        "00_source_and_model_freeze",
        "01_origin_embedding_and_target_templates",
        "02_population_gradcam_extraction",
        "03_saliency_feature_validation",
        "04_step2_compression_characterization",
    )
    if not all(
        _completed_phase(run.run_dir, phase_name)
        for phase_name in required_completed
    ):
        return False
    latest = _latest_phase_attempt_manifest(
        run,
        "05_saliency_compression_join",
    )
    if latest is None or latest.get("status") != "failed":
        return False
    failure = latest.get("failure")
    if not isinstance(failure, Mapping):
        return False
    message = str(failure.get("message", ""))
    if not message.startswith(
        "retrieval_sensitivity rows are not unique by"
    ):
        return False
    retrieval_path = (
        run.run_dir
        / "artifacts"
        / "step2_workflow"
        / "retrieval_metrics.csv"
    )
    if not retrieval_path.is_file():
        return False
    columns = pd.read_csv(retrieval_path, nrows=0).columns
    return "search_mode" in columns or "target_fpir" in columns


def _is_known_saliency_atomic_publish_failure(
    run: RunStore,
    plan: CommonExperimentPlan,
) -> bool:
    existing_config = run.config.get("step4")
    if not isinstance(existing_config, Mapping):
        return False
    if (
        _config_without_source_snapshot(existing_config)
        != _config_without_source_snapshot(plan.effective_step4_config)
    ):
        return False
    if not all(
        _completed_phase(run.run_dir, phase_name)
        for phase_name in (
            "00_source_and_model_freeze",
            "01_origin_embedding_and_target_templates",
        )
    ):
        return False
    if any(
        _completed_phase(run.run_dir, phase_name)
        for phase_name in (
            "03_saliency_feature_validation",
            "04_step2_compression_characterization",
            "05_saliency_compression_join",
            "06_representative_cases",
        )
    ):
        return False
    latest = _latest_phase_attempt_manifest(
        run,
        "02_population_gradcam_extraction",
    )
    if latest is None or latest.get("status") != "failed":
        return False
    failure = latest.get("failure")
    if not isinstance(failure, Mapping):
        return False
    message = str(failure.get("message", ""))
    traceback_text = str(failure.get("traceback", ""))
    if str(failure.get("type", "")) != "PermissionError":
        return False
    if not all(
        marker in f"{message}\n{traceback_text}"
        for marker in (
            "[WinError 5]",
            "write_population_saliency_artifact",
            ".saliency_population.",
        )
    ):
        return False
    saliency_path = (
        run.run_dir
        / "artifacts"
        / "step2_workflow"
        / "saliency_population"
    )
    return not saliency_path.exists()


def _config_without_storage_policy(
    config: Mapping[str, object],
) -> dict[str, object]:
    comparable = _config_without_source_snapshot(config)
    orchestration = comparable.get("orchestration")
    if isinstance(orchestration, dict):
        orchestration.pop("artifact_storage_mode", None)
    workflow = comparable.get("workflow")
    if isinstance(workflow, dict):
        for key in (
            "artifact_storage_mode",
            "persist_large_join_artifacts",
            "representative_case_candidates_path",
        ):
            workflow.pop(key, None)
    gradcam = comparable.get("gradcam")
    if isinstance(gradcam, dict):
        gradcam.pop("persistence", None)
    return comparable


def _is_known_representative_case_memory_failure(
    run: RunStore,
    plan: CommonExperimentPlan,
) -> bool:
    existing_config = run.config.get("step4")
    if not isinstance(existing_config, Mapping):
        return False
    if _config_without_storage_policy(existing_config) != (
        _config_without_storage_policy(plan.effective_step4_config)
    ):
        return False
    required_completed = (
        "00_source_and_model_freeze",
        "01_origin_embedding_and_target_templates",
        "02_population_gradcam_extraction",
        "03_saliency_feature_validation",
        "04_step2_compression_characterization",
        "05_saliency_compression_join",
    )
    if not all(
        _completed_phase(run.run_dir, phase_name)
        for phase_name in required_completed
    ):
        return False
    latest = _latest_phase_attempt_manifest(
        run,
        "06_representative_case_visualization",
    )
    if latest is not None:
        if latest.get("status") != "failed":
            return False
        failure = latest.get("failure")
        if not isinstance(failure, Mapping):
            return False
        message = str(failure.get("message", "")).lower()
        if "out of memory" not in message:
            return False
    # The legacy finalizer loaded both CSVs before opening the phase context.
    # An OOM at that point therefore leaves no phase-06 attempt manifest.
    workflow_root = run.run_dir / "artifacts" / "step2_workflow"
    return all(
        (workflow_root / name).is_file()
        for name in (
            "saliency_geometry_join.csv",
            "saliency_retrieval_join.csv",
        )
    )


def _recorded_step4_config_path(run: RunStore) -> Path:
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    candidates = [
        entry
        for entry in manifest.get("inputs", [])
        if entry.get("role") == "step4_config"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "the active run must contain exactly one frozen step4_config input"
        )
    entry = candidates[0]
    path = Path(str(entry["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen Step 4 config is missing: {path}")
    if sha256_file(path) != str(entry.get("sha256")):
        raise ValueError(f"frozen Step 4 config hash mismatch: {path}")
    recorded_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(recorded_config, Mapping):
        raise ValueError(f"frozen Step 4 config must be a mapping: {path}")
    if recorded_config != run.config.get("step4"):
        raise ValueError(
            "frozen Step 4 config content differs from the active run config"
        )
    return path


def _resolve_execution_run(
    plan: CommonExperimentPlan,
    *,
    current_config_path: Path,
    start_new_run: bool = False,
) -> tuple[RunStore | None, Path, dict[str, object] | None]:
    run = _active_dataset_run(plan)
    if run is None:
        return None, current_config_path, None
    existing_config = run.config.get("step4")
    if existing_config == plan.effective_step4_config:
        return run, current_config_path, None
    retrieval_join_failure = _is_known_retrieval_join_grain_failure(
        run,
        plan,
    )
    if start_new_run and not retrieval_join_failure:
        return None, current_config_path, None
    quick_protocol_failure = (
        False
        if retrieval_join_failure
        else _is_known_survface_quick_protocol_index_failure(run, plan)
    )
    representative_case_failure = (
        False
        if retrieval_join_failure
        else _is_known_representative_case_memory_failure(run, plan)
    )
    saliency_publish_failure = (
        False
        if any(
            (
                retrieval_join_failure,
                quick_protocol_failure,
                representative_case_failure,
            )
        )
        else _is_known_saliency_atomic_publish_failure(run, plan)
    )
    if not any(
        (
            quick_protocol_failure,
            retrieval_join_failure,
            representative_case_failure,
            saliency_publish_failure,
        )
    ):
        raise RuntimeError(
            "a different incomplete dataset run is active. Complete or "
            "selectively clean that run before executing this plan: "
            f"{run.run_dir}"
        )

    frozen_config_path = _recorded_step4_config_path(run)
    existing_orchestration = existing_config.get("orchestration", {})
    if quick_protocol_failure:
        correction_id = SURVFACE_QUICK_PROTOCOL_REBASE_CORRECTION_ID
        reason = (
            "resume the known SurvFace Quick protocol_index gap failure "
            "without rewriting completed phase artifacts"
        )
    elif retrieval_join_failure:
        correction_id = RETRIEVAL_JOIN_GRAIN_CORRECTION_ID
        reason = (
            "resume the known retrieval search-mode/target-FPIR join-grain failure "
            "without rewriting completed phases 00-04"
        )
    elif representative_case_failure:
        correction_id = REPRESENTATIVE_CASE_STREAMING_CORRECTION_ID
        reason = (
            "resume the known representative-case full-CSV memory failure "
            "with bounded streaming selection and without rewriting phases 00-05"
        )
    else:
        correction_id = SALIENCY_ATOMIC_PUBLISH_CORRECTION_ID
        reason = (
            "resume the known Windows saliency artifact atomic-publish lock "
            "after preserving completed phases 00-01"
        )
    correction_context = {
        "correction_id": correction_id,
        "reason": reason,
        "frozen_source_snapshot": existing_orchestration.get(
            "source_snapshot"
        ),
        "resume_source_snapshot": plan.effective_step4_config.get(
            "orchestration", {}
        ).get("source_snapshot"),
        "resume_plan_id": plan.plan_id,
        "frozen_config_path": str(frozen_config_path),
    }
    run.record_event(
        "source_correction_resume_authorized",
        **correction_context,
    )
    return run, frozen_config_path, correction_context


def _call_step4_stage(
    stage_id: str,
    *,
    config_path: Path,
    plan: CommonExperimentPlan,
    progress: Callable[[str, dict[str, object]], None] | None,
    execution_context: Mapping[str, object] | None = None,
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
            execution_context=execution_context,
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

    This dispatcher covers the Step 4 Grad-CAM workflow and the four Step 2
    search spaces, including exhaustive PQ ADC.  The complete calibration
    matrix and repeated systems benchmark remain separate validation work.
    """

    if execution_acknowledged is not True:
        raise RuntimeError("common execution requires explicit local acknowledgement")
    preflight = inspect_common_experiment_plan(plan)
    if not preflight["ready_to_execute_pipeline"]:
        raise RuntimeError("local preflight failed; inspect readiness before execution")
    current_config_path = materialize_effective_step4_config(plan)
    run, config_path, execution_context = _resolve_execution_run(
        plan,
        current_config_path=current_config_path,
        start_new_run=start_new_run,
    )
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
            start_new_run=start_new_run,
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
                start_new_run=False,
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
            execution_context=(
                execution_context
                if stage.stage_id == "compression_characterization"
                else None
            ),
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
        "source_correction_resume": execution_context,
        "materialization": materialized,
        "stages": stage_results,
        "comparison_paper_eligible": False,
        "comparison_limitation": (
            "The common Step 4 dispatcher and exhaustive PQ ADC are implemented, "
            "but the complete calibration/FPIR matrix, repeated latency benchmark, "
            "and same-commit dual-dataset clean full reruns remain outstanding."
        ),
    }
