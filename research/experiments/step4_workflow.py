from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import yaml

from research.embeddings import (
    create_pytorch_adapter_from_spec,
    select_model_spec_by_profile,
)
from research.evaluation import (
    WEIGHTED_RERANK_ALGORITHM_VERSION,
    WEIGHTED_RERANK_STRATEGY,
    annotate_compression_lineage,
    saliency_geometry_associations,
    saliency_retrieval_associations,
    stream_join_population_saliency_with_compression,
    stream_join_population_saliency_with_retrieval,
    stream_select_population_representative_cases,
)
from research.experiments.scope import (
    ExperimentScope,
    select_manifest_fraction,
    select_open_set_protocol_fraction,
)
from research.experiments.step2_compression import (
    characterize_step2_compression,
    characterize_step2_rfw_custom_compression,
    characterize_step2_survface_compression,
)
from research.datasets.rfw_custom import select_rfw_custom_protocol_fraction
from research.experiments.step4_datasets import (
    Step4DatasetSpec,
    load_step4_source_manifest,
    resolve_step4_dataset_spec,
    select_step4_saliency_sample_mask,
)
from research.experiments.rfw_custom_pipeline import (
    materialize_rfw_custom_aligned_bundle,
)
from research.explainability.gradcam import (
    build_origin_top1_gallery_templates,
    extract_population_gradcam,
    materialize_landmark_region_bundle,
    prepare_population_saliency_inputs,
    read_landmark_region_bundle,
    read_population_heatmaps,
    read_population_saliency_features,
    read_prepared_population_artifact,
    write_population_saliency_artifact,
    write_prepared_population_artifact,
)
from research.preprocessing.aligned_crops import (
    materialize_aligned_crops,
    validate_aligned_crop_bundle,
)
from research.protocols import (
    build_survface_official_protocol,
    rebase_survface_protocol_subset_indexes,
)
from research.runtime import (
    RunStore,
    inspect_git_provenance,
    resolve_active_dataset_run,
    resolve_or_create_dataset_run_root,
)
from research.runtime.hashing import sha256_file


ProgressCallback = Callable[[str, dict[str, object]], None]
STEP4_JOIN_CHUNK_ROWS = 100_000
STEP4_BOOTSTRAP_BATCH_SIZE = 4
SOURCE_SNAPSHOT_FIELDS = (
    "commit",
    "branch",
    "dirty",
    "working_tree_diff_sha256",
    "untracked_content_sha256",
)


def load_step4_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Step 4 config root must be a mapping")
    return config


def _git_source_readiness(
    config: dict[str, Any],
    git_provenance: dict[str, object],
) -> dict[str, object]:
    git_clean = not bool(git_provenance["dirty"])
    allow_dirty = bool(config["execution"].get("allow_dirty", False))
    expected_source_snapshot = config.get("orchestration", {}).get(
        "source_snapshot"
    )
    current_source_snapshot = {
        field: git_provenance.get(field)
        for field in SOURCE_SNAPSHOT_FIELDS
    }
    return {
        "git_clean": git_clean,
        "allow_dirty": allow_dirty,
        "git_policy_satisfied": git_clean or allow_dirty,
        "source_snapshot_matches": (
            expected_source_snapshot is None
            or expected_source_snapshot == current_source_snapshot
        ),
        "expected_source_snapshot": expected_source_snapshot,
        "current_source_snapshot": current_source_snapshot,
    }


def inspect_step4_readiness(
    config_path: str | Path,
    *,
    project_root: str | Path,
    dataset_id: str,
) -> dict[str, object]:
    """Perform read-only local readiness checks; no GitHub service is involved."""

    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_step4_config(config_file)
    spec = resolve_step4_dataset_spec(
        config,
        project_root=root,
        dataset_id=dataset_id,
    )
    source = load_step4_source_manifest(spec)
    aligned_complete = (spec.aligned_bundle_dir / "_SUCCESS").is_file()
    aligned_valid = False
    aligned_error: str | None = None
    if aligned_complete:
        try:
            validate_aligned_crop_bundle(
                spec.aligned_bundle_dir,
                dataset_id=spec.dataset_id,
                expected_source_count=len(source),
                preprocessing_mode=spec.preprocessing_mode,
                require_full_coverage=spec.require_full_coverage,
            )
            aligned_valid = True
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            aligned_error = str(exc)
    profile_id = str(config["execution"]["model_profile"])
    profile = config["models"]["profiles"][profile_id]
    model_path, model_spec = select_model_spec_by_profile(
        root / config["models"]["registry_root"],
        profile_id=profile_id,
        profile_config=profile,
        verify_checkpoint=True,
    )
    import onnxruntime as ort
    import torch

    git_provenance = inspect_git_provenance(
        root,
        run_root=root / config["run"]["root"],
    )
    git_readiness = _git_source_readiness(config, git_provenance)
    required_provider = str(
        config["aligned_crops"]["required_primary_provider"]
    )
    available_providers = tuple(ort.get_available_providers())
    cuda_available = bool(torch.cuda.is_available())
    gates = {
        name: bool(config["execution"][name])
        for name in ("execute_stage", "write_outputs", "overwrite")
    }
    checks = {
        **git_readiness,
        "git_provenance": git_provenance,
        "cuda_available": cuda_available,
        "required_onnx_provider_available": required_provider
        in available_providers,
        "model_spec_verified": model_path.is_file(),
        "source_manifest_rows": int(len(source)),
        "aligned_bundle_complete": aligned_complete,
        "aligned_bundle_valid": aligned_valid,
        "aligned_bundle_error": aligned_error,
        "landmark_region_bundle_complete": (
            spec.landmark_region_bundle_dir / "_SUCCESS"
        ).is_file(),
        "execution_gates": gates,
    }
    ready_to_materialize = bool(
        checks["git_policy_satisfied"]
        and checks["source_snapshot_matches"]
        and checks["cuda_available"]
        and checks["required_onnx_provider_available"]
        and checks["model_spec_verified"]
    )
    ready_to_run = bool(
        ready_to_materialize
        and checks["aligned_bundle_valid"]
        and checks["landmark_region_bundle_complete"]
    )
    return {
        "dataset_id": spec.dataset_id,
        "model_uid": model_spec.model_uid,
        "model_spec_path": str(model_path),
        "cuda_device": (
            torch.cuda.get_device_name(0) if cuda_available else None
        ),
        "available_onnx_providers": list(available_providers),
        "checks": checks,
        "ready_to_materialize": ready_to_materialize,
        "ready_to_run_experiment": ready_to_run,
    }


def _scope(config: dict[str, Any]) -> ExperimentScope:
    execution = config["execution"]
    return ExperimentScope(
        mode=str(execution["mode"]),
        data_fraction=float(execution["data_fraction"]),
        seed=int(execution["seed"]),
    )


def select_step4_source_manifest(
    source: pd.DataFrame,
    *,
    dataset_id: str,
    scope: ExperimentScope,
) -> pd.DataFrame:
    if scope.is_full_dataset:
        return source.copy().reset_index(drop=True)
    if dataset_id == "rfw_custom":
        return select_rfw_custom_protocol_fraction(
            source,
            data_fraction=scope.data_fraction,
            seed=scope.seed,
        ).reset_index(drop=True)
    if dataset_id == "survface":
        official_roles = {
            "gallery",
            "registered_probe",
            "unknown_unknown_probe",
        }
        official_mask = source["protocol_role"].astype(str).isin(official_roles)
        official_protocol = build_survface_official_protocol(
            source.loc[official_mask].copy()
        )
        selected_protocol = select_open_set_protocol_fraction(
            official_protocol,
            scope,
            namespace="step4:survface:official",
        )
        official_selected = pd.concat(
            [
                selected_protocol.gallery,
                selected_protocol.registered_probes,
                selected_protocol.unknown_unknown_probes,
            ],
            ignore_index=True,
        )
        official_selected = rebase_survface_protocol_subset_indexes(
            official_selected
        )
        training_parts = [
            select_manifest_fraction(
                group,
                scope,
                namespace=f"step4:survface:training:{split}",
            )
            for split, group in source.loc[~official_mask].groupby(
                "split",
                sort=True,
            )
        ]
        training_selected = pd.concat(training_parts, ignore_index=True)
        selected = pd.concat(
            [training_selected, official_selected],
            ignore_index=True,
        )
        selected_ids = set(selected["image_id"].astype(str))
        source_order = source.loc[
            source["image_id"].astype(str).isin(selected_ids)
        ].copy()
        protocol_columns = official_selected[
            ["image_id", "source_protocol_index", "protocol_index"]
        ].copy()
        source_order = source_order.drop(
            columns=["source_protocol_index"],
            errors="ignore",
        ).merge(
            protocol_columns,
            on="image_id",
            how="left",
            suffixes=("_source", ""),
            validate="one_to_one",
        )
        official_selected_mask = source_order["protocol_role"].astype(str).isin(
            official_roles
        )
        source_order.loc[
            ~official_selected_mask,
            "protocol_index",
        ] = source_order.loc[
            ~official_selected_mask,
            "protocol_index_source",
        ]
        return source_order.drop(
            columns=["protocol_index_source"]
        ).reset_index(drop=True)
    selected_parts = [
        select_manifest_fraction(
            group,
            scope,
            namespace=f"step4:{dataset_id}:{split}",
        )
        for split, group in source.groupby("split", sort=True)
    ]
    return pd.concat(selected_parts, ignore_index=True)


def _freeze_selected_manifest(
    source: pd.DataFrame,
    aligned_index: pd.DataFrame,
    failed_index: pd.DataFrame,
    *,
    dataset_id: str,
    scope: ExperimentScope,
) -> pd.DataFrame:
    selected = select_step4_source_manifest(
        source,
        dataset_id=dataset_id,
        scope=scope,
    )
    selected["image_id"] = selected["image_id"].astype(str)
    aligned = aligned_index.copy()
    aligned["sample_id"] = aligned["sample_id"].astype(str)
    selected = selected.merge(
        aligned[
            [
                "sample_id",
                "aligned_face_index",
                "aligned_content_sha256",
            ]
        ].rename(columns={"sample_id": "image_id"}),
        on="image_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = selected.loc[selected["_merge"] != "both"].copy()
    if not missing.empty:
        failed_ids = set(failed_index["sample_id"].astype(str))
        missing_ids = set(missing["image_id"].astype(str))
        if not missing_ids.issubset(failed_ids):
            raise ValueError("alignment exclusions are not fully recorded")
        if dataset_id == "survface":
            official_roles = {
                "gallery",
                "registered_probe",
                "unknown_unknown_probe",
            }
            failed_official = missing["protocol_role"].astype(str).isin(
                official_roles
            )
            if failed_official.any():
                examples = missing.loc[failed_official, "image_id"].head(5).tolist()
                raise RuntimeError(
                    "SurvFace official protocol has alignment failures; "
                    f"no fallback is allowed: {examples}"
                )
    selected = (
        selected.loc[selected["_merge"] == "both"]
        .drop(columns="_merge")
        .sort_values("aligned_face_index", kind="stable")
        .reset_index(drop=True)
        .rename(columns={"image_id": "sample_id"})
    )
    selection_digest = hashlib.sha256(
        selected.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    selected["template_scope_id"] = (
        selected["split"].astype(str) + ":" + selection_digest[:16]
    )
    return selected


def _aligned_face_selection(
    aligned_faces: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    if len(indices) == 0:
        raise ValueError("selected aligned face indices must not be empty")
    expected = np.arange(int(indices[0]), int(indices[0]) + len(indices))
    if np.array_equal(indices, expected):
        return aligned_faces[int(indices[0]) : int(indices[-1]) + 1]
    return np.asarray(aligned_faces[indices], dtype=np.uint8)


def _write_csv(path: Path, frame: pd.DataFrame, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def _latest_completed_phase_counts(
    run: RunStore,
    phase_name: str,
) -> dict[str, int]:
    attempts_dir = run.run_dir / "phases" / phase_name / "attempts"
    candidates: list[tuple[int, dict[str, Any]]] = []
    for path in sorted(attempts_dir.glob("A*/phase_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        candidates.append((int(payload["attempt"]), payload))
    if not candidates:
        raise RuntimeError(f"no completed attempt for prerequisite phase {phase_name}")
    _, latest = max(candidates, key=lambda item: item[0])
    raw_counts = latest.get("details", {}).get("counts", {})
    if not isinstance(raw_counts, dict):
        raise ValueError(f"{phase_name} counts must be a mapping")
    return {str(key): int(value) for key, value in raw_counts.items()}


def _scoped_progress(
    progress: ProgressCallback | None,
    scope: str,
) -> ProgressCallback | None:
    if progress is None:
        return None

    def report(message: str, details: dict[str, object]) -> None:
        progress(f"{scope} {message}", details)

    return report


def _current_git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        raise RuntimeError(
            "cannot resolve the current Git commit for phase provenance"
        )
    return commit


def _require_execution_acknowledgement(
    execution_acknowledged: bool,
) -> None:
    if execution_acknowledged is not True:
        raise RuntimeError(
            "Step 4 execution requires an explicit local acknowledgement"
        )


def _step4_run_config(
    config: dict[str, Any],
    *,
    dataset_id: str,
    model_uid: str,
    scope: ExperimentScope,
    overwrite: bool,
    execution_source: str,
) -> dict[str, object]:
    return {
        "step4": config,
        "dataset_id": dataset_id,
        "model_uid": model_uid,
        "scope": scope.as_dict(),
        "effective_execution": {
            "acknowledged": True,
            "write_outputs": True,
            "overwrite_existing": overwrite,
            "source": execution_source,
        },
    }


def _open_step4_run(
    config: dict[str, Any],
    *,
    project_root: Path,
    dataset_id: str,
) -> tuple[RunStore, Path, Step4DatasetSpec]:
    spec = resolve_step4_dataset_spec(
        config,
        project_root=project_root,
        dataset_id=dataset_id,
    )
    run_dir = resolve_active_dataset_run(
        project_root / config["run"]["root"],
        dataset_id=spec.dataset_id,
        directory_template=config["run"]["dataset_date_dir_template"],
    )
    run = RunStore.open(run_dir)
    if str(run.config.get("dataset_id")) != spec.dataset_id:
        raise ValueError("active Step 4 run dataset does not match the notebook")
    if run.config.get("step4") != config:
        raise ValueError(
            "active Step 4 run config differs from the tracked Step 4 config"
        )
    workflow_root = run.run_dir / config["workflow"]["artifact_subdir"]
    if not workflow_root.is_dir():
        raise FileNotFoundError(workflow_root)
    return run, workflow_root, spec


def _write_json(
    path: Path,
    payload: dict[str, object],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def materialize_step4_aligned_crops(
    config_path: str | Path,
    *,
    project_root: str | Path,
    dataset_id: str,
    execution_acknowledged: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Materialize only the dataset-specific aligned-crop prerequisite."""

    _require_execution_acknowledgement(execution_acknowledged)
    root = Path(project_root).resolve()
    config = load_step4_config(config_path)
    spec = resolve_step4_dataset_spec(
        config,
        project_root=root,
        dataset_id=dataset_id,
    )
    success = spec.aligned_bundle_dir / "_SUCCESS"
    source = load_step4_source_manifest(spec)
    if success.is_file():
        validate_aligned_crop_bundle(
            spec.aligned_bundle_dir,
            dataset_id=spec.dataset_id,
            expected_source_count=len(source),
            preprocessing_mode=spec.preprocessing_mode,
            require_full_coverage=spec.require_full_coverage,
        )
    else:
        if spec.dataset_id == "rfw_custom":
            if (
                len(spec.manifest_paths) != 1
                or spec.aligned_bin_archive_path is None
                or spec.source_archive_sha256 is None
                or spec.aligned_bin_archive_sha256 is None
            ):
                raise ValueError("RFW custom aligned source contract is incomplete")
            materialize_rfw_custom_aligned_bundle(
                source,
                project_root=root,
                jpg_archive_path=spec.manifest_paths[0],
                aligned_bin_archive_path=spec.aligned_bin_archive_path,
                output_dir=spec.aligned_bundle_dir,
                expected_jpg_archive_sha256=spec.source_archive_sha256,
                expected_aligned_bin_archive_sha256=(
                    spec.aligned_bin_archive_sha256
                ),
                dataset_id=spec.dataset_id,
                preprocessing_mode=spec.preprocessing_mode,
                overwrite=bool(config["execution"]["overwrite"]),
                progress=progress,
            )
        else:
            materialize_aligned_crops(
                source,
                project_root=root,
                output_dir=spec.aligned_bundle_dir,
                dataset_id=spec.dataset_id,
                providers=tuple(config["aligned_crops"]["providers"]),
                overwrite=bool(config["execution"]["overwrite"]),
                preprocessing_mode=spec.preprocessing_mode,
                require_full_coverage=spec.require_full_coverage,
                progress=progress,
            )
        validate_aligned_crop_bundle(
            spec.aligned_bundle_dir,
            dataset_id=spec.dataset_id,
            expected_source_count=len(source),
            preprocessing_mode=spec.preprocessing_mode,
            require_full_coverage=spec.require_full_coverage,
        )
    return {
        "dataset_id": spec.dataset_id,
        "aligned_bundle_dir": str(spec.aligned_bundle_dir),
        "complete": success.is_file(),
    }


def materialize_step4_landmark_regions(
    config_path: str | Path,
    *,
    project_root: str | Path,
    dataset_id: str,
    execution_acknowledged: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Materialize only the 106-point landmark-region prerequisite."""

    _require_execution_acknowledgement(execution_acknowledged)
    root = Path(project_root).resolve()
    config = load_step4_config(config_path)
    spec = resolve_step4_dataset_spec(
        config,
        project_root=root,
        dataset_id=dataset_id,
    )
    if not (spec.aligned_bundle_dir / "_SUCCESS").is_file():
        raise FileNotFoundError(
            "aligned-crop stage must complete before landmark materialization: "
            f"{spec.aligned_bundle_dir}"
        )
    source = load_step4_source_manifest(spec)
    validate_aligned_crop_bundle(
        spec.aligned_bundle_dir,
        dataset_id=spec.dataset_id,
        expected_source_count=len(source),
        preprocessing_mode=spec.preprocessing_mode,
        require_full_coverage=spec.require_full_coverage,
    )
    success = spec.landmark_region_bundle_dir / "_SUCCESS"
    if not success.is_file():
        materialize_landmark_region_bundle(
            spec.aligned_bundle_dir,
            output_dir=spec.landmark_region_bundle_dir,
            dataset_id=spec.dataset_id,
            model_path=config["gradcam"]["regions"].get("landmark_model_path"),
            providers=tuple(config["aligned_crops"]["providers"]),
            overwrite=bool(config["execution"]["overwrite"]),
            progress=progress,
        )
    return {
        "dataset_id": spec.dataset_id,
        "landmark_region_bundle_dir": str(
            spec.landmark_region_bundle_dir
        ),
        "complete": success.is_file(),
    }


def freeze_step4_source_and_model(
    config_path: str | Path,
    *,
    project_root: str | Path,
    dataset_id: str,
    execution_acknowledged: bool = False,
    execution_source: str = "dataset_notebook",
) -> dict[str, object]:
    """Freeze one dataset's inputs and create its incomplete Step 4 run."""

    _require_execution_acknowledgement(execution_acknowledged)
    source_label = str(execution_source).strip()
    if not source_label:
        raise ValueError("execution_source must not be empty")
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_step4_config(config_file)
    execution = config["execution"]
    overwrite = bool(execution["overwrite"])
    scope = _scope(config)
    dataset_spec = resolve_step4_dataset_spec(
        config,
        project_root=root,
        dataset_id=dataset_id,
    )
    for name, path in (
        ("aligned crop", dataset_spec.aligned_bundle_dir),
        ("landmark region", dataset_spec.landmark_region_bundle_dir),
    ):
        if not (path / "_SUCCESS").is_file():
            raise FileNotFoundError(
                f"{name} prerequisite is incomplete; run the matching "
                f"00_data_preparation notebook first: {path}"
            )
    source = load_step4_source_manifest(dataset_spec)
    validate_aligned_crop_bundle(
        dataset_spec.aligned_bundle_dir,
        dataset_id=dataset_spec.dataset_id,
        expected_source_count=len(source),
        preprocessing_mode=dataset_spec.preprocessing_mode,
        require_full_coverage=dataset_spec.require_full_coverage,
    )
    aligned_index = pd.read_csv(
        dataset_spec.aligned_bundle_dir / "aligned_index.csv"
    )
    failed_index = pd.read_csv(
        dataset_spec.aligned_bundle_dir / "failed_samples.csv"
    )
    selected = _freeze_selected_manifest(
        source,
        aligned_index,
        failed_index,
        dataset_id=dataset_spec.dataset_id,
        scope=scope,
    )

    profile_id = str(execution["model_profile"])
    profile_config = config["models"]["profiles"][profile_id]
    registry_root = root / config["models"]["registry_root"]
    model_spec_path, model_spec = select_model_spec_by_profile(
        registry_root,
        profile_id=profile_id,
        profile_config=profile_config,
        verify_checkpoint=True,
    )
    run_root = resolve_or_create_dataset_run_root(
        root / config["run"]["root"],
        dataset_id=dataset_spec.dataset_id,
        directory_template=config["run"]["dataset_date_dir_template"],
    )
    run_config = _step4_run_config(
        config,
        dataset_id=dataset_spec.dataset_id,
        model_uid=model_spec.model_uid,
        scope=scope,
        overwrite=overwrite,
        execution_source=source_label,
    )
    run = RunStore.create_or_reuse_active(
        experiment_name=f"step4_{dataset_spec.dataset_id}_{model_spec.model_uid}",
        config=run_config,
        root=run_root,
        repo_root=root,
        partition_by_date=bool(config["run"]["partition_by_date"]),
        allow_dirty=bool(execution.get("allow_dirty", False)),
    )
    workflow_root = run.run_dir / config["workflow"]["artifact_subdir"]
    workflow_root.mkdir(parents=True, exist_ok=True)
    with run.phase("00_source_and_model_freeze") as phase:
        for role, path in (
            ("step4_config", config_file),
            ("model_spec", model_spec_path),
            (
                "aligned_bundle_manifest",
                dataset_spec.aligned_bundle_dir / "bundle_manifest.json",
            ),
            (
                "landmark_region_manifest",
                dataset_spec.landmark_region_bundle_dir / "bundle_manifest.json",
            ),
            *[
                (f"dataset_manifest_{index}", path)
                for index, path in enumerate(dataset_spec.manifest_paths)
            ],
        ):
            run.record_input(path, role=role)

        selected_path = (
            workflow_root / config["workflow"]["selected_manifest_path"]
        )
        _write_csv(selected_path, selected, overwrite=overwrite)
        selected_sha256 = sha256_file(selected_path)
        extraction_uid = "population-" + hashlib.sha256(
            "\x1f".join(
                [
                    dataset_spec.dataset_id,
                    model_spec.model_uid,
                    selected_sha256,
                    sha256_file(config_file),
                ]
            ).encode("utf-8")
        ).hexdigest()[:24]
        freeze_manifest = {
            "schema_version": 2,
            "run_id": run.run_id,
            "dataset_id": dataset_spec.dataset_id,
            "model_uid": model_spec.model_uid,
            "checkpoint_sha256": model_spec.checkpoint.sha256,
            "preprocess_hash": model_spec.preprocessing.preprocess_hash,
            "target_layer": model_spec.target_layer,
            "extraction_uid": extraction_uid,
            "scope": scope.as_dict(),
            "selected_sample_count": int(len(selected)),
            "selected_manifest_sha256": selected_sha256,
            "aligned_bundle_manifest_sha256": sha256_file(
                dataset_spec.aligned_bundle_dir / "bundle_manifest.json"
            ),
            "landmark_region_manifest_sha256": sha256_file(
                dataset_spec.landmark_region_bundle_dir
                / "bundle_manifest.json"
            ),
            "fallback_free": True,
        }
        freeze_path = (
            workflow_root / config["workflow"]["freeze_manifest_path"]
        )
        _write_json(freeze_path, freeze_manifest, overwrite=overwrite)
        phase.record_counts(selected_samples=len(selected))
    return {
        "run_id": run.run_id,
        "dataset_id": dataset_spec.dataset_id,
        "model_uid": model_spec.model_uid,
        "selected_samples": int(len(selected)),
        "next_stage": "01_origin_embedding_and_target_templates",
    }


def extract_step4_origin_embeddings(
    config_path: str | Path,
    *,
    project_root: str | Path,
    dataset_id: str,
    execution_acknowledged: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Extract origin embeddings and freeze dataset-specific saliency targets."""

    _require_execution_acknowledgement(execution_acknowledged)
    root = Path(project_root).resolve()
    config = load_step4_config(config_path)
    execution = config["execution"]
    overwrite = bool(execution["overwrite"])
    run, workflow_root, dataset_spec = _open_step4_run(
        config,
        project_root=root,
        dataset_id=dataset_id,
    )
    freeze_path = workflow_root / config["workflow"]["freeze_manifest_path"]
    selected_path = workflow_root / config["workflow"]["selected_manifest_path"]
    if not freeze_path.is_file() or not selected_path.is_file():
        raise FileNotFoundError(
            "source/model freeze stage must complete before embedding extraction"
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    selected = pd.read_csv(selected_path)
    if len(selected) != int(freeze["selected_sample_count"]):
        raise ValueError("selected manifest does not match the frozen sample count")
    profile_id = str(execution["model_profile"])
    profile_config = config["models"]["profiles"][profile_id]
    _, model_spec = select_model_spec_by_profile(
        root / config["models"]["registry_root"],
        profile_id=profile_id,
        profile_config=profile_config,
        verify_checkpoint=True,
    )
    if model_spec.model_uid != str(freeze["model_uid"]):
        raise ValueError("verified model does not match the frozen model UID")
    source_faces = np.load(
        dataset_spec.aligned_bundle_dir / "aligned_faces.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    selected_indices = selected["aligned_face_index"].to_numpy(dtype=np.int64)
    aligned_faces = _aligned_face_selection(source_faces, selected_indices)
    adapter = create_pytorch_adapter_from_spec(
        model_spec,
        device=str(execution["device"]),
    )
    with run.phase("01_origin_embedding_and_target_templates") as phase:
        prepared = prepare_population_saliency_inputs(
            adapter,
            aligned_faces,
            sample_ids=selected["sample_id"].astype(str),
            identity_ids=selected["identity_id"],
            scope_ids=selected["template_scope_id"].astype(str),
            extraction_uid=str(freeze["extraction_uid"]),
            dataset_id=dataset_spec.dataset_id,
            embedding_batch_size=int(
                config["gradcam"]["extraction"]["embedding_batch_size"]
            ),
            require_all_eligible=False,
            progress=progress,
        )
        if dataset_spec.dataset_id == "survface":
            top1_templates = build_origin_top1_gallery_templates(
                prepared.sample_ids,
                prepared.identity_ids,
                prepared.normalized_embeddings,
                selected,
                model_uid=prepared.model_uid,
                scope_ids=prepared.scope_ids,
            )
            prepared = replace(prepared, loo_templates=top1_templates)
        prepared_dir = (
            workflow_root / config["workflow"]["prepared_population_dir"]
        )
        write_prepared_population_artifact(
            prepared,
            prepared_dir,
            shard_size=int(config["gradcam"]["extraction"]["shard_size"]),
            overwrite=overwrite,
        )
        phase.record_counts(
            samples=len(prepared.sample_ids),
            saliency_target_eligible=int(prepared.loo_templates.eligible.sum()),
        )
    return {
        "run_id": run.run_id,
        "dataset_id": dataset_spec.dataset_id,
        "samples": int(len(prepared.sample_ids)),
        "saliency_target_eligible": int(
            prepared.loo_templates.eligible.sum()
        ),
        "target_name": prepared.loo_templates.target_name,
        "next_stage": "02_population_gradcam_extraction",
    }


def extract_step4_population_gradcam(
    config_path: str | Path,
    *,
    project_root: str | Path,
    dataset_id: str,
    execution_acknowledged: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run only the expensive Grad-CAM stage for one frozen dataset run."""

    _require_execution_acknowledgement(execution_acknowledged)
    root = Path(project_root).resolve()
    config = load_step4_config(config_path)
    execution = config["execution"]
    overwrite = bool(execution["overwrite"])
    run, workflow_root, dataset_spec = _open_step4_run(
        config,
        project_root=root,
        dataset_id=dataset_id,
    )
    prepared_dir = (
        workflow_root / config["workflow"]["prepared_population_dir"]
    )
    selected_path = workflow_root / config["workflow"]["selected_manifest_path"]
    if not prepared_dir.is_dir() or not selected_path.is_file():
        raise FileNotFoundError(
            "origin embedding stage must complete before Grad-CAM extraction"
        )
    prepared = read_prepared_population_artifact(prepared_dir)
    selected = pd.read_csv(selected_path)
    if not np.array_equal(
        selected["sample_id"].astype(str).to_numpy(),
        prepared.sample_ids.astype(str),
    ):
        raise ValueError("selected manifest and prepared population differ")
    profile_id = str(execution["model_profile"])
    profile_config = config["models"]["profiles"][profile_id]
    _, model_spec = select_model_spec_by_profile(
        root / config["models"]["registry_root"],
        profile_id=profile_id,
        profile_config=profile_config,
        verify_checkpoint=True,
    )
    if model_spec.model_uid != prepared.model_uid:
        raise ValueError("verified model does not match prepared population")
    source_faces = np.load(
        dataset_spec.aligned_bundle_dir / "aligned_faces.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    selected_indices = selected["aligned_face_index"].to_numpy(dtype=np.int64)
    aligned_faces = _aligned_face_selection(source_faces, selected_indices)
    adapter = create_pytorch_adapter_from_spec(
        model_spec,
        device=str(execution["device"]),
    )
    region_provider = read_landmark_region_bundle(
        dataset_spec.landmark_region_bundle_dir
    ).subset(
        selected_indices,
        expected_sample_ids=prepared.sample_ids,
    )
    sample_cap = config["gradcam"]["population"]["saliency_sample_cap"].get(
        dataset_spec.dataset_id
    )
    saliency_mask = select_step4_saliency_sample_mask(
        selected,
        prepared.loo_templates.eligible,
        dataset_id=dataset_spec.dataset_id,
        maximum_samples=sample_cap,
        seed=int(execution["seed"]),
    )
    faithfulness_enabled = (
        bool(config["gradcam"]["faithfulness"]["enabled"])
        and dataset_spec.dataset_id
        in set(config["gradcam"]["faithfulness"].get("enabled_datasets", []))
    )
    with run.phase("02_population_gradcam_extraction") as phase:
        saliency = extract_population_gradcam(
            adapter,
            aligned_faces,
            prepared,
            gradcam_batch_size=int(
                config["gradcam"]["extraction"]["gradcam_batch_size"]
            ),
            expected_heatmap_size=tuple(
                int(value)
                for value in config["gradcam"]["extraction"][
                    "expected_heatmap_size"
                ]
            ),
            region_masks=region_provider,
            region_mask_uid=region_provider.region_mask_uid,
            capture_intermediates=bool(
                config["gradcam"]["extraction"]["capture_intermediates"]
            ),
            minimum_pass_repeat_cosine=float(
                config["gradcam"]["two_pass_extraction"]["pass_b"][
                    "minimum_pass_repeat_cosine"
                ]
            ),
            faithfulness_fraction=(
                float(
                    config["gradcam"]["faithfulness"][
                        "primary_occlusion_fraction"
                    ]
                )
                if faithfulness_enabled
                else None
            ),
            faithfulness_random_repeats=int(
                config["gradcam"]["faithfulness"]["random_repeats"]
            ),
            faithfulness_seed=int(execution["seed"]),
            saliency_sample_mask=saliency_mask,
            progress=progress,
        )
        saliency_dir = (
            workflow_root / config["workflow"]["saliency_population_dir"]
        )
        write_population_saliency_artifact(
            saliency,
            saliency_dir,
            shard_size=int(config["gradcam"]["extraction"]["shard_size"]),
            heatmap_dtype="float16",
            persistence=config["gradcam"].get("persistence", {}),
            overwrite=overwrite,
        )
        phase.record_counts(
            population_rows=len(prepared.sample_ids),
            gradcam_selected=int(saliency_mask.sum()),
            heatmaps=len(saliency.heatmap_sample_ids),
        )
    selection_sha256 = hashlib.sha256(
        "\n".join(
            selected.loc[saliency_mask, "sample_id"].astype(str)
        ).encode("utf-8")
    ).hexdigest()
    return {
        "run_id": run.run_id,
        "dataset_id": dataset_spec.dataset_id,
        "population_rows": int(len(prepared.sample_ids)),
        "gradcam_selected": int(saliency_mask.sum()),
        "heatmaps": int(len(saliency.heatmap_sample_ids)),
        "saliency_selection_sha256": selection_sha256,
        "region_mask_uid": region_provider.region_mask_uid,
        "next_stage": "03_saliency_feature_validation",
    }


def validate_step4_saliency(
    config_path: str | Path,
    *,
    project_root: str | Path,
    dataset_id: str,
    execution_acknowledged: bool = False,
) -> dict[str, object]:
    """Validate full-population rows and the configured Grad-CAM coverage."""

    _require_execution_acknowledgement(execution_acknowledged)
    root = Path(project_root).resolve()
    config = load_step4_config(config_path)
    execution = config["execution"]
    overwrite = bool(execution["overwrite"])
    run, workflow_root, dataset_spec = _open_step4_run(
        config,
        project_root=root,
        dataset_id=dataset_id,
    )
    prepared_dir = (
        workflow_root / config["workflow"]["prepared_population_dir"]
    )
    saliency_dir = (
        workflow_root / config["workflow"]["saliency_population_dir"]
    )
    selected_path = workflow_root / config["workflow"]["selected_manifest_path"]
    if (
        not prepared_dir.is_dir()
        or not saliency_dir.is_dir()
        or not selected_path.is_file()
    ):
        raise FileNotFoundError(
            "Grad-CAM extraction must complete before saliency validation"
        )
    prepared = read_prepared_population_artifact(prepared_dir)
    features = read_population_saliency_features(saliency_dir)
    selected = pd.read_csv(selected_path)
    if not np.array_equal(
        features["sample_id"].astype(str).to_numpy(),
        prepared.sample_ids.astype(str),
    ):
        raise ValueError("saliency features and prepared population differ")
    maximum_samples = config["gradcam"]["population"][
        "saliency_sample_cap"
    ].get(dataset_spec.dataset_id)
    expected_mask = select_step4_saliency_sample_mask(
        selected,
        prepared.loo_templates.eligible,
        dataset_id=dataset_spec.dataset_id,
        maximum_samples=maximum_samples,
        seed=int(execution["seed"]),
    )
    available = features["heatmap_available"].astype(bool).to_numpy()
    if not np.array_equal(available, expected_mask):
        raise RuntimeError(
            "stored heatmap coverage differs from the deterministic selection"
        )
    eligible = features["saliency_target_eligible"].astype(bool).to_numpy()
    if np.any(available & ~eligible):
        raise RuntimeError("ineligible samples contain a Grad-CAM heatmap")
    valid = (
        available
        & features["gradcam_valid_heatmap"].fillna(False).astype(bool).to_numpy()
    )
    selection_sha256 = hashlib.sha256(
        "\n".join(
            selected.loc[expected_mask, "sample_id"].astype(str)
        ).encode("utf-8")
    ).hexdigest()
    coverage = selected[["sample_id"]].copy()
    coverage["eligible"] = eligible
    coverage["selected_for_gradcam"] = expected_mask
    if "protocol_role" in selected:
        coverage["protocol_role"] = selected["protocol_role"].astype(str)
        role_counts = (
            coverage.groupby("protocol_role", dropna=False)[
                ["eligible", "selected_for_gradcam"]
            ]
            .sum()
            .astype(int)
            .to_dict(orient="index")
        )
    else:
        role_counts = {}
    summary = {
        "run_id": run.run_id,
        "dataset_id": dataset_spec.dataset_id,
        "population_rows": int(len(features)),
        "eligible_rows": int(eligible.sum()),
        "gradcam_selected_rows": int(expected_mask.sum()),
        "valid_heatmap_rows": int(valid.sum()),
        "saliency_sample_cap": maximum_samples,
        "saliency_selection_sha256": selection_sha256,
        "coverage_by_protocol_role": role_counts,
        "semantic_masked_rows": int(
            (
                features["semantic_region_mask_count"].fillna(0) > 0
            ).sum()
        ),
        "faithfulness_rows": int(
            features.get(
                "high_saliency_occlusion_score_drop",
                pd.Series(np.nan, index=features.index),
            ).notna().sum()
        ),
    }
    with run.phase("03_saliency_feature_validation") as phase:
        _write_json(
            workflow_root / config["workflow"]["saliency_validation_path"],
            summary,
            overwrite=overwrite,
        )
        phase.record_counts(
            population_rows=len(features),
            eligible_rows=int(eligible.sum()),
            gradcam_selected_rows=int(expected_mask.sum()),
            valid_heatmap_rows=int(valid.sum()),
        )
    return {
        **summary,
        "next_stage": "04_step2_compression_characterization",
    }


def characterize_step4_compression(
    config_path: str | Path,
    *,
    project_root: str | Path,
    dataset_id: str,
    execution_acknowledged: bool = False,
    progress: ProgressCallback | None = None,
    execution_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run only PCA/PQ characterization and open-set evaluation."""

    _require_execution_acknowledgement(execution_acknowledged)
    root = Path(project_root).resolve()
    config = load_step4_config(config_path)
    execution = config["execution"]
    overwrite = bool(execution["overwrite"])
    run, workflow_root, dataset_spec = _open_step4_run(
        config,
        project_root=root,
        dataset_id=dataset_id,
    )
    prepared_dir = (
        workflow_root / config["workflow"]["prepared_population_dir"]
    )
    selected_path = workflow_root / config["workflow"]["selected_manifest_path"]
    if not prepared_dir.is_dir() or not selected_path.is_file():
        raise FileNotFoundError(
            "origin embedding stage must complete before compression"
        )
    prepared = read_prepared_population_artifact(prepared_dir)
    selected = pd.read_csv(selected_path)
    pca_dimensions = tuple(
        int(value)
        for value in config["compression"]["families"]["pca"]["dimensions"]
    )
    pq_settings = tuple(
        (int(item["m"]), int(item["nbits"]))
        for item in config["compression"]["families"]["pq"]["settings"]
    )
    configured_targets = config["evaluation"].get("reported_target_fpirs")
    if configured_targets is None:
        primary_target_key = (
            "survface_target_fpir"
            if dataset_spec.dataset_id == "survface"
            else "rfw_custom_target_fpir"
            if dataset_spec.dataset_id == "rfw_custom"
            else "target_fpir"
        )
        configured_targets = (config["evaluation"][primary_target_key],)
    reported_target_fpirs = tuple(float(value) for value in configured_targets)
    with run.phase("04_step2_compression_characterization") as phase:
        if execution_context is not None:
            phase.details["execution_context"] = dict(execution_context)
            phase.record(
                "phase_execution_context",
                execution_context=dict(execution_context),
            )
        if dataset_spec.dataset_id == "survface":
            expected_survface_calibration = {
                "survface_calibration_gallery_identities": 3000,
                "survface_calibration_protocol": (
                    "training_3000_half_gallery_v2"
                ),
                "survface_threshold_selection": "non_mated_only",
            }
            mismatches = {
                key: (config["evaluation"].get(key), expected)
                for key, expected in expected_survface_calibration.items()
                if config["evaluation"].get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    "SurvFace matched calibration contract mismatch: "
                    f"{mismatches}"
                )
            compression = characterize_step2_survface_compression(
                prepared,
                selected,
                pca_dimensions=pca_dimensions,
                pq_settings=pq_settings,
                seed=int(execution["seed"]),
                target_fpir=float(
                    config["evaluation"]["survface_target_fpir"]
                ),
                target_fpirs=reported_target_fpirs,
                calibration_gallery_identities=int(
                    config["evaluation"][
                        "survface_calibration_gallery_identities"
                    ]
                ),
                top_k=int(config["evaluation"]["top_k"]),
                progress=progress,
            )
        elif dataset_spec.dataset_id == "rfw_custom":
            compression = characterize_step2_rfw_custom_compression(
                prepared,
                selected,
                pca_dimensions=pca_dimensions,
                pq_settings=pq_settings,
                seed=int(execution["seed"]),
                target_fpir=float(
                    config["evaluation"]["rfw_custom_target_fpir"]
                ),
                target_fpirs=reported_target_fpirs,
                calibration_gallery_identities=int(
                    config["evaluation"][
                        "rfw_custom_calibration_gallery_identities"
                    ]
                ),
                top_k=int(config["evaluation"]["top_k"]),
                progress=progress,
            )
        else:
            def read_ids(key: str) -> tuple[str, ...]:
                return tuple(
                    line.strip()
                    for line in (
                        root / config["datasets"]["lfw"][key]
                    ).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )

            compression = characterize_step2_compression(
                prepared,
                selected,
                gallery_identities=read_ids("gallery_identities_path"),
                unknown_unknown_identities=read_ids(
                    "unknown_unknown_identities_path"
                ),
                pca_dimensions=pca_dimensions,
                pq_settings=pq_settings,
                seed=int(execution["seed"]),
                target_fpir=float(config["evaluation"]["target_fpir"]),
                target_fpirs=reported_target_fpirs,
                enrollment_count=int(
                    config["evaluation"]["lfw_enrollment_count"]
                ),
                calibration_gallery_identities=int(
                    config["evaluation"]["calibration_gallery_identities"]
                ),
                top_k=int(config["evaluation"]["top_k"]),
                progress=progress,
            )
        lineage = {
            "extraction_uid": prepared.extraction_uid,
            "dataset_id": prepared.dataset_id,
            "model_uid": prepared.model_uid,
            "origin_embedding_artifact_uid": (
                prepared.origin_embedding_artifact_uid
            ),
        }
        paired = annotate_compression_lineage(
            compression.paired_metrics,
            **lineage,
        )
        retrieval = annotate_compression_lineage(
            compression.retrieval_metrics,
            **lineage,
        )
        demographic_summary = getattr(
            compression,
            "demographic_summary",
            pd.DataFrame(),
        ).copy()
        if not demographic_summary.empty:
            for column, value in lineage.items():
                if column in demographic_summary.columns:
                    observed = set(
                        demographic_summary[column].dropna().astype(str)
                    )
                    if observed and observed != {str(value)}:
                        raise ValueError(
                            "demographic summary lineage mismatch for "
                            f"{column}: {sorted(observed)}"
                        )
                demographic_summary[column] = value
        origin_score_audit = annotate_compression_lineage(
            compression.origin_score_audit,
            **lineage,
        )
        calibration_diagnostics = dict(compression.calibration_diagnostics)
        calibration_diagnostics["lineage"] = dict(lineage)
        paired_path = (
            workflow_root / config["workflow"]["paired_metrics_path"]
        )
        retrieval_path = (
            workflow_root / config["workflow"]["retrieval_metrics_path"]
        )
        origin_score_audit_path = (
            workflow_root / config["workflow"]["origin_score_audit_path"]
        )
        calibration_diagnostics_path = (
            workflow_root
            / config["workflow"]["calibration_diagnostics_path"]
        )
        demographic_summary_path = workflow_root / config["workflow"].get(
            "rfw_custom_demographic_summary_path",
            "rfw_custom_demographic_summary.csv",
        )
        codec_entries: list[dict[str, object]] = []
        for family, profile, compressor in compression.fitted_codecs:
            attempt_suffix = f"A{phase.attempt:03d}"
            extension = ".joblib" if family == "pca" else ".faiss"
            source_codec_path = phase.attempt_dir / (
                f"{profile}_{attempt_suffix}{extension}"
            )
            compressor.save(source_codec_path)
            published_codec_path = phase.publish_artifact(source_codec_path)
            codec_entries.append(
                {
                    "family": family,
                    "profile_name": profile,
                    "artifact": str(
                        published_codec_path.relative_to(run.run_dir)
                    ).replace("\\", "/"),
                    "artifact_sha256": sha256_file(published_codec_path),
                    "artifact_byte_count": int(
                        published_codec_path.stat().st_size
                    ),
                    "fit_count": int(compressor.fit_count or 0),
                    "fit_seed": int(execution["seed"]),
                    "fit_source_dataset": dataset_spec.dataset_id,
                    "fit_source_run_id": run.run_id,
                    "fit_on_rfw": False,
                }
            )
        frozen_codec_manifest_path = (
            workflow_root
            / config["workflow"].get(
                "frozen_codec_manifest_path",
                "frozen_codec_manifest.json",
            )
        )
        frozen_codec_manifest = {
            "schema_version": 1,
            "status": "completed",
            "artifact_type": "frozen_compression_codec_bundle",
            "fit_source_dataset": dataset_spec.dataset_id,
            "fit_source_run_id": run.run_id,
            "model_uid": prepared.model_uid,
            "extraction_uid": prepared.extraction_uid,
            "origin_embedding_artifact_uid": (
                prepared.origin_embedding_artifact_uid
            ),
            "fit_seed": int(execution["seed"]),
            "fit_on_rfw": False,
            "codecs": codec_entries,
        }
        _write_csv(paired_path, paired, overwrite=overwrite)
        _write_csv(retrieval_path, retrieval, overwrite=overwrite)
        if not demographic_summary.empty:
            _write_csv(
                demographic_summary_path,
                demographic_summary,
                overwrite=overwrite,
            )
        _write_csv(
            origin_score_audit_path,
            origin_score_audit,
            overwrite=overwrite,
        )
        _write_json(
            calibration_diagnostics_path,
            calibration_diagnostics,
            overwrite=overwrite,
        )
        _write_json(
            frozen_codec_manifest_path,
            frozen_codec_manifest,
            overwrite=overwrite,
        )
        phase.record_counts(
            paired_rows=len(paired),
            retrieval_rows=len(retrieval),
            demographic_summary_rows=len(demographic_summary),
            origin_score_audit_rows=len(origin_score_audit),
            frozen_codec_count=len(codec_entries),
        )
    return {
        "run_id": run.run_id,
        "dataset_id": dataset_spec.dataset_id,
        "paired_rows": int(len(paired)),
        "retrieval_rows": int(len(retrieval)),
        "demographic_summary_rows": int(len(demographic_summary)),
        "origin_score_audit_rows": int(len(origin_score_audit)),
        "frozen_codec_count": int(len(codec_entries)),
        "frozen_codec_manifest_path": str(frozen_codec_manifest_path),
        "frozen_codec_manifest_sha256": sha256_file(
            frozen_codec_manifest_path
        ),
        "calibration_origin_fpir": float(
            calibration_diagnostics["splits"]["calibration"]["origin_fpir"]
        ),
        "test_origin_fpir": float(
            calibration_diagnostics["splits"]["test"]["origin_fpir"]
        ),
        "origin_calibration_transfer_status": str(
            calibration_diagnostics["calibration_transfer_assessment"]["status"]
        ),
        "next_stage": "05_saliency_compression_join",
    }


def analyze_step4_saliency_compression(
    config_path: str | Path,
    *,
    project_root: str | Path,
    dataset_id: str,
    execution_acknowledged: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Stream strict joins and estimate scalable identity-cluster associations."""

    _require_execution_acknowledgement(execution_acknowledged)
    root = Path(project_root).resolve()
    config = load_step4_config(config_path)
    execution = config["execution"]
    overwrite = bool(execution["overwrite"])
    run, workflow_root, dataset_spec = _open_step4_run(
        config,
        project_root=root,
        dataset_id=dataset_id,
    )
    saliency_dir = (
        workflow_root / config["workflow"]["saliency_population_dir"]
    )
    paired_path = workflow_root / config["workflow"]["paired_metrics_path"]
    retrieval_path = (
        workflow_root / config["workflow"]["retrieval_metrics_path"]
    )
    for path in (paired_path, retrieval_path):
        if not path.is_file():
            raise FileNotFoundError(
                "compression characterization must complete before join: "
                f"{path}"
            )
    association_config = config["joint_analysis"]["association"]
    bootstrap_method = str(association_config["bootstrap_method"])
    bootstrap_unit = str(association_config["bootstrap_unit"])
    if bootstrap_method != "identity_cluster":
        raise ValueError(
            "Step 4 saliency association requires identity_cluster bootstrap"
        )
    if bootstrap_unit != "identity_id":
        raise ValueError(
            "Step 4 saliency association requires bootstrap_unit=identity_id"
        )
    raw_bootstrap = association_config["bootstrap_repeats"]
    if (
        isinstance(raw_bootstrap, bool)
        or not isinstance(raw_bootstrap, int)
        or raw_bootstrap < 0
    ):
        raise ValueError("bootstrap_repeats must be a non-negative integer")
    bootstrap = raw_bootstrap
    output_paths = {
        key: workflow_root / config["workflow"][key]
        for key in (
            "geometry_joined_metrics_path",
            "retrieval_joined_metrics_path",
            "geometry_association_path",
            "retrieval_association_path",
        )
    }
    candidate_path = workflow_root / config["workflow"].get(
        "representative_case_candidates_path",
        "representative_case_candidates.csv",
    )
    persist_large_joins = bool(
        config["workflow"].get("persist_large_join_artifacts", True)
    )
    checked_output_paths = [
        output_paths["geometry_association_path"],
        output_paths["retrieval_association_path"],
        candidate_path,
    ]
    if persist_large_joins:
        checked_output_paths.extend(
            (
                output_paths["geometry_joined_metrics_path"],
                output_paths["retrieval_joined_metrics_path"],
            )
        )
    if not overwrite:
        existing = [path for path in checked_output_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Step 4 join outputs already exist and overwrite=false: "
                + ", ".join(str(path) for path in existing)
            )
    phase04_counts = _latest_completed_phase_counts(
        run,
        "04_step2_compression_characterization",
    )
    expected_geometry_rows = int(phase04_counts["paired_rows"])
    expected_retrieval_rows = int(phase04_counts["retrieval_rows"])

    with run.phase("05_saliency_compression_join") as phase:
        implementation_sources = {
            "association": (
                root / "research/evaluation/saliency_compression.py"
            ),
            "streaming_join": (
                root / "research/evaluation/saliency_streaming.py"
            ),
            "workflow": root / "research/experiments/step4_workflow.py",
        }
        phase.details["implementation"] = {
            "association_algorithm_version": (
                WEIGHTED_RERANK_ALGORITHM_VERSION
            ),
            "bootstrap_method": bootstrap_method,
            "bootstrap_unit": bootstrap_unit,
            "bootstrap_rank_strategy": WEIGHTED_RERANK_STRATEGY,
            "bootstrap_repeats": bootstrap,
            "bootstrap_batch_size": STEP4_BOOTSTRAP_BATCH_SIZE,
            "join_chunk_rows": STEP4_JOIN_CHUNK_ROWS,
            "source_git_commit": _current_git_commit(root),
            "source_sha256": {
                name: sha256_file(path)
                for name, path in implementation_sources.items()
            },
        }
        phase.record(
            "saliency_compression_analysis_started",
            **phase.details["implementation"],
        )
        features = read_population_saliency_features(saliency_dir)
        with tempfile.TemporaryDirectory(
            prefix=".05_saliency_compression_join.",
            dir=workflow_root,
        ) as staging_name:
            staging = Path(staging_name)
            staged_geometry_join = staging / "saliency_geometry_join.csv"
            staged_geometry_projection = (
                staging / "saliency_geometry_projection.parquet"
            )
            staged_geometry_association = (
                staging / "saliency_geometry_associations.csv"
            )
            staged_retrieval_join = staging / "saliency_retrieval_join.csv"
            staged_retrieval_projection = (
                staging / "saliency_retrieval_projection.parquet"
            )
            staged_retrieval_association = (
                staging / "saliency_retrieval_associations.csv"
            )
            staged_case_candidates = (
                staging / "representative_case_candidates.csv"
            )

            geometry_stream = (
                stream_join_population_saliency_with_compression(
                    features,
                    paired_path,
                    joined_output_path=(
                        staged_geometry_join if persist_large_joins else None
                    ),
                    association_projection_path=staged_geometry_projection,
                    chunksize=STEP4_JOIN_CHUNK_ROWS,
                    expected_rows=expected_geometry_rows,
                    progress=_scoped_progress(progress, "geometry"),
                )
            )
            geometry_projection = pd.read_parquet(
                geometry_stream.association_projection_path
            )
            geometry_associations = saliency_geometry_associations(
                geometry_projection,
                bootstrap_repeats=bootstrap,
                seed=int(execution["seed"]),
                bootstrap_rank_strategy=WEIGHTED_RERANK_STRATEGY,
                bootstrap_batch_size=STEP4_BOOTSTRAP_BATCH_SIZE,
                progress=_scoped_progress(
                    progress,
                    "geometry association",
                ),
            )
            _write_csv(
                staged_geometry_association,
                geometry_associations,
                overwrite=False,
            )
            geometry_association_rows = int(len(geometry_associations))
            del geometry_projection, geometry_associations
            gc.collect()

            retrieval_stream = stream_join_population_saliency_with_retrieval(
                features,
                retrieval_path,
                joined_output_path=(
                    staged_retrieval_join if persist_large_joins else None
                ),
                association_projection_path=staged_retrieval_projection,
                chunksize=STEP4_JOIN_CHUNK_ROWS,
                expected_rows=expected_retrieval_rows,
                progress=_scoped_progress(progress, "retrieval"),
            )
            retrieval_projection = pd.read_parquet(
                retrieval_stream.association_projection_path
            )
            retrieval_associations = saliency_retrieval_associations(
                retrieval_projection,
                bootstrap_repeats=bootstrap,
                seed=int(execution["seed"]),
                bootstrap_rank_strategy=WEIGHTED_RERANK_STRATEGY,
                bootstrap_batch_size=STEP4_BOOTSTRAP_BATCH_SIZE,
                progress=_scoped_progress(
                    progress,
                    "retrieval association",
                ),
            )
            _write_csv(
                staged_retrieval_association,
                retrieval_associations,
                overwrite=False,
            )
            retrieval_association_rows = int(len(retrieval_associations))
            del retrieval_projection, retrieval_associations, features
            gc.collect()
            case_config = config.get("gradcam", {}).get(
                "representative_case_visualization",
                {
                    "threshold_policy": "frozen_origin",
                    "samples_per_stratum": 8,
                },
            )
            case_candidates = stream_select_population_representative_cases(
                retrieval_stream.association_projection_path,
                geometry_stream.association_projection_path,
                threshold_policy=str(case_config["threshold_policy"]),
                cases_per_group=int(case_config["samples_per_stratum"]),
                seed=int(execution["seed"]),
                chunksize=STEP4_JOIN_CHUNK_ROWS,
            )
            _write_csv(
                staged_case_candidates,
                case_candidates,
                overwrite=False,
            )
            representative_candidate_rows = int(len(case_candidates))

            staged_outputs = {
                "geometry_association_path": staged_geometry_association,
                "retrieval_association_path": staged_retrieval_association,
                "representative_case_candidates_path": staged_case_candidates,
            }
            if persist_large_joins:
                staged_outputs.update(
                    {
                        "geometry_joined_metrics_path": staged_geometry_join,
                        "retrieval_joined_metrics_path": staged_retrieval_join,
                    }
                )
            if not overwrite:
                appeared = [
                    (
                        candidate_path
                        if key == "representative_case_candidates_path"
                        else output_paths[key]
                    )
                    for key in staged_outputs
                    if (
                        candidate_path
                        if key == "representative_case_candidates_path"
                        else output_paths[key]
                    ).exists()
                ]
                if appeared:
                    raise FileExistsError(
                        "Step 4 join outputs appeared during computation: "
                        + ", ".join(str(path) for path in appeared)
                    )
            for key, staged_path in staged_outputs.items():
                destination = (
                    candidate_path
                    if key == "representative_case_candidates_path"
                    else output_paths[key]
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_path, destination)

        phase.record_counts(
            geometry_join_rows=geometry_stream.row_count,
            geometry_projection_rows=geometry_stream.projected_row_count,
            retrieval_join_rows=retrieval_stream.row_count,
            retrieval_projection_rows=retrieval_stream.projected_row_count,
            geometry_association_rows=geometry_association_rows,
            retrieval_association_rows=retrieval_association_rows,
            representative_candidate_rows=representative_candidate_rows,
            geometry_join_chunks=geometry_stream.chunk_count,
            retrieval_join_chunks=retrieval_stream.chunk_count,
        )
        phase.record(
            "saliency_compression_analysis_completed",
            association_algorithm_version=(
                WEIGHTED_RERANK_ALGORITHM_VERSION
            ),
        )
    return {
        "run_id": run.run_id,
        "dataset_id": dataset_spec.dataset_id,
        "geometry_join_rows": int(geometry_stream.row_count),
        "retrieval_join_rows": int(retrieval_stream.row_count),
        "geometry_association_rows": geometry_association_rows,
        "retrieval_association_rows": retrieval_association_rows,
        "representative_candidate_rows": representative_candidate_rows,
        "persisted_large_join_artifacts": persist_large_joins,
        "association_algorithm_version": WEIGHTED_RERANK_ALGORITHM_VERSION,
        "bootstrap_rank_strategy": WEIGHTED_RERANK_STRATEGY,
        "next_stage": "06_representative_case_visualization",
    }


def finalize_step4_representative_cases(
    config_path: str | Path,
    *,
    project_root: str | Path,
    dataset_id: str,
    execution_acknowledged: bool = False,
) -> dict[str, object]:
    """Create read-only-from-Grad-CAM figures and complete the dataset run."""

    _require_execution_acknowledgement(execution_acknowledged)
    root = Path(project_root).resolve()
    config = load_step4_config(config_path)
    execution = config["execution"]
    overwrite = bool(execution["overwrite"])
    run, workflow_root, dataset_spec = _open_step4_run(
        config,
        project_root=root,
        dataset_id=dataset_id,
    )
    joined_path = (
        workflow_root / config["workflow"]["retrieval_joined_metrics_path"]
    )
    geometry_path = (
        workflow_root / config["workflow"]["geometry_joined_metrics_path"]
    )
    saliency_dir = (
        workflow_root / config["workflow"]["saliency_population_dir"]
    )
    selected_path = workflow_root / config["workflow"]["selected_manifest_path"]
    validation_path = (
        workflow_root / config["workflow"]["saliency_validation_path"]
    )
    candidate_path = workflow_root / config["workflow"].get(
        "representative_case_candidates_path",
        "representative_case_candidates.csv",
    )
    for path in (selected_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(
                "all preceding Step 4 stages must complete before finalization: "
                f"{path}"
            )
    threshold_policy = str(
        config["gradcam"]["representative_case_visualization"][
            "threshold_policy"
        ]
    )
    cases_per_group = int(
        config["gradcam"]["representative_case_visualization"][
            "samples_per_stratum"
        ]
    )
    if candidate_path.is_file():
        cases = pd.read_csv(candidate_path, low_memory=False)
    else:
        for path in (joined_path, geometry_path):
            if not path.is_file():
                raise FileNotFoundError(
                    "phase 05 must provide compact case candidates or legacy "
                    f"joined metrics: {path}"
                )
        cases = stream_select_population_representative_cases(
            joined_path,
            geometry_path,
            threshold_policy=threshold_policy,
            cases_per_group=cases_per_group,
            seed=int(execution["seed"]),
            chunksize=STEP4_JOIN_CHUNK_ROWS,
        )
        _write_csv(candidate_path, cases, overwrite=False)
    heatmap_ids, heatmaps = read_population_heatmaps(saliency_dir)
    heatmap_index = {
        str(sample_id): index
        for index, sample_id in enumerate(heatmap_ids.astype(str))
    }
    selected = pd.read_csv(selected_path)
    selected["sample_id"] = selected["sample_id"].astype(str)
    sample_to_face = (
        selected.set_index("sample_id")["aligned_face_index"]
        .astype(int)
        .to_dict()
    )
    aligned = np.load(
        dataset_spec.aligned_bundle_dir / "aligned_faces.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    case_path = (
        workflow_root / config["workflow"]["representative_cases_path"]
    )
    figure_dir = workflow_root / config["workflow"]["figure_dir"]
    if (case_path.exists() or figure_dir.exists()) and not overwrite:
        raise FileExistsError(
            "representative-case artifacts already exist; completed runs "
            "must not be overwritten"
        )
    with run.phase("06_representative_case_visualization") as phase:
        import matplotlib.pyplot as plt

        case_path.parent.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        cases.to_csv(case_path, index=False, encoding="utf-8")
        for row in cases.itertuples(index=False):
            sample_id = str(row.sample_id)
            image = np.asarray(aligned[sample_to_face[sample_id]])
            heatmap = heatmaps[heatmap_index[sample_id]]
            figure, axis = plt.subplots(figsize=(3, 3))
            axis.imshow(image)
            axis.imshow(
                heatmap,
                cmap="jet",
                alpha=0.45,
                extent=(0, image.shape[1], image.shape[0], 0),
            )
            axis.set_title(f"{row.case_group}: {sample_id}")
            axis.axis("off")
            figure.savefig(
                figure_dir / f"{row.case_id}.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(figure)
        phase.record_counts(cases=len(cases), figures=len(cases))

    prepared = read_prepared_population_artifact(
        workflow_root / config["workflow"]["prepared_population_dir"]
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    phase04_counts = _latest_completed_phase_counts(
        run,
        "04_step2_compression_characterization",
    )
    calibration_diagnostics = json.loads(
        (
            workflow_root
            / config["workflow"]["calibration_diagnostics_path"]
        ).read_text(encoding="utf-8")
    )
    frozen_codec_manifest_path = (
        workflow_root
        / config["workflow"].get(
            "frozen_codec_manifest_path",
            "frozen_codec_manifest.json",
        )
    )
    if not frozen_codec_manifest_path.is_file():
        raise FileNotFoundError(
            "compression phase did not publish its frozen codec manifest: "
            f"{frozen_codec_manifest_path}"
        )
    geometry_associations = pd.read_csv(
        workflow_root / config["workflow"]["geometry_association_path"]
    )
    retrieval_associations = pd.read_csv(
        workflow_root / config["workflow"]["retrieval_association_path"]
    )
    summary = {
        "run_id": run.run_id,
        "dataset_id": dataset_spec.dataset_id,
        "model_uid": prepared.model_uid,
        "selected_samples": int(len(prepared.sample_ids)),
        "saliency_selected_samples": int(
            validation["gradcam_selected_rows"]
        ),
        "saliency_selection_sha256": validation[
            "saliency_selection_sha256"
        ],
        "paired_rows": int(phase04_counts["paired_rows"]),
        "retrieval_rows": int(phase04_counts["retrieval_rows"]),
        "origin_score_audit_rows": int(
            phase04_counts["origin_score_audit_rows"]
        ),
        "frozen_codec_count": int(phase04_counts["frozen_codec_count"]),
        "frozen_codec_manifest_sha256": sha256_file(
            frozen_codec_manifest_path
        ),
        "calibration_origin_fpir": float(
            calibration_diagnostics["splits"]["calibration"]["origin_fpir"]
        ),
        "test_origin_fpir": float(
            calibration_diagnostics["splits"]["test"]["origin_fpir"]
        ),
        "origin_calibration_transfer_status": str(
            calibration_diagnostics["calibration_transfer_assessment"]["status"]
        ),
        "geometry_association_rows": int(len(geometry_associations)),
        "retrieval_association_rows": int(len(retrieval_associations)),
        "representative_cases": int(len(cases)),
        "threshold_policy": threshold_policy,
        "regenerated_gradcam": False,
    }
    summary_path = workflow_root / "step4_summary.json"
    _write_json(summary_path, summary, overwrite=overwrite)
    run.complete()
    return summary
