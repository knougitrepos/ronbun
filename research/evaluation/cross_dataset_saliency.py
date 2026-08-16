from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from research.evaluation.saliency_compression import (
    SALIENCY_THRESHOLD_METRICS_VERSION,
)
from research.runtime.hashing import sha256_file


_WORKFLOW_SUBDIR = Path("artifacts/step2_workflow")
_GEOMETRY_FILE = "saliency_geometry_associations.csv"
_RETRIEVAL_FILE = "saliency_retrieval_associations.csv"
_THRESHOLD_INSTABILITY_FILE = "saliency_threshold_instability_associations.csv"
_THRESHOLD_POLICY_FILE = "saliency_threshold_policy_comparisons.csv"
_THRESHOLD_POLICY_RHO_FILE = "saliency_threshold_policy_rho_comparisons.csv"
_CASES_FILE = "representative_cases.csv"
_RFW_CALIBRATION_CONTRACT = "rfw_custom_gallery_group_matched_calibration_v2"
_RFW_GALLERY_POLICY = "evaluation_group_matched"


@dataclass(frozen=True)
class CrossDatasetSaliencyAssociations:
    geometry: pd.DataFrame
    retrieval: pd.DataFrame
    threshold_instability: pd.DataFrame
    threshold_policy_comparison: pd.DataFrame
    threshold_policy_saliency_rho: pd.DataFrame
    representative_cases: pd.DataFrame
    source_files: dict[str, dict[str, dict[str, object]]]


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _latest_completed_attempt(run_dir: Path, phase_name: str) -> Path:
    attempts = run_dir / "phases" / phase_name / "attempts"
    candidates: list[tuple[int, Path, dict[str, object]]] = []
    for path in attempts.glob("A*/phase_manifest.json"):
        payload = _read_json(path)
        candidates.append((int(payload["attempt"]), path, payload))
    if not candidates:
        raise RuntimeError(f"phase has no attempts: {phase_name}")
    _attempt, path, payload = max(candidates, key=lambda item: item[0])
    if payload.get("status") != "completed":
        raise RuntimeError(f"latest phase attempt is not completed: {phase_name}")
    return path


def _file_entry(path: Path, *, run_dir: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_identity(
    payload: Mapping[str, object],
    *,
    dataset: str,
    model_uid: str,
    run_id: str,
    artifact_name: str,
) -> None:
    expected = {
        "dataset_id": dataset,
        "model_uid": model_uid,
        "run_id": run_id,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{artifact_name} identity mismatch: {mismatches}")


def _validate_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    model_uid: str,
    analysis_scope: str | None,
    required_columns: set[str],
    artifact_name: str,
) -> None:
    if frame.empty:
        raise ValueError(f"{artifact_name} must not be empty")
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{artifact_name} columns are missing: {sorted(missing)}")
    if set(frame["dataset_id"].astype(str)) != {dataset}:
        raise ValueError(f"{artifact_name} dataset_id mismatch")
    if set(frame["model_uid"].astype(str)) != {model_uid}:
        raise ValueError(f"{artifact_name} model_uid mismatch")
    if analysis_scope is not None and set(frame["analysis_scope"].astype(str)) != {
        analysis_scope
    }:
        raise ValueError(f"{artifact_name} analysis_scope mismatch")


def _with_lineage(frame: pd.DataFrame, *, run_id: str, run_dir: Path) -> pd.DataFrame:
    enriched = frame.copy()
    if "run_id" in enriched and set(enriched["run_id"].astype(str)) != {run_id}:
        raise ValueError("association frame contains a conflicting run_id")
    enriched["run_id"] = run_id
    enriched["source_run_dir"] = str(run_dir)
    return enriched


def _validate_threshold_metric_version(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
) -> None:
    versions = set(frame["threshold_metric_derivation_version"].astype(str))
    if versions != {SALIENCY_THRESHOLD_METRICS_VERSION}:
        raise ValueError(
            f"{artifact_name} threshold metric version mismatch: "
            f"{sorted(versions)}"
        )


def validate_rfw_custom_calibration_contract(
    run_manifest: Mapping[str, object],
    diagnostics: Mapping[str, object],
) -> None:
    """Fail closed unless an RFW-Custom run uses the current gallery contract."""
    config = run_manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("RFW-Custom run manifest is missing its frozen config")
    step4 = config.get("step4")
    evaluation = step4.get("evaluation") if isinstance(step4, Mapping) else None
    if not isinstance(evaluation, Mapping) or evaluation.get(
        "rfw_custom_calibration_gallery_policy"
    ) != _RFW_GALLERY_POLICY:
        raise ValueError(
            "RFW-Custom run predates gallery-size-matched calibration"
        )
    contract = diagnostics.get("calibration_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("RFW-Custom calibration diagnostics lack a contract")
    expected = {
        "name": _RFW_CALIBRATION_CONTRACT,
        "score_statistic": "maximum_gallery_score",
        "gallery_matching_policy": _RFW_GALLERY_POLICY,
        "gallery_size_match_verified": True,
        "gallery_group_match_verified": True,
    }
    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"RFW-Custom calibration contract mismatch: {mismatches}"
        )
    splits = diagnostics.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("RFW-Custom calibration diagnostics lack split summaries")
    calibration = splits.get("calibration")
    test = splits.get("test")
    if not isinstance(calibration, Mapping) or not isinstance(test, Mapping):
        raise ValueError("RFW-Custom calibration/test split summaries are required")
    calibration_templates = int(calibration.get("template_count", -1))
    test_templates = int(test.get("template_count", -1))
    if calibration_templates <= 0 or calibration_templates != test_templates:
        raise ValueError(
            "RFW-Custom calibration/test gallery template counts do not match"
        )


def load_cross_dataset_saliency_associations(
    selected_runs: Mapping[str, str | Path],
    *,
    expected_model_uids: Mapping[str, str],
    expected_run_ids: Mapping[str, str],
    expected_target_fpirs: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.30),
) -> CrossDatasetSaliencyAssociations:
    """Load full association tables from explicit completed Step 4 runs."""

    dataset_keys = set(selected_runs)
    if not dataset_keys:
        raise ValueError("at least one selected run is required")
    for name, mapping in (
        ("expected_model_uids", expected_model_uids),
        ("expected_run_ids", expected_run_ids),
    ):
        if set(mapping) != dataset_keys:
            raise ValueError(f"{name} keys must exactly match selected_runs")

    geometry_frames: list[pd.DataFrame] = []
    retrieval_frames: list[pd.DataFrame] = []
    threshold_instability_frames: list[pd.DataFrame] = []
    threshold_policy_frames: list[pd.DataFrame] = []
    threshold_policy_rho_frames: list[pd.DataFrame] = []
    case_frames: list[pd.DataFrame] = []
    sources: dict[str, dict[str, dict[str, object]]] = {}
    expected_fpirs = {float(value) for value in expected_target_fpirs}

    for dataset, raw_run_dir in selected_runs.items():
        run_dir = Path(raw_run_dir).expanduser().resolve()
        model_uid = str(expected_model_uids[dataset])
        run_id = str(expected_run_ids[dataset])
        completed = run_dir / "COMPLETED"
        if not completed.is_file():
            raise FileNotFoundError(f"completed marker is missing: {completed}")

        run_manifest_path = run_dir / "run_manifest.json"
        freeze_path = run_dir / _WORKFLOW_SUBDIR / "freeze_manifest.json"
        summary_path = run_dir / _WORKFLOW_SUBDIR / "step4_summary.json"
        diagnostics_path = (
            run_dir / _WORKFLOW_SUBDIR / "origin_calibration_diagnostics.json"
        )
        run_manifest = _read_json(run_manifest_path)
        freeze = _read_json(freeze_path)
        summary = _read_json(summary_path)
        if run_manifest.get("status") != "completed":
            raise ValueError(f"run manifest is not completed: {run_dir}")
        if run_manifest.get("run_id") != run_id:
            raise ValueError(f"run manifest run_id mismatch: {run_dir}")
        _validate_identity(
            freeze,
            dataset=dataset,
            model_uid=model_uid,
            run_id=run_id,
            artifact_name="freeze_manifest.json",
        )
        _validate_identity(
            summary,
            dataset=dataset,
            model_uid=model_uid,
            run_id=run_id,
            artifact_name="step4_summary.json",
        )
        if freeze.get("fallback_free") is not True:
            raise ValueError(f"selected run is not fallback-free: {run_dir}")
        diagnostics: dict[str, object] | None = None
        if dataset == "rfw_custom":
            diagnostics = _read_json(diagnostics_path)
            validate_rfw_custom_calibration_contract(run_manifest, diagnostics)

        phase05_path = _latest_completed_attempt(
            run_dir,
            "05_saliency_compression_join",
        )
        phase05 = _read_json(phase05_path)
        phase05_details = phase05.get("details")
        implementation = (
            phase05_details.get("implementation")
            if isinstance(phase05_details, Mapping)
            else None
        )
        if not isinstance(implementation, Mapping) or implementation.get(
            "threshold_metric_derivation_version"
        ) != SALIENCY_THRESHOLD_METRICS_VERSION:
            raise ValueError(
                f"{dataset}: Phase05 does not use "
                f"{SALIENCY_THRESHOLD_METRICS_VERSION}"
            )
        phase06_path = _latest_completed_attempt(
            run_dir,
            "06_representative_case_visualization",
        )
        workflow = run_dir / _WORKFLOW_SUBDIR
        geometry_path = workflow / _GEOMETRY_FILE
        retrieval_path = workflow / _RETRIEVAL_FILE
        threshold_instability_path = workflow / _THRESHOLD_INSTABILITY_FILE
        threshold_policy_path = workflow / _THRESHOLD_POLICY_FILE
        threshold_policy_rho_path = workflow / _THRESHOLD_POLICY_RHO_FILE
        cases_path = workflow / _CASES_FILE
        for path in (
            geometry_path,
            retrieval_path,
            threshold_instability_path,
            threshold_policy_path,
            threshold_policy_rho_path,
            cases_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        geometry = pd.read_csv(geometry_path, low_memory=False)
        retrieval = pd.read_csv(retrieval_path, low_memory=False)
        threshold_instability = pd.read_csv(
            threshold_instability_path,
            low_memory=False,
        )
        threshold_policy = pd.read_csv(threshold_policy_path, low_memory=False)
        threshold_policy_rho = pd.read_csv(
            threshold_policy_rho_path,
            low_memory=False,
        )
        cases = pd.read_csv(cases_path, low_memory=False)
        _validate_frame(
            geometry,
            dataset=dataset,
            model_uid=model_uid,
            analysis_scope="geometry",
            required_columns={
                "analysis_scope",
                "dataset_id",
                "model_uid",
                "compression_family",
                "compression_profile",
                "saliency_feature",
                "sensitivity_metric",
                "sample_count",
                "identity_count",
                "spearman_rho",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
            },
            artifact_name=_GEOMETRY_FILE,
        )
        _validate_frame(
            retrieval,
            dataset=dataset,
            model_uid=model_uid,
            analysis_scope="retrieval",
            required_columns={
                "analysis_scope",
                "threshold_metric_derivation_version",
                "dataset_id",
                "model_uid",
                "compression_family",
                "compression_profile",
                "search_mode",
                "target_fpir",
                "threshold_policy",
                "is_mated",
                "saliency_feature",
                "sensitivity_metric",
                "sample_count",
                "identity_count",
                "spearman_rho",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
            },
            artifact_name=_RETRIEVAL_FILE,
        )
        _validate_threshold_metric_version(
            retrieval,
            artifact_name=_RETRIEVAL_FILE,
        )
        _validate_frame(
            threshold_instability,
            dataset=dataset,
            model_uid=model_uid,
            analysis_scope="threshold_instability",
            required_columns={
                "analysis_scope",
                "threshold_metric_derivation_version",
                "dataset_id",
                "model_uid",
                "compression_family",
                "compression_profile",
                "search_mode",
                "target_fpir",
                "threshold_policy",
                "is_mated",
                "instability_predictor",
                "sensitivity_metric",
                "sample_count",
                "identity_count",
                "event_count",
                "non_event_count",
                "event_rate",
                "spearman_rho",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
            },
            artifact_name=_THRESHOLD_INSTABILITY_FILE,
        )
        _validate_threshold_metric_version(
            threshold_instability,
            artifact_name=_THRESHOLD_INSTABILITY_FILE,
        )
        _validate_frame(
            threshold_policy,
            dataset=dataset,
            model_uid=model_uid,
            analysis_scope="threshold_policy_comparison",
            required_columns={
                "analysis_scope",
                "threshold_metric_derivation_version",
                "dataset_id",
                "model_uid",
                "compression_family",
                "compression_profile",
                "search_mode",
                "target_fpir",
                "is_mated",
                "event_metric",
                "paired_query_count",
                "identity_count",
                "frozen_event_count",
                "frozen_event_rate",
                "recalibrated_event_count",
                "recalibrated_event_rate",
                "recalibrated_minus_frozen_rate",
                "resolved_event_count",
                "introduced_event_count",
                "paired_bootstrap_ci_low",
                "paired_bootstrap_ci_high",
            },
            artifact_name=_THRESHOLD_POLICY_FILE,
        )
        _validate_threshold_metric_version(
            threshold_policy,
            artifact_name=_THRESHOLD_POLICY_FILE,
        )
        _validate_frame(
            threshold_policy_rho,
            dataset=dataset,
            model_uid=model_uid,
            analysis_scope="threshold_policy_saliency_rho",
            required_columns={
                "analysis_scope",
                "threshold_metric_derivation_version",
                "dataset_id",
                "model_uid",
                "compression_family",
                "compression_profile",
                "search_mode",
                "target_fpir",
                "is_mated",
                "saliency_feature",
                "event_metric",
                "paired_query_count",
                "identity_count",
                "frozen_event_count",
                "recalibrated_event_count",
                "frozen_spearman_rho",
                "recalibrated_spearman_rho",
                "recalibrated_minus_frozen_rho",
                "paired_bootstrap_ci_low",
                "paired_bootstrap_ci_high",
            },
            artifact_name=_THRESHOLD_POLICY_RHO_FILE,
        )
        _validate_threshold_metric_version(
            threshold_policy_rho,
            artifact_name=_THRESHOLD_POLICY_RHO_FILE,
        )
        _validate_frame(
            cases,
            dataset=dataset,
            model_uid=model_uid,
            analysis_scope=None,
            required_columns={
                "case_id",
                "dataset_id",
                "model_uid",
                "compression_family",
                "compression_profile",
                "search_mode",
                "target_fpir",
                "threshold_policy",
                "case_group",
            },
            artifact_name=_CASES_FILE,
        )
        observed_fpirs = {float(value) for value in retrieval["target_fpir"].unique()}
        if observed_fpirs != expected_fpirs:
            raise ValueError(
                f"{dataset}: saliency retrieval FPIR coverage mismatch: "
                f"{sorted(observed_fpirs)}"
            )
        for artifact_name, frame in (
            (_THRESHOLD_INSTABILITY_FILE, threshold_instability),
            (_THRESHOLD_POLICY_FILE, threshold_policy),
            (_THRESHOLD_POLICY_RHO_FILE, threshold_policy_rho),
        ):
            observed = {float(value) for value in frame["target_fpir"].unique()}
            if observed != expected_fpirs:
                raise ValueError(
                    f"{dataset}: {artifact_name} FPIR coverage mismatch: "
                    f"{sorted(observed)}"
                )
        if int(summary.get("geometry_association_rows", -1)) != len(geometry):
            raise ValueError(f"{dataset}: geometry association row count mismatch")
        if int(summary.get("retrieval_association_rows", -1)) != len(retrieval):
            raise ValueError(f"{dataset}: retrieval association row count mismatch")
        if int(summary.get("threshold_instability_association_rows", -1)) != len(
            threshold_instability
        ):
            raise ValueError(
                f"{dataset}: threshold instability row count mismatch"
            )
        if int(summary.get("threshold_policy_comparison_rows", -1)) != len(
            threshold_policy
        ):
            raise ValueError(f"{dataset}: threshold policy row count mismatch")
        if int(summary.get("threshold_policy_saliency_rho_rows", -1)) != len(
            threshold_policy_rho
        ):
            raise ValueError(
                f"{dataset}: threshold policy saliency rho row count mismatch"
            )
        if int(summary.get("representative_cases", -1)) != len(cases):
            raise ValueError(f"{dataset}: representative case row count mismatch")

        geometry_frames.append(_with_lineage(geometry, run_id=run_id, run_dir=run_dir))
        retrieval_frames.append(_with_lineage(retrieval, run_id=run_id, run_dir=run_dir))
        threshold_instability_frames.append(
            _with_lineage(
                threshold_instability,
                run_id=run_id,
                run_dir=run_dir,
            )
        )
        threshold_policy_frames.append(
            _with_lineage(threshold_policy, run_id=run_id, run_dir=run_dir)
        )
        threshold_policy_rho_frames.append(
            _with_lineage(threshold_policy_rho, run_id=run_id, run_dir=run_dir)
        )
        case_frames.append(_with_lineage(cases, run_id=run_id, run_dir=run_dir))
        sources[dataset] = {
            "run_manifest.json": _file_entry(run_manifest_path, run_dir=run_dir),
            "freeze_manifest.json": _file_entry(freeze_path, run_dir=run_dir),
            "step4_summary.json": _file_entry(summary_path, run_dir=run_dir),
            "phase05_manifest.json": _file_entry(phase05_path, run_dir=run_dir),
            "phase06_manifest.json": _file_entry(phase06_path, run_dir=run_dir),
            _GEOMETRY_FILE: _file_entry(geometry_path, run_dir=run_dir),
            _RETRIEVAL_FILE: _file_entry(retrieval_path, run_dir=run_dir),
            _THRESHOLD_INSTABILITY_FILE: _file_entry(
                threshold_instability_path,
                run_dir=run_dir,
            ),
            _THRESHOLD_POLICY_FILE: _file_entry(
                threshold_policy_path,
                run_dir=run_dir,
            ),
            _THRESHOLD_POLICY_RHO_FILE: _file_entry(
                threshold_policy_rho_path,
                run_dir=run_dir,
            ),
            _CASES_FILE: _file_entry(cases_path, run_dir=run_dir),
        }
        if diagnostics is not None:
            sources[dataset]["origin_calibration_diagnostics.json"] = _file_entry(
                diagnostics_path,
                run_dir=run_dir,
            )

    return CrossDatasetSaliencyAssociations(
        geometry=pd.concat(geometry_frames, ignore_index=True),
        retrieval=pd.concat(retrieval_frames, ignore_index=True),
        threshold_instability=pd.concat(
            threshold_instability_frames,
            ignore_index=True,
        ),
        threshold_policy_comparison=pd.concat(
            threshold_policy_frames,
            ignore_index=True,
        ),
        threshold_policy_saliency_rho=pd.concat(
            threshold_policy_rho_frames,
            ignore_index=True,
        ),
        representative_cases=pd.concat(case_frames, ignore_index=True),
        source_files=sources,
    )
