from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from research.runtime.hashing import sha256_file


_WORKFLOW_SUBDIR = Path("artifacts/step2_workflow")
_GEOMETRY_FILE = "saliency_geometry_associations.csv"
_RETRIEVAL_FILE = "saliency_retrieval_associations.csv"
_CASES_FILE = "representative_cases.csv"


@dataclass(frozen=True)
class CrossDatasetSaliencyAssociations:
    geometry: pd.DataFrame
    retrieval: pd.DataFrame
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


def load_cross_dataset_saliency_associations(
    selected_runs: Mapping[str, str | Path],
    *,
    expected_model_uids: Mapping[str, str],
    expected_run_ids: Mapping[str, str],
    expected_target_fpirs: tuple[float, ...] = (0.10, 0.01),
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

        phase05_path = _latest_completed_attempt(
            run_dir,
            "05_saliency_compression_join",
        )
        phase06_path = _latest_completed_attempt(
            run_dir,
            "06_representative_case_visualization",
        )
        workflow = run_dir / _WORKFLOW_SUBDIR
        geometry_path = workflow / _GEOMETRY_FILE
        retrieval_path = workflow / _RETRIEVAL_FILE
        cases_path = workflow / _CASES_FILE
        for path in (geometry_path, retrieval_path, cases_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        geometry = pd.read_csv(geometry_path, low_memory=False)
        retrieval = pd.read_csv(retrieval_path, low_memory=False)
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
        if int(summary.get("geometry_association_rows", -1)) != len(geometry):
            raise ValueError(f"{dataset}: geometry association row count mismatch")
        if int(summary.get("retrieval_association_rows", -1)) != len(retrieval):
            raise ValueError(f"{dataset}: retrieval association row count mismatch")
        if int(summary.get("representative_cases", -1)) != len(cases):
            raise ValueError(f"{dataset}: representative case row count mismatch")

        geometry_frames.append(_with_lineage(geometry, run_id=run_id, run_dir=run_dir))
        retrieval_frames.append(_with_lineage(retrieval, run_id=run_id, run_dir=run_dir))
        case_frames.append(_with_lineage(cases, run_id=run_id, run_dir=run_dir))
        sources[dataset] = {
            "run_manifest.json": _file_entry(run_manifest_path, run_dir=run_dir),
            "freeze_manifest.json": _file_entry(freeze_path, run_dir=run_dir),
            "step4_summary.json": _file_entry(summary_path, run_dir=run_dir),
            "phase05_manifest.json": _file_entry(phase05_path, run_dir=run_dir),
            "phase06_manifest.json": _file_entry(phase06_path, run_dir=run_dir),
            _GEOMETRY_FILE: _file_entry(geometry_path, run_dir=run_dir),
            _RETRIEVAL_FILE: _file_entry(retrieval_path, run_dir=run_dir),
            _CASES_FILE: _file_entry(cases_path, run_dir=run_dir),
        }

    return CrossDatasetSaliencyAssociations(
        geometry=pd.concat(geometry_frames, ignore_index=True),
        retrieval=pd.concat(retrieval_frames, ignore_index=True),
        representative_cases=pd.concat(case_frames, ignore_index=True),
        source_files=sources,
    )
