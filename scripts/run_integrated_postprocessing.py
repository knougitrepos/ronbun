from __future__ import annotations

import hashlib
import json
from pathlib import Path
from pprint import pformat
from typing import Any, Mapping


REPORT_MODEL_NAMES = {
    "arc": "arcface",
    "arcface": "arcface",
    "ada": "adaface",
    "adaface": "adaface",
    "mag": "magface",
    "magface": "magface",
    "edge": "edgeface",
    "edgeface": "edgeface",
}
EXPECTED_CROSS_MODEL_NAMES = (
    "arcface",
    "adaface",
    "magface",
    "edgeface",
)
EXPECTED_OPEN_SET_DATASETS = ("lfw", "survface", "rfw_custom")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completed_run_identity(run_dir: str | Path) -> dict[str, str]:
    source = Path(run_dir).resolve()
    manifest_path = source / "run_manifest.json"
    freeze_path = source / "artifacts/step2_workflow/freeze_manifest.json"
    if not (source / "COMPLETED").is_file():
        raise FileNotFoundError(f"completed marker is missing: {source}")
    manifest = _read_json(manifest_path)
    freeze = _read_json(freeze_path)
    if manifest.get("status") != "completed":
        raise ValueError(f"source run is not completed: {source}")
    run_id = str(manifest.get("run_id", ""))
    if run_id != str(freeze.get("run_id", "")):
        raise ValueError(f"run/freeze identity mismatch: {source}")
    if freeze.get("fallback_free") is not True:
        raise ValueError(f"source run is not fallback-free: {source}")
    dataset_id = str(freeze.get("dataset_id", ""))
    if dataset_id not in {"lfw", "survface", "rfw_custom"}:
        raise ValueError(f"unsupported dataset_id: {dataset_id!r}")
    model_uid = str(freeze.get("model_uid", ""))
    if not model_uid:
        raise ValueError(f"model_uid is missing: {freeze_path}")
    return {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "model_uid": model_uid,
        "run_dir": str(source),
    }


def postprocess_completed_run(
    run_dir: str | Path,
    *,
    refresh_search_spaces: bool = True,
    derive_survface_faithfulness: bool = True,
    faithfulness_options: Mapping[str, Any] | None = None,
    target_fpirs: tuple[float, ...] = (0.10, 0.01),
) -> dict[str, Any]:
    """Build derived artifacts without mutating an immutable completed run."""

    identity = _completed_run_identity(run_dir)
    result: dict[str, Any] = {
        "status": "completed",
        "source": identity,
        "search_space_v4_multi_fpir": {"status": "disabled"},
        "survface_faithfulness": {"status": "not_applicable"},
    }
    if refresh_search_spaces:
        from scripts.refresh_step4_search_spaces import refresh

        result["search_space_v4_multi_fpir"] = refresh(
            Path(identity["run_dir"]),
            output_dir=None,
            families=("pca", "pq"),
            target_fpirs=tuple(float(value) for value in target_fpirs),
        )
    if identity["dataset_id"] == "survface":
        if derive_survface_faithfulness:
            from scripts.derive_survface_faithfulness import (
                derive_survface_faithfulness as derive,
            )

            options = dict(faithfulness_options or {})
            manifest = derive(identity["run_dir"], **options)
            result["survface_faithfulness"] = {
                "status": "completed",
                "artifact_type": manifest.get("artifact_type"),
                "source_run_id": manifest.get("source_run_id"),
                "model_uid": manifest.get("model_uid"),
            }
        else:
            result["survface_faithfulness"] = {"status": "disabled"}
    return result


def verify_cross_model_faithfulness(
    project_root: str | Path,
    *,
    expected_survface_run_id: str | None = None,
    expected_model_uid: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_dir = root / "results/paper/survface/faithfulness_cross_model_v1"
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return {
            "status": "not_ready",
            "reason": "cross-model faithfulness manifest is absent",
            "output_dir": str(output_dir),
        }
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_type") != "survface_gradcam_faithfulness_cross_model":
        raise ValueError(f"unexpected faithfulness artifact: {manifest_path}")
    model_families = set(manifest.get("model_families", []))
    baseline_families = {
        "arcface",
        "adaface",
        "magface",
    }
    if not baseline_families.issubset(model_families) or not model_families <= (
        baseline_families | {"edgeface"}
    ):
        raise ValueError(
            "cross-model faithfulness must contain the baseline trio and may "
            "add EdgeFace"
        )
    if manifest.get("threshold_independent") is not True:
        raise ValueError("cross-model faithfulness must be threshold-independent")
    for entry in manifest.get("outputs", []):
        path = output_dir / str(entry["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(entry["bytes"])
            or _sha256_file(path) != str(entry["sha256"])
        ):
            raise ValueError(f"faithfulness output failed validation: {path}")
    if expected_survface_run_id is not None or expected_model_uid is not None:
        matching_sources = [
            source
            for source in manifest.get("sources", [])
            if (
                expected_survface_run_id is None
                or str(source.get("source_run_id")) == expected_survface_run_id
            )
            and (
                expected_model_uid is None
                or str(source.get("model_uid")) == expected_model_uid
            )
        ]
        if len(matching_sources) != 1:
            return {
                "status": "not_ready",
                "reason": (
                    "cross-model faithfulness does not contain the selected "
                    "SurvFace run/model"
                ),
                "output_dir": str(output_dir),
            }
    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "paper_eligible": manifest.get("paper_eligible"),
        "strong_faithfulness_pass_count": manifest.get(
            "strong_faithfulness_pass_count"
        ),
    }


def build_report_parameter_source(
    *,
    model_name: str,
    selected_runs: Mapping[str, str | Path],
    include_survface_faithfulness: bool,
    write_outputs: bool,
    overwrite_outputs: bool,
    rfw_evaluation_dir: str | Path | None = None,
    cross_model_run_matrix: Mapping[
        str,
        Mapping[str, str | Path],
    ]
    | None = None,
) -> tuple[str, dict[str, dict[str, str]]]:
    normalized_model = REPORT_MODEL_NAMES.get(str(model_name).lower())
    if normalized_model is None:
        raise ValueError(f"unsupported model_name: {model_name!r}")
    identities = {
        dataset: _completed_run_identity(run_dir)
        for dataset, run_dir in selected_runs.items()
    }
    if not identities:
        raise ValueError("at least one completed run is required")
    if set(identities) - {"lfw", "survface", "rfw_custom"}:
        raise ValueError(f"unsupported dataset keys: {sorted(identities)}")
    for dataset, identity in identities.items():
        if identity["dataset_id"] != dataset:
            raise ValueError(
                f"selected run key/manifest mismatch: {dataset} != "
                f"{identity['dataset_id']}"
            )
        family = identity["model_uid"].split("-", 1)[0].lower()
        if family != normalized_model:
            raise ValueError(
                f"selected run model mismatch: {family} != {normalized_model}"
            )
    datasets = tuple(identities)
    model_uids = {
        dataset: identity["model_uid"] for dataset, identity in identities.items()
    }
    run_ids = {
        dataset: identity["run_id"] for dataset, identity in identities.items()
    }
    resolved_rfw_dir: str | None = None
    if rfw_evaluation_dir is not None:
        from research.experiments import load_rfw_frozen_codec_evaluation

        rfw_evaluation = load_rfw_frozen_codec_evaluation(rfw_evaluation_dir)
        rfw_model_uid = str(rfw_evaluation.manifest.get("model_uid", ""))
        rfw_family = rfw_model_uid.split("-", 1)[0].lower()
        if rfw_family != normalized_model:
            raise ValueError(
                f"RFW evaluation model mismatch: {rfw_family} != {normalized_model}"
            )
        selected_model_uids = {identity["model_uid"] for identity in identities.values()}
        if rfw_model_uid not in selected_model_uids:
            raise ValueError(
                "RFW evaluation model UID is absent from selected open-set runs: "
                f"{rfw_model_uid}"
            )
        resolved_rfw_dir = str(rfw_evaluation.root)
    resolved_cross_model_matrix: dict[str, dict[str, str]] = {}
    if cross_model_run_matrix:
        for supplied_model, supplied_runs in cross_model_run_matrix.items():
            matrix_model = REPORT_MODEL_NAMES.get(str(supplied_model).lower())
            if matrix_model is None:
                raise ValueError(
                    f"unsupported cross-model matrix model: {supplied_model!r}"
                )
            if matrix_model in resolved_cross_model_matrix:
                raise ValueError(
                    f"duplicate cross-model matrix model: {matrix_model!r}"
                )
            if set(supplied_runs) != set(EXPECTED_OPEN_SET_DATASETS):
                raise ValueError(
                    f"{matrix_model}: cross-model matrix requires datasets "
                    f"{EXPECTED_OPEN_SET_DATASETS}"
                )
            resolved_cross_model_matrix[matrix_model] = {}
            for matrix_dataset in EXPECTED_OPEN_SET_DATASETS:
                matrix_identity = _completed_run_identity(
                    supplied_runs[matrix_dataset]
                )
                if matrix_identity["dataset_id"] != matrix_dataset:
                    raise ValueError(
                        "cross-model matrix dataset key/manifest mismatch: "
                        f"{matrix_dataset} != {matrix_identity['dataset_id']}"
                    )
                matrix_family = matrix_identity["model_uid"].split(
                    "-", 1
                )[0].lower()
                if matrix_family != matrix_model:
                    raise ValueError(
                        "cross-model matrix model mismatch: "
                        f"{matrix_family} != {matrix_model}"
                    )
                resolved_cross_model_matrix[matrix_model][matrix_dataset] = (
                    matrix_identity["run_dir"]
                )
        if set(resolved_cross_model_matrix) != set(
            EXPECTED_CROSS_MODEL_NAMES
        ):
            raise ValueError(
                "cross-model matrix requires ArcFace, AdaFace, MagFace, and "
                "EdgeFace completed runs"
            )
    assignments = {
        "MODEL_NAME": normalized_model,
        "DATASETS": datasets,
        "MODEL_UIDS": model_uids,
        "RUN_IDS": run_ids,
        "INCLUDE_SURVFACE_FAITHFULNESS": bool(
            include_survface_faithfulness and "survface" in datasets
        ),
        "WRITE_OUTPUTS": bool(write_outputs),
        "OVERWRITE_COMMON_OUTPUTS": bool(overwrite_outputs),
        "RFW_EVALUATION_DIR": resolved_rfw_dir,
        "MODEL_RUN_MATRIX": resolved_cross_model_matrix,
        "REQUIRE_COMPLETE_MODEL_MATRIX": bool(
            resolved_cross_model_matrix
        ),
    }
    source = "CROSS_DATASET_REPORT_PARAMETERS_INJECTED = True\n" + "\n".join(
        f"{name} = {pformat(value, sort_dicts=True)}"
        for name, value in assignments.items()
    ) + "\n"
    return source, identities


def run_cross_dataset_report_notebook(
    project_root: str | Path,
    *,
    model_name: str,
    selected_runs: Mapping[str, str | Path],
    include_survface_faithfulness: bool = True,
    write_outputs: bool = True,
    overwrite_outputs: bool = True,
    rfw_evaluation_dir: str | Path | None = None,
    cross_model_run_matrix: Mapping[
        str,
        Mapping[str, str | Path],
    ]
    | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Execute the output-free report notebook in memory with explicit selectors."""

    import nbformat
    from nbclient import NotebookClient

    root = Path(project_root).resolve()
    parameter_source, identities = build_report_parameter_source(
        model_name=model_name,
        selected_runs=selected_runs,
        include_survface_faithfulness=include_survface_faithfulness,
        write_outputs=write_outputs,
        overwrite_outputs=overwrite_outputs,
        rfw_evaluation_dir=rfw_evaluation_dir,
        cross_model_run_matrix=cross_model_run_matrix,
    )
    notebook_path = root / "notebooks/common/reports/00_cross_dataset_results.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    notebook.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            parameter_source,
            metadata={"tags": ["injected-parameters"]},
        ),
    )
    client = NotebookClient(
        notebook,
        timeout=timeout_seconds,
        kernel_name=notebook.metadata.kernelspec.name,
        resources={"metadata": {"path": str(root)}},
        allow_errors=False,
    )
    client.execute()

    normalized_model = REPORT_MODEL_NAMES[str(model_name).lower()]
    datasets = tuple(identities)
    selection_tag = "__".join(
        f"{dataset}-{identities[dataset]['run_id']}" for dataset in datasets
    )
    output_dir = root / "results/paper/common" / normalized_model / selection_tag
    manifest_path = output_dir / "cross_dataset_summary_manifest.json"
    if write_outputs and not manifest_path.is_file():
        raise FileNotFoundError(f"report manifest was not produced: {manifest_path}")
    return {
        "status": "written" if write_outputs else "validated_not_written",
        "notebook": str(notebook_path),
        "datasets": list(datasets),
        "selected_runs": identities,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path) if write_outputs else None,
    }
