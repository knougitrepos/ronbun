from __future__ import annotations

import hashlib
import json
from pathlib import Path
from pprint import pformat
from typing import Any, Mapping, Sequence


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


def _normalize_paper_pq_sdc_settings(
    values: object,
    *,
    label: str,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be a list or tuple")
    normalized: list[tuple[int, int]] = []
    for item in values:
        if isinstance(item, Mapping):
            raw_m = item.get("m")
            raw_nbits = item.get("nbits")
        else:
            try:
                raw_m, raw_nbits = item  # type: ignore[misc]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} entries must be (m, nbits) pairs") from exc
        if isinstance(raw_m, bool) or isinstance(raw_nbits, bool):
            raise ValueError(f"{label} entries must contain integers")
        try:
            setting = (int(raw_m), int(raw_nbits))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} entries must contain integers") from exc
        if setting != (raw_m, raw_nbits):
            raise ValueError(f"{label} entries must contain integers")
        normalized.append(setting)
    resolved = tuple(normalized)
    if resolved not in ((), ((128, 8),)):
        raise ValueError(f"{label} must be empty or exactly ((128, 8),)")
    return resolved


def _completed_run_identity(run_dir: str | Path) -> dict[str, Any]:
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
    try:
        pq_sdc_settings = manifest["config"]["step4"]["compression"][
            "families"
        ]["pq"]["sdc_settings"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"completed run PQ SDC config is missing: {manifest_path}"
        ) from exc
    return {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "model_uid": model_uid,
        "run_dir": str(source),
        "pq_sdc_settings": _normalize_paper_pq_sdc_settings(
            pq_sdc_settings,
            label=f"{dataset_id}.pq_sdc_settings",
        ),
    }


def inspect_step4_retrieval_source(
    run_dir: str | Path,
    *,
    expected_dataset_id: str,
    expected_model_uid: str,
    expected_logical_rows: int,
) -> dict[str, Any]:
    """Resolve a legacy CSV or normalized ledger for report candidate checks."""

    source = Path(run_dir).resolve()
    workflow = source / "artifacts/step2_workflow"
    legacy_csv = workflow / "retrieval_metrics.csv"
    expected_rows = int(expected_logical_rows)
    if expected_rows < 1:
        raise ValueError("expected retrieval row count must be positive")
    if legacy_csv.is_file():
        return {
            "status": "completed",
            "kind": "legacy_retrieval_metrics_csv",
            "path": str(legacy_csv),
            "bytes": int(legacy_csv.stat().st_size),
            "logical_row_count": expected_rows,
        }

    ledger_path = workflow / "retrieval_ledger/manifest.json"
    if not ledger_path.is_file():
        raise FileNotFoundError(
            "Step 4 retrieval source is missing both retrieval_metrics.csv "
            f"and retrieval_ledger/manifest.json: {source}"
        )
    from research.evaluation import load_retrieval_ledger_manifest

    ledger = load_retrieval_ledger_manifest(ledger_path)
    lineage = ledger.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("retrieval ledger lineage must be a mapping")
    observed_dataset = str(lineage.get("dataset_id", ""))
    observed_model = str(lineage.get("model_uid", ""))
    observed_rows = int(ledger.get("logical_row_count", -1))
    if observed_dataset != str(expected_dataset_id):
        raise ValueError(
            "retrieval ledger dataset mismatch: "
            f"{observed_dataset!r} != {expected_dataset_id!r}"
        )
    if observed_model != str(expected_model_uid):
        raise ValueError(
            "retrieval ledger model mismatch: "
            f"{observed_model!r} != {expected_model_uid!r}"
        )
    if observed_rows != expected_rows:
        raise ValueError(
            "retrieval ledger row-count mismatch: "
            f"{observed_rows} != {expected_rows}"
        )

    declared_bytes = int(ledger_path.stat().st_size)
    for condition in ledger["conditions"]:
        if not isinstance(condition, Mapping):
            raise ValueError("retrieval ledger condition must be a mapping")
        entries: list[Mapping[str, object]] = []
        for key in ("core", "topk_detail"):
            entry = condition.get(key)
            if isinstance(entry, Mapping):
                entries.append(entry)
        decisions = condition.get("decisions", ())
        if not isinstance(decisions, list):
            raise ValueError("retrieval ledger decisions must be a list")
        for decision in decisions:
            if not isinstance(decision, Mapping):
                raise ValueError("retrieval ledger decision must be a mapping")
            artifact = decision.get("artifact")
            if isinstance(artifact, Mapping):
                entries.append(artifact)
        for entry in entries:
            relative = entry.get("path")
            expected_bytes = int(entry.get("bytes", -1))
            if not isinstance(relative, str) or not relative or expected_bytes < 0:
                raise ValueError("retrieval ledger artifact entry is invalid")
            artifact_path = (ledger_path.parent / relative).resolve()
            try:
                artifact_path.relative_to(ledger_path.parent.resolve())
            except ValueError as exc:
                raise ValueError(
                    "retrieval ledger artifact escapes its root"
                ) from exc
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            if artifact_path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"retrieval ledger artifact byte count mismatch: {artifact_path}"
                )
            declared_bytes += expected_bytes
    return {
        "status": "completed",
        "kind": "normalized_retrieval_ledger",
        "path": str(ledger_path),
        "bytes": declared_bytes,
        "logical_row_count": observed_rows,
    }


def postprocess_completed_run(
    run_dir: str | Path,
    *,
    refresh_search_spaces: bool = True,
    derive_faithfulness: bool = True,
    derive_survface_faithfulness: bool | None = None,
    faithfulness_options: Mapping[str, Any] | None = None,
    target_fpirs: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.30),
) -> dict[str, Any]:
    """Build derived artifacts without mutating an immutable completed run."""

    identity = _completed_run_identity(run_dir)
    result: dict[str, Any] = {
        "status": "completed",
        "source": identity,
        "search_space_v6_query_gallery_conditions": {"status": "disabled"},
        "faithfulness": {"status": "disabled"},
    }
    if derive_survface_faithfulness is not None:
        derive_faithfulness = bool(derive_survface_faithfulness)
    if refresh_search_spaces:
        from scripts.refresh_step4_search_spaces import refresh

        result["search_space_v6_query_gallery_conditions"] = refresh(
            Path(identity["run_dir"]),
            output_dir=None,
            families=("pca", "pq"),
            target_fpirs=tuple(float(value) for value in target_fpirs),
        )
    if identity["dataset_id"] in {"lfw", "survface", "rfw_custom"}:
        if derive_faithfulness:
            from scripts.derive_survface_faithfulness import (
                derive_open_set_faithfulness as derive,
            )

            options = dict(faithfulness_options or {})
            options.setdefault("maximum_samples", 10_000)
            manifest = derive(identity["run_dir"], **options)
            result["faithfulness"] = {
                "status": "completed",
                "artifact_type": manifest.get("artifact_type"),
                "source_run_id": manifest.get("source_run_id"),
                "model_uid": manifest.get("model_uid"),
                "dataset_id": manifest.get("dataset_id"),
                "evaluation_mode": manifest.get("evaluation_mode"),
                "candidate_count": manifest.get("sampling", {}).get(
                    "candidate_count"
                ),
                "selected_count": manifest.get("sampling", {}).get(
                    "selected_count"
                ),
                "maximum_samples": manifest.get("sampling", {}).get(
                    "maximum_samples"
                ),
            }
        else:
            result["faithfulness"] = {"status": "disabled"}
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
    include_faithfulness: bool,
    write_outputs: bool,
    overwrite_outputs: bool,
    pq_sdc_settings: Sequence[tuple[int, int]] | None = None,
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
    selected_run_paths = dict(selected_runs)
    tinyface_run_dir = selected_run_paths.pop("tinyface", None)
    identities = {
        dataset: _completed_run_identity(run_dir)
        for dataset, run_dir in selected_run_paths.items()
    }
    if not identities:
        raise ValueError("at least one completed open-set run is required")
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
    resolved_tinyface_dir: str | None = None
    tinyface_identity: dict[str, str] | None = None
    if tinyface_run_dir is not None:
        from research.evaluation import load_tinyface_completed_evaluation

        tinyface_evaluation = load_tinyface_completed_evaluation(tinyface_run_dir)
        tinyface_model_uid = str(tinyface_evaluation.manifest.get("model_uid", ""))
        tinyface_family = tinyface_model_uid.split("-", 1)[0].lower()
        if tinyface_family != normalized_model:
            raise ValueError(
                f"TinyFace evaluation model mismatch: {tinyface_family} != {normalized_model}"
            )
        if tinyface_model_uid not in set(model_uids.values()):
            raise ValueError(
                "TinyFace model UID is absent from selected open-set runs: "
                f"{tinyface_model_uid}"
            )
        tinyface_manifest = _read_json(tinyface_evaluation.root / "run_manifest.json")
        resolved_tinyface_dir = str(tinyface_evaluation.root)
        tinyface_identity = {
            "dataset_id": "tinyface",
            "run_id": str(tinyface_manifest["run_id"]),
            "model_uid": tinyface_model_uid,
            "run_dir": resolved_tinyface_dir,
            "pq_sdc_settings": _normalize_paper_pq_sdc_settings(
                tinyface_manifest.get("config", {}).get("pq_sdc_settings"),
                label="tinyface.pq_sdc_settings",
            ),
        }
    selected_contracts = {
        tuple(identity["pq_sdc_settings"])
        for identity in (
            *identities.values(),
            *((tinyface_identity,) if tinyface_identity is not None else ()),
        )
    }
    if len(selected_contracts) != 1:
        raise ValueError("selected runs use mixed PQ SDC settings")
    selected_pq_sdc = next(iter(selected_contracts))
    if pq_sdc_settings is not None:
        requested_pq_sdc = _normalize_paper_pq_sdc_settings(
            pq_sdc_settings,
            label="requested pq_sdc_settings",
        )
        if requested_pq_sdc != selected_pq_sdc:
            raise ValueError(
                "requested PQ SDC settings differ from selected completed runs"
            )
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
                if tuple(matrix_identity["pq_sdc_settings"]) != selected_pq_sdc:
                    raise ValueError(
                        "cross-model matrix PQ SDC settings differ from the "
                        "selected report runs"
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
        "INCLUDE_FAITHFULNESS": bool(include_faithfulness),
        "GENERATE_MISSING_SEARCH_CONDITION_ARTIFACTS": False,
        "WRITE_OUTPUTS": bool(write_outputs),
        "OVERWRITE_COMMON_OUTPUTS": bool(overwrite_outputs),
        "EXPECTED_PQ_SDC_SETTINGS": selected_pq_sdc,
        "RFW_EVALUATION_DIR": resolved_rfw_dir,
        "TINYFACE_EVALUATION_DIR": resolved_tinyface_dir,
        "MODEL_RUN_MATRIX": resolved_cross_model_matrix,
        "REQUIRE_COMPLETE_MODEL_MATRIX": bool(
            resolved_cross_model_matrix
        ),
    }
    source = "CROSS_DATASET_REPORT_PARAMETERS_INJECTED = True\n" + "\n".join(
        f"{name} = {pformat(value, sort_dicts=True)}"
        for name, value in assignments.items()
    ) + "\n"
    returned_identities = dict(identities)
    if tinyface_identity is not None:
        returned_identities["tinyface"] = tinyface_identity
    return source, returned_identities


def run_cross_dataset_report_notebook(
    project_root: str | Path,
    *,
    model_name: str,
    selected_runs: Mapping[str, str | Path],
    include_faithfulness: bool = True,
    write_outputs: bool = True,
    overwrite_outputs: bool = True,
    pq_sdc_settings: Sequence[tuple[int, int]] | None = None,
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
        include_faithfulness=include_faithfulness,
        write_outputs=write_outputs,
        overwrite_outputs=overwrite_outputs,
        pq_sdc_settings=pq_sdc_settings,
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
    datasets = tuple(dataset for dataset in identities if dataset != "tinyface")
    selection_tag = "__".join(
        f"{dataset}-{identities[dataset]['run_id']}" for dataset in datasets
    )
    if "tinyface" in identities:
        selection_tag += f"__tinyface-{identities['tinyface']['run_id']}"
    output_dir = root / "results/paper/common" / normalized_model / selection_tag
    manifest_path = output_dir / "cross_dataset_summary_manifest.json"
    if write_outputs and not manifest_path.is_file():
        raise FileNotFoundError(f"report manifest was not produced: {manifest_path}")
    return {
        "status": "written" if write_outputs else "validated_not_written",
        "notebook": str(notebook_path),
        "datasets": list(datasets),
        "selected_runs": identities,
        "tinyface_run": identities.get("tinyface"),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path) if write_outputs else None,
    }
