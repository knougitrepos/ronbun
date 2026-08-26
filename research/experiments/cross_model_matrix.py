from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.evaluation.metrics import rate_ratio_matches_counts_or_compact_csv
from research.evaluation.search_conditions import (
    ALL_SEARCH_MODES,
    search_condition,
    threshold_policies_for_search_mode,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_MODEL_FAMILIES = ("arcface", "adaface", "magface", "edgeface")
SUPPORTED_OPEN_SET_DATASETS = ("lfw", "survface", "rfw_custom")
TARGET_FPIRS = (0.01, 0.05, 0.10, 0.20, 0.30)

SEARCH_SPACE_DIRECTORY = "search_space_v6_query_gallery_conditions"
SEARCH_SPACE_SCHEMA_VERSION = 6
SEARCH_SPACE_ARTIFACT_TYPE = "step4_search_space_query_gallery_conditions_v6"
FAMILY_ARTIFACT_TYPE = "step4_search_space_query_gallery_conditions_family_v6"

MATRIX_SCHEMA_VERSION = 3
MATRIX_ARTIFACT_TYPE = "cross_model_open_set_completed_run_matrix_v3"

REQUIRED_SOURCE_FILES = {
    "run_manifest.json",
    "freeze_manifest.json",
    "prepared_population/manifest.json",
    "selected_manifest.csv",
}
REQUIRED_FAMILIES = {"pca", "pq"}
REQUIRED_SEARCH_MODES = set(ALL_SEARCH_MODES)
REQUIRED_THRESHOLD_POLICIES = {
    "frozen_origin",
    "recalibrated_compressed",
}

COMPRESSION_IDENTITY_COLUMNS = (
    "dataset",
    "model_uid",
    "run_id",
    "extraction_uid",
    "origin_embedding_artifact_uid",
    "compression_family",
    "compression_profile",
)
RETRIEVAL_IDENTITY_COLUMNS = (
    *COMPRESSION_IDENTITY_COLUMNS,
    "search_mode",
    "threshold_policy",
    "target_fpir",
)

REQUIRED_COMPRESSION_COLUMNS = {
    *COMPRESSION_IDENTITY_COLUMNS,
    "origin_fallback_count",
    "sample_count",
}
REQUIRED_RETRIEVAL_COLUMNS = {
    *RETRIEVAL_IDENTITY_COLUMNS,
    "query_representation",
    "gallery_representation",
    "distance_function",
    "compressed_score_space",
    "score_spaces_comparable",
    "frozen_origin_threshold_applicable",
    "origin_fallback_count",
    "origin_dir_rank1_count",
    "origin_dir_rank1_denominator",
    "origin_dir_rank1",
    "origin_dir_rank1_wilson95_low",
    "origin_dir_rank1_wilson95_high",
    "compressed_dir_rank1_count",
    "compressed_dir_rank1_denominator",
    "compressed_dir_rank1",
    "compressed_dir_rank1_wilson95_low",
    "compressed_dir_rank1_wilson95_high",
    "compressed_minus_origin_dir_rank1",
    "compressed_minus_origin_dir_rank1_paired_bootstrap95_low",
    "compressed_minus_origin_dir_rank1_paired_bootstrap95_high",
    "tpir_rank",
    "origin_tpir20_count",
    "origin_tpir20_denominator",
    "origin_tpir20",
    "origin_tpir_at_rank_k_wilson95_low",
    "origin_tpir_at_rank_k_wilson95_high",
    "compressed_tpir20_count",
    "compressed_tpir20_denominator",
    "compressed_tpir20",
    "compressed_tpir_at_rank_k_wilson95_low",
    "compressed_tpir_at_rank_k_wilson95_high",
    "compressed_tpir20_retention",
    "origin_closed_set_rank20_recall",
    "compressed_closed_set_rank20_recall",
    "compressed_minus_origin_tpir_at_rank_k",
    "compressed_minus_origin_tpir_at_rank_k_paired_bootstrap95_low",
    "compressed_minus_origin_tpir_at_rank_k_paired_bootstrap95_high",
    "origin_false_accept_count",
    "origin_fpir_denominator",
    "origin_fpir",
    "origin_realized_fpir",
    "origin_fpir_wilson95_low",
    "origin_fpir_wilson95_high",
    "compressed_false_accept_count",
    "compressed_fpir_denominator",
    "compressed_fpir",
    "compressed_realized_fpir",
    "compressed_fpir_wilson95_low",
    "compressed_fpir_wilson95_high",
    "compressed_minus_origin_fpir",
    "compressed_minus_origin_fpir_paired_bootstrap95_low",
    "compressed_minus_origin_fpir_paired_bootstrap95_high",
    "confidence_interval_unit",
    "rate_confidence_interval_method",
    "difference_confidence_interval_method",
    "difference_confidence_interval_resamples",
    "difference_confidence_interval_random_seed",
}


@dataclass(frozen=True)
class CrossModelOpenSetMatrix:
    """Verified 4-model x 3-dataset completed-run summaries.

    All frames are copies owned by this result.  ``joined_summary`` has one row
    per retrieval result and attaches the matching compression-summary fields.
    The loader never discovers or selects runs automatically.
    """

    compression_summary: pd.DataFrame
    retrieval_summary: pd.DataFrame
    joined_summary: pd.DataFrame
    selection_manifest: dict[str, Any]


@dataclass(frozen=True)
class CrossModelOpenSetMatrixWriteResult:
    output_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _RunSelection:
    model_family: str
    dataset_id: str
    run_dir: Path
    run_id: str
    model_uid: str
    checkpoint_sha256: str
    extraction_uid: str
    origin_embedding_artifact_uid: str
    run_manifest_sha256: str
    freeze_manifest_sha256: str
    summary_manifest_sha256: str
    compression_summary_sha256: str
    retrieval_summary_sha256: str

    def content_identity(self) -> dict[str, str]:
        return {
            "model_family": self.model_family,
            "dataset_id": self.dataset_id,
            "run_id": self.run_id,
            "model_uid": self.model_uid,
            "checkpoint_sha256": self.checkpoint_sha256,
            "extraction_uid": self.extraction_uid,
            "origin_embedding_artifact_uid": (self.origin_embedding_artifact_uid),
            "run_manifest_sha256": self.run_manifest_sha256,
            "freeze_manifest_sha256": self.freeze_manifest_sha256,
            "summary_manifest_sha256": self.summary_manifest_sha256,
            "compression_summary_sha256": self.compression_summary_sha256,
            "retrieval_summary_sha256": self.retrieval_summary_sha256,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _require_text(value: object, *, field: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} must be a non-empty string")
    return normalized


def _normalize_matrix(
    model_run_matrix: Mapping[str, Mapping[str, str | Path]],
) -> dict[str, dict[str, Path]]:
    if not isinstance(model_run_matrix, Mapping):
        raise TypeError("model_run_matrix must be a nested mapping")
    normalized: dict[str, dict[str, Path]] = {}
    for raw_model, raw_datasets in model_run_matrix.items():
        model = str(raw_model).strip().lower()
        if model in normalized:
            raise ValueError(f"duplicate normalized model key: {model!r}")
        if not isinstance(raw_datasets, Mapping):
            raise TypeError(f"dataset selections for {model!r} must be a mapping")
        datasets: dict[str, Path] = {}
        for raw_dataset, raw_run_dir in raw_datasets.items():
            dataset = str(raw_dataset).strip().lower()
            if dataset in datasets:
                raise ValueError(
                    f"duplicate normalized dataset key for {model}: {dataset!r}"
                )
            if not isinstance(raw_run_dir, (str, Path)):
                raise TypeError(
                    "matrix values must be explicit run directory paths; "
                    f"got {type(raw_run_dir).__name__} for {model}/{dataset}"
                )
            datasets[dataset] = Path(raw_run_dir).expanduser().resolve()
        normalized[model] = datasets

    observed_models = set(normalized)
    expected_models = set(SUPPORTED_MODEL_FAMILIES)
    if observed_models != expected_models:
        raise ValueError(
            "model matrix must contain exactly "
            f"{sorted(expected_models)}; got {sorted(observed_models)}"
        )
    expected_datasets = set(SUPPORTED_OPEN_SET_DATASETS)
    for model, datasets in normalized.items():
        if set(datasets) != expected_datasets:
            raise ValueError(
                f"{model} must contain exactly {sorted(expected_datasets)}; "
                f"got {sorted(datasets)}"
            )
    return normalized


def _resolve_declared_path(
    entry: Mapping[str, object],
    *,
    root: Path,
    label: str,
) -> Path:
    raw_path = _require_text(entry.get("path"), field=f"{label}.path")
    candidate = Path(raw_path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    return resolved


def _verify_file_entry(
    entry: object,
    *,
    root: Path,
    label: str,
    expected_path: Path | None = None,
) -> tuple[Path, str]:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{label} must be a file-entry object")
    path = _resolve_declared_path(entry, root=root, label=label)
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(
            f"{label} path lineage mismatch: {path} != {expected_path.resolve()}"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_bytes = int(entry.get("bytes", -1))
    if expected_bytes < 0 or path.stat().st_size != expected_bytes:
        raise ValueError(f"{label} byte-size mismatch: {path}")
    expected_sha256 = _require_text(entry.get("sha256"), field=f"{label}.sha256")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return path, actual_sha256


def _workflow_paths(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    step4 = config.get("step4")
    if not isinstance(step4, Mapping):
        raise ValueError("source run manifest is missing config.step4")
    workflow = step4.get("workflow")
    if not isinstance(workflow, Mapping):
        raise ValueError("source run manifest is missing config.step4.workflow")
    workflow_root = run_dir / _require_text(
        workflow.get("artifact_subdir"), field="workflow.artifact_subdir"
    )
    prepared_dir = workflow_root / _require_text(
        workflow.get("prepared_population_dir"),
        field="workflow.prepared_population_dir",
    )
    return {
        "freeze_manifest.json": workflow_root
        / _require_text(
            workflow.get("freeze_manifest_path"),
            field="workflow.freeze_manifest_path",
        ),
        "prepared_population/manifest.json": prepared_dir / "manifest.json",
        "selected_manifest.csv": workflow_root
        / _require_text(
            workflow.get("selected_manifest_path"),
            field="workflow.selected_manifest_path",
        ),
    }


def _verify_source_run(
    run_dir: Path,
    *,
    model_family: str,
    dataset_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    run_manifest_path = run_dir / "run_manifest.json"
    completed_path = run_dir / "COMPLETED"
    if not run_manifest_path.is_file() or not completed_path.is_file():
        raise FileNotFoundError(
            f"completed source run requires run_manifest.json and COMPLETED: {run_dir}"
        )
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("status") != "completed":
        raise ValueError(f"source run status is not completed: {run_dir}")
    run_id = _require_text(run_manifest.get("run_id"), field="run_manifest.run_id")
    config = run_manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("source run manifest is missing its frozen config")
    observed_dataset = _require_text(
        config.get("dataset_id"), field="run_manifest.config.dataset_id"
    )
    if observed_dataset != dataset_id:
        raise ValueError(
            f"selected dataset differs from source run: {dataset_id} != "
            f"{observed_dataset}"
        )
    if dataset_id == "rfw_custom":
        step4 = config.get("step4")
        evaluation = step4.get("evaluation") if isinstance(step4, Mapping) else None
        if not isinstance(evaluation, Mapping) or evaluation.get(
            "rfw_custom_calibration_gallery_policy"
        ) != "evaluation_group_matched":
            raise ValueError(
                "RFW-Custom source run predates gallery-size-matched calibration"
            )
    model_uid = _require_text(
        config.get("model_uid"), field="run_manifest.config.model_uid"
    )
    observed_family = model_uid.split("-", maxsplit=1)[0].strip().lower()
    if observed_family != model_family:
        raise ValueError(
            f"selected model differs from source checkpoint: {model_family} != "
            f"{observed_family}"
        )

    workflow_paths = _workflow_paths(run_dir, config)
    freeze_path = workflow_paths["freeze_manifest.json"]
    if not freeze_path.is_file():
        raise FileNotFoundError(freeze_path)
    freeze = _read_json(freeze_path)
    expected_identity = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "model_uid": model_uid,
    }
    for field, expected in expected_identity.items():
        if _require_text(freeze.get(field), field=f"freeze.{field}") != expected:
            raise ValueError(f"freeze {field} differs from source run")
    if freeze.get("fallback_free") is not True:
        raise ValueError("source freeze manifest must declare fallback_free=true")
    checkpoint_sha256 = _require_text(
        freeze.get("checkpoint_sha256"), field="freeze.checkpoint_sha256"
    )
    if len(checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in checkpoint_sha256
    ):
        raise ValueError("freeze.checkpoint_sha256 must be a SHA-256 hex digest")
    selected_sha256 = _require_text(
        freeze.get("selected_manifest_sha256"),
        field="freeze.selected_manifest_sha256",
    )
    selected_path = workflow_paths["selected_manifest.csv"]
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    if selected_sha256 != _sha256_file(selected_path):
        raise ValueError("freeze selected_manifest_sha256 differs from source file")
    return (
        run_manifest,
        freeze,
        {
            "run_manifest.json": run_manifest_path,
            **workflow_paths,
        },
    )


def _verify_family_artifacts(
    summary_dir: Path,
    summary_manifest: Mapping[str, Any],
    *,
    project_root: Path,
    dataset_id: str,
    model_uid: str,
    run_id: str,
) -> None:
    family_entries = summary_manifest.get("family_manifests")
    if (
        not isinstance(family_entries, Mapping)
        or set(family_entries) != REQUIRED_FAMILIES
    ):
        raise ValueError(
            "summary_manifest must pin exactly PCA and PQ family manifests"
        )
    for family in sorted(REQUIRED_FAMILIES):
        family_manifest_path = summary_dir / family / "family_manifest.json"
        _verify_file_entry(
            family_entries[family],
            root=project_root,
            label=f"family_manifests.{family}",
            expected_path=family_manifest_path,
        )
        family_manifest = _read_json(family_manifest_path)
        expected_fields = {
            "schema_version": SEARCH_SPACE_SCHEMA_VERSION,
            "artifact_type": FAMILY_ARTIFACT_TYPE,
            "family": family,
            "dataset_id": dataset_id,
            "model_uid": model_uid,
            "source_run_id": run_id,
        }
        for field, expected in expected_fields.items():
            if family_manifest.get(field) != expected:
                raise ValueError(
                    f"{family} family manifest {field} mismatch: "
                    f"{family_manifest.get(field)!r} != {expected!r}"
                )
        _validate_target_fpirs(
            family_manifest.get("target_fpirs"),
            label=f"{family} family target_fpirs",
        )
        outputs = family_manifest.get("outputs")
        if not isinstance(outputs, Mapping) or not outputs:
            raise ValueError(f"{family} family manifest has no outputs")
        for name, entry in outputs.items():
            expected_path = summary_dir / family / str(name)
            _verify_file_entry(
                entry,
                root=summary_dir / family,
                label=f"{family}.outputs.{name}",
                expected_path=expected_path,
            )


def _validate_target_fpirs(values: object, *, label: str) -> None:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be a target-FPIR sequence")
    targets = tuple(float(value) for value in values)
    if (
        len(targets) != len(TARGET_FPIRS)
        or len(set(targets)) != len(TARGET_FPIRS)
        or set(targets) != set(TARGET_FPIRS)
    ):
        raise ValueError(
            f"{label} must contain exactly {set(TARGET_FPIRS)}; got {targets}"
        )


def _require_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _require_single_value(
    frame: pd.DataFrame,
    column: str,
    expected: str,
    *,
    label: str,
) -> None:
    observed = set(frame[column].astype(str))
    if observed != {expected}:
        raise ValueError(
            f"{label}.{column} lineage mismatch: {sorted(observed)} != {expected!r}"
        )


def _numeric(frame: pd.DataFrame, column: str, *, label: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{label}.{column} contains non-finite values")
    return values


def _validate_retrieval_statistics(frame: pd.DataFrame, *, label: str) -> None:
    if frame["confidence_interval_unit"].ne("probe").any():
        raise ValueError(f"{label} confidence intervals must use probe units")
    if frame["rate_confidence_interval_method"].ne("wilson_score").any():
        raise ValueError(f"{label} rate confidence intervals must use Wilson score")
    if (
        frame["difference_confidence_interval_method"]
        .ne("paired_nonparametric_bootstrap_percentile")
        .any()
    ):
        raise ValueError(f"{label} differences must use paired bootstrap intervals")
    repeats = _numeric(frame, "difference_confidence_interval_resamples", label=label)
    seeds = _numeric(frame, "difference_confidence_interval_random_seed", label=label)
    if repeats.ne(2000).any() or seeds.ne(42).any():
        raise ValueError(
            f"{label} paired bootstrap must use 2000 resamples and seed 42"
        )

    rate_specs = (
        (
            "dir_rank1",
            "origin_dir_rank1_count",
            "origin_dir_rank1_denominator",
            "compressed_dir_rank1_count",
            "compressed_dir_rank1_denominator",
        ),
        (
            "fpir",
            "origin_false_accept_count",
            "origin_fpir_denominator",
            "compressed_false_accept_count",
            "compressed_fpir_denominator",
        ),
    )
    for (
        metric,
        origin_count_name,
        origin_denom_name,
        compressed_count_name,
        compressed_denom_name,
    ) in rate_specs:
        origin_rate = _numeric(frame, f"origin_{metric}", label=label)
        compressed_rate = _numeric(frame, f"compressed_{metric}", label=label)
        delta = _numeric(frame, f"compressed_minus_origin_{metric}", label=label)
        if not np.allclose(
            delta,
            compressed_rate - origin_rate,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{label} {metric} delta does not reconcile")

        for source, rate, count_name, denominator_name in (
            ("origin", origin_rate, origin_count_name, origin_denom_name),
            (
                "compressed",
                compressed_rate,
                compressed_count_name,
                compressed_denom_name,
            ),
        ):
            count = _numeric(frame, count_name, label=label)
            denominator = _numeric(frame, denominator_name, label=label)
            if count.mod(1).ne(0).any() or denominator.mod(1).ne(0).any():
                raise ValueError(f"{label} {source} {metric} counts must be integers")
            if (
                (denominator <= 0).any()
                or (count < 0).any()
                or (count > denominator).any()
            ):
                raise ValueError(f"{label} {source} {metric} counts are invalid")
            if not np.allclose(rate, count / denominator, rtol=0.0, atol=1e-12):
                raise ValueError(f"{label} {source} {metric} rate/count mismatch")
            low = _numeric(frame, f"{source}_{metric}_wilson95_low", label=label)
            high = _numeric(frame, f"{source}_{metric}_wilson95_high", label=label)
            if ((low < 0.0) | (high > 1.0) | (low > rate) | (rate > high)).any():
                raise ValueError(
                    f"{label} {source} {metric} Wilson interval is invalid"
                )
        origin_denominator = _numeric(frame, origin_denom_name, label=label)
        compressed_denominator = _numeric(frame, compressed_denom_name, label=label)
        if not origin_denominator.equals(compressed_denominator):
            raise ValueError(f"{label} origin/compressed {metric} denominators differ")
        delta_low = _numeric(
            frame,
            f"compressed_minus_origin_{metric}_paired_bootstrap95_low",
            label=label,
        )
        delta_high = _numeric(
            frame,
            f"compressed_minus_origin_{metric}_paired_bootstrap95_high",
            label=label,
        )
        if (delta_low > delta_high).any():
            raise ValueError(f"{label} {metric} paired-bootstrap interval is inverted")

    for source in ("origin", "compressed"):
        realized = _numeric(frame, f"{source}_realized_fpir", label=label)
        fpir = _numeric(frame, f"{source}_fpir", label=label)
        if not np.allclose(realized, fpir, rtol=0.0, atol=1e-12):
            raise ValueError(f"{label} {source} realized FPIR alias drifted")

    if _numeric(frame, "tpir_rank", label=label).ne(20).any():
        raise ValueError(f"{label} TPIR rank must be 20")
    origin_tpir = _numeric(frame, "origin_tpir20", label=label)
    compressed_tpir = _numeric(frame, "compressed_tpir20", label=label)
    for source, rate in (
        ("origin", origin_tpir),
        ("compressed", compressed_tpir),
    ):
        count = _numeric(frame, f"{source}_tpir20_count", label=label)
        denominator = _numeric(
            frame, f"{source}_tpir20_denominator", label=label
        )
        if not np.allclose(rate, count / denominator, rtol=0.0, atol=1e-12):
            raise ValueError(f"{label} {source} TPIR20 rate/count mismatch")
        low = _numeric(
            frame, f"{source}_tpir_at_rank_k_wilson95_low", label=label
        )
        high = _numeric(
            frame, f"{source}_tpir_at_rank_k_wilson95_high", label=label
        )
        if ((low > rate) | (rate > high)).any():
            raise ValueError(f"{label} {source} TPIR20 Wilson interval is invalid")
    tpir_delta = _numeric(
        frame, "compressed_minus_origin_tpir_at_rank_k", label=label
    )
    if not np.allclose(
        tpir_delta,
        compressed_tpir - origin_tpir,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"{label} TPIR20 delta does not reconcile")
    if not rate_ratio_matches_counts_or_compact_csv(
        frame["compressed_tpir20_retention"],
        reference_successes=frame["origin_tpir20_count"],
        reference_totals=frame["origin_tpir20_denominator"],
        candidate_successes=frame["compressed_tpir20_count"],
        candidate_totals=frame["compressed_tpir20_denominator"],
    ):
        raise ValueError(f"{label} TPIR20 retention does not reconcile")


def _validate_mode_coverage(frame: pd.DataFrame, *, label: str) -> None:
    observed_modes = set(frame["search_mode"].astype(str))
    if observed_modes != REQUIRED_SEARCH_MODES:
        raise ValueError(
            f"{label} search-mode coverage mismatch: {sorted(observed_modes)}"
        )
    observed_policies = set(frame["threshold_policy"].astype(str))
    if observed_policies != REQUIRED_THRESHOLD_POLICIES:
        raise ValueError(
            f"{label} threshold-policy coverage mismatch: {sorted(observed_policies)}"
        )
    _validate_target_fpirs(
        sorted(set(_numeric(frame, "target_fpir", label=label))),
        label=f"{label}.target_fpir",
    )

    for (family, profile, mode), group in frame.groupby(
        ["compression_family", "compression_profile", "search_mode"],
        sort=False,
        dropna=False,
    ):
        family = str(family)
        mode = str(mode)
        if mode.startswith("pca_") != (family == "pca"):
            raise ValueError(f"{label} compression family/search mode mismatch")
        if mode.startswith("pq_") != (family == "pq"):
            raise ValueError(f"{label} compression family/search mode mismatch")
        expected_policies = set(threshold_policies_for_search_mode(mode))
        condition = search_condition(mode)
        for column, expected in {
            "query_representation": condition.query_representation,
            "gallery_representation": condition.gallery_representation,
            "distance_function": condition.distance_function,
            "compressed_score_space": condition.compressed_score_space,
            "score_spaces_comparable": condition.score_spaces_comparable,
            "frozen_origin_threshold_applicable": (
                condition.frozen_origin_threshold_applicable
            ),
        }.items():
            if set(group[column].tolist()) != {expected}:
                raise ValueError(f"{label} {mode} {column} mismatch")
        observed_pairs = {
            (str(row.threshold_policy), float(row.target_fpir))
            for row in group[["threshold_policy", "target_fpir"]].itertuples(
                index=False
            )
        }
        expected_pairs = {
            (policy, target) for policy in expected_policies for target in TARGET_FPIRS
        }
        if observed_pairs != expected_pairs:
            raise ValueError(
                f"{label} incomplete policy/FPIR grid for {family}/{profile}/{mode}"
            )


def _load_one_selection(
    run_dir: Path,
    *,
    model_family: str,
    dataset_id: str,
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, _RunSelection, dict[str, Any]]:
    run_manifest, freeze, source_paths = _verify_source_run(
        run_dir,
        model_family=model_family,
        dataset_id=dataset_id,
    )
    run_id = _require_text(run_manifest.get("run_id"), field="run_manifest.run_id")
    config = run_manifest["config"]
    model_uid = _require_text(config.get("model_uid"), field="config.model_uid")
    summary_dir = (
        project_root
        / "results"
        / "paper"
        / dataset_id
        / run_id
        / SEARCH_SPACE_DIRECTORY
    ).resolve()
    summary_manifest_path = summary_dir / "summary_manifest.json"
    if (
        summary_dir.name != SEARCH_SPACE_DIRECTORY
        or not summary_manifest_path.is_file()
    ):
        raise FileNotFoundError(summary_manifest_path)
    summary_manifest = _read_json(summary_manifest_path)
    expected_summary_fields = {
        "schema_version": SEARCH_SPACE_SCHEMA_VERSION,
        "artifact_type": SEARCH_SPACE_ARTIFACT_TYPE,
        "dataset_id": dataset_id,
        "model_uid": model_uid,
        "run_id": run_id,
        "source_run_id": run_id,
        "source_run_preserved_immutable": True,
        "compact_only": True,
        "producer_script": "scripts/refresh_step4_search_spaces.py",
    }
    for field, expected in expected_summary_fields.items():
        if summary_manifest.get(field) != expected:
            raise ValueError(
                f"summary_manifest {field} mismatch for {model_family}/{dataset_id}: "
                f"{summary_manifest.get(field)!r} != {expected!r}"
            )
    _validate_target_fpirs(
        summary_manifest.get("target_fpirs"), label="summary_manifest.target_fpirs"
    )

    source_entries = summary_manifest.get("source_files")
    if (
        not isinstance(source_entries, Mapping)
        or set(source_entries) != REQUIRED_SOURCE_FILES
    ):
        raise ValueError(
            "summary_manifest.source_files must pin the complete source lineage"
        )
    source_hashes: dict[str, str] = {}
    for name in sorted(REQUIRED_SOURCE_FILES):
        _, source_hashes[name] = _verify_file_entry(
            source_entries[name],
            root=project_root,
            label=f"source_files.{name}",
            expected_path=source_paths[name],
        )

    _verify_family_artifacts(
        summary_dir,
        summary_manifest,
        project_root=project_root,
        dataset_id=dataset_id,
        model_uid=model_uid,
        run_id=run_id,
    )
    output_entries = summary_manifest.get("output_files")
    expected_outputs = {"compression_summary.csv", "retrieval_summary.csv"}
    if (
        not isinstance(output_entries, Mapping)
        or set(output_entries) != expected_outputs
    ):
        raise ValueError("summary_manifest must pin exactly both compact CSV outputs")
    verified_outputs: dict[str, tuple[Path, str]] = {}
    for name in sorted(expected_outputs):
        verified_outputs[name] = _verify_file_entry(
            output_entries[name],
            root=project_root,
            label=f"output_files.{name}",
            expected_path=summary_dir / name,
        )

    compression = pd.read_csv(verified_outputs["compression_summary.csv"][0])
    retrieval = pd.read_csv(verified_outputs["retrieval_summary.csv"][0])
    _require_columns(
        compression, REQUIRED_COMPRESSION_COLUMNS, label="compression_summary"
    )
    _require_columns(retrieval, REQUIRED_RETRIEVAL_COLUMNS, label="retrieval_summary")
    for frame, label in (
        (compression, "compression_summary"),
        (retrieval, "retrieval_summary"),
    ):
        _require_single_value(frame, "dataset", dataset_id, label=label)
        _require_single_value(frame, "model_uid", model_uid, label=label)
        _require_single_value(frame, "run_id", run_id, label=label)
    if set(compression["compression_family"].astype(str)) != REQUIRED_FAMILIES:
        raise ValueError("compression summary must contain PCA and PQ")
    if set(retrieval["compression_family"].astype(str)) != REQUIRED_FAMILIES:
        raise ValueError("retrieval summary must contain PCA and PQ")
    if compression.duplicated(list(COMPRESSION_IDENTITY_COLUMNS)).any():
        raise ValueError("compression summary contains duplicate identity rows")
    if retrieval.duplicated(list(RETRIEVAL_IDENTITY_COLUMNS)).any():
        raise ValueError("retrieval summary contains duplicate identity rows")
    if (
        _numeric(compression, "origin_fallback_count", label="compression_summary")
        .ne(0)
        .any()
    ):
        raise ValueError("compression summary is not fallback-free")
    if (
        _numeric(retrieval, "origin_fallback_count", label="retrieval_summary")
        .ne(0)
        .any()
    ):
        raise ValueError("retrieval summary is not fallback-free")
    _validate_mode_coverage(retrieval, label=f"{model_family}/{dataset_id}")
    _validate_retrieval_statistics(retrieval, label=f"{model_family}/{dataset_id}")

    counts = summary_manifest.get("validated_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("summary_manifest.validated_counts is missing")
    if int(counts.get("compression_summary_rows", -1)) != len(compression):
        raise ValueError("compression summary row count differs from manifest")
    if int(counts.get("retrieval_summary_rows", -1)) != len(retrieval):
        raise ValueError("retrieval summary row count differs from manifest")

    extraction_values = set(compression["extraction_uid"].astype(str)) | set(
        retrieval["extraction_uid"].astype(str)
    )
    origin_values = set(compression["origin_embedding_artifact_uid"].astype(str)) | set(
        retrieval["origin_embedding_artifact_uid"].astype(str)
    )
    if len(extraction_values) != 1 or len(origin_values) != 1:
        raise ValueError("compact summaries contain mixed embedding lineage")
    extraction_uid = next(iter(extraction_values))
    origin_uid = next(iter(origin_values))
    prepared_manifest = _read_json(source_paths["prepared_population/manifest.json"])
    for field, expected in (
        ("dataset_id", dataset_id),
        ("model_uid", model_uid),
        ("extraction_uid", extraction_uid),
        ("origin_embedding_artifact_uid", origin_uid),
    ):
        if str(prepared_manifest.get(field)) != expected:
            raise ValueError(f"prepared-population {field} lineage mismatch")

    selection = _RunSelection(
        model_family=model_family,
        dataset_id=dataset_id,
        run_dir=run_dir,
        run_id=run_id,
        model_uid=model_uid,
        checkpoint_sha256=_require_text(
            freeze.get("checkpoint_sha256"), field="freeze.checkpoint_sha256"
        ),
        extraction_uid=extraction_uid,
        origin_embedding_artifact_uid=origin_uid,
        run_manifest_sha256=source_hashes["run_manifest.json"],
        freeze_manifest_sha256=source_hashes["freeze_manifest.json"],
        summary_manifest_sha256=_sha256_file(summary_manifest_path),
        compression_summary_sha256=verified_outputs["compression_summary.csv"][1],
        retrieval_summary_sha256=verified_outputs["retrieval_summary.csv"][1],
    )
    path_provenance = {
        "model_family": model_family,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "model_uid": model_uid,
        "run_dir": str(run_dir),
        "summary_dir": str(summary_dir),
        **selection.content_identity(),
    }
    return compression, retrieval, selection, path_provenance


def _add_interpretation_columns(
    frame: pd.DataFrame, *, model_family: str, dataset_id: str
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "model_family", model_family)
    result["evaluation_protocol"] = "1:N_open_set"
    result["model_comparison_scope"] = "checkpoint_level_generalization"
    result["checkpoint_training_identity_overlap_status"] = "UNKNOWN"
    result["strict_unseen_identity_evidence"] = False
    result["rfw_protocol_variant"] = (
        "RFW-Custom" if dataset_id == "rfw_custom" else "not_applicable"
    )
    return result


def load_cross_model_open_set_matrix(
    model_run_matrix: Mapping[str, Mapping[str, str | Path]],
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> CrossModelOpenSetMatrix:
    """Load an explicit, complete 4-model x 3-open-set completed-run matrix.

    ``model_run_matrix`` must contain exactly the canonical four model families,
    each with explicit run-directory paths for LFW, SurvFace, and RFW-Custom.
    This function performs no latest-run discovery and writes nothing.
    """

    root = Path(project_root).expanduser().resolve()
    normalized = _normalize_matrix(model_run_matrix)
    compression_frames: list[pd.DataFrame] = []
    retrieval_frames: list[pd.DataFrame] = []
    selections: list[_RunSelection] = []
    path_provenance: list[dict[str, Any]] = []
    for model_family in SUPPORTED_MODEL_FAMILIES:
        for dataset_id in SUPPORTED_OPEN_SET_DATASETS:
            compression, retrieval, selection, paths = _load_one_selection(
                normalized[model_family][dataset_id],
                model_family=model_family,
                dataset_id=dataset_id,
                project_root=root,
            )
            compression_frames.append(
                _add_interpretation_columns(
                    compression,
                    model_family=model_family,
                    dataset_id=dataset_id,
                )
            )
            retrieval_frames.append(
                _add_interpretation_columns(
                    retrieval,
                    model_family=model_family,
                    dataset_id=dataset_id,
                )
            )
            selections.append(selection)
            path_provenance.append(paths)

    compression_summary = pd.concat(compression_frames, ignore_index=True)
    retrieval_summary = pd.concat(retrieval_frames, ignore_index=True)
    matrix_compression_key = ("model_family", *COMPRESSION_IDENTITY_COLUMNS)
    matrix_retrieval_key = ("model_family", *RETRIEVAL_IDENTITY_COLUMNS)
    if compression_summary.duplicated(list(matrix_compression_key)).any():
        raise ValueError("cross-model compression matrix contains duplicate rows")
    if retrieval_summary.duplicated(list(matrix_retrieval_key)).any():
        raise ValueError("cross-model retrieval matrix contains duplicate rows")

    join_keys = list(matrix_compression_key)
    compression_payload_columns = [
        column
        for column in compression_summary.columns
        if column not in join_keys
        and column
        not in {
            "evaluation_protocol",
            "model_comparison_scope",
            "checkpoint_training_identity_overlap_status",
            "strict_unseen_identity_evidence",
            "rfw_protocol_variant",
        }
    ]
    joined_summary = retrieval_summary.merge(
        compression_summary[join_keys + compression_payload_columns],
        how="left",
        on=join_keys,
        validate="many_to_one",
        suffixes=("", "_compression"),
    )
    if len(joined_summary) != len(retrieval_summary):
        raise ValueError("joined matrix row count changed during compression join")
    if joined_summary[compression_payload_columns].isna().all(axis=1).any():
        raise ValueError(
            "joined matrix contains retrieval rows without compression data"
        )

    content_selections = [selection.content_identity() for selection in selections]
    matrix_uid = f"open-set-matrix-{_sha256_payload(content_selections)[:20]}"
    selection_manifest: dict[str, Any] = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "artifact_type": MATRIX_ARTIFACT_TYPE,
        "matrix_uid": matrix_uid,
        "selection_mode": "explicit_run_directories_only",
        "auto_selection_used": False,
        "complete_matrix": True,
        "model_families": list(SUPPORTED_MODEL_FAMILIES),
        "open_set_datasets": list(SUPPORTED_OPEN_SET_DATASETS),
        "target_fpirs": list(TARGET_FPIRS),
        "matrix_shape": {
            "model_count": len(SUPPORTED_MODEL_FAMILIES),
            "dataset_count": len(SUPPORTED_OPEN_SET_DATASETS),
            "completed_run_count": len(selections),
        },
        "row_counts": {
            "compression_summary": int(len(compression_summary)),
            "retrieval_summary": int(len(retrieval_summary)),
            "joined_summary": int(len(joined_summary)),
        },
        "content_identity": content_selections,
        "selected_runs": path_provenance,
        "statistical_contract": {
            "rate_interval": "probe-level two-sided 95% Wilson score",
            "difference_interval": (
                "compressed-minus-origin probe-level paired nonparametric "
                "bootstrap percentile"
            ),
            "paired_bootstrap_resamples": 2000,
            "paired_bootstrap_random_seed": 42,
        },
        "interpretation": {
            "model_comparison_scope": "checkpoint_level_generalization",
            "strict_unseen_identity_evidence": False,
            "checkpoint_training_identity_overlap_status": "UNKNOWN",
        },
        "rfw_custom_boundary": {
            "task": "1:N_open_set",
            "official_protocol": False,
            "checkpoint_training_identity_overlap_status": "UNKNOWN",
            "strict_unseen_identity_evidence": False,
            "rfw_official_1to1_included": False,
        },
        "edgeface_rfw_overlap_boundary": {
            "checkpoint_training_identity_overlap_status": "UNKNOWN",
            "strict_unseen_identity_evidence": False,
            "permitted_claim": "checkpoint_level_generalization",
        },
    }
    return CrossModelOpenSetMatrix(
        compression_summary=compression_summary,
        retrieval_summary=retrieval_summary,
        joined_summary=joined_summary,
        selection_manifest=selection_manifest,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.12g",
    )


def _local_file_entry(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def write_cross_model_open_set_matrix(
    matrix: CrossModelOpenSetMatrix,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> CrossModelOpenSetMatrixWriteResult:
    """Atomically publish verified matrix CSVs under ``<root>/<matrix_uid>``."""

    if not isinstance(matrix, CrossModelOpenSetMatrix):
        raise TypeError("matrix must be a CrossModelOpenSetMatrix")
    matrix_uid = _require_text(
        matrix.selection_manifest.get("matrix_uid"),
        field="selection_manifest.matrix_uid",
    )
    expected_uid = (
        "open-set-matrix-"
        + _sha256_payload(matrix.selection_manifest.get("content_identity"))[:20]
    )
    if matrix_uid != expected_uid:
        raise ValueError(
            "selection_manifest.matrix_uid does not match content identity"
        )
    row_counts = matrix.selection_manifest.get("row_counts")
    expected_counts = {
        "compression_summary": len(matrix.compression_summary),
        "retrieval_summary": len(matrix.retrieval_summary),
        "joined_summary": len(matrix.joined_summary),
    }
    if not isinstance(row_counts, Mapping) or any(
        int(row_counts.get(name, -1)) != count
        for name, count in expected_counts.items()
    ):
        raise ValueError("matrix frame row counts differ from selection manifest")

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / matrix_uid
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    temporary = Path(tempfile.mkdtemp(prefix=f".{matrix_uid}.tmp-", dir=root))
    backup: Path | None = None
    try:
        output_frames = {
            "compression_summary.csv": matrix.compression_summary,
            "retrieval_summary.csv": matrix.retrieval_summary,
            "joined_summary.csv": matrix.joined_summary,
        }
        for name, frame in output_frames.items():
            _write_frame(temporary / name, frame)
        selection_path = temporary / "selection_manifest.json"
        _write_json(selection_path, matrix.selection_manifest)
        files = {
            name: _local_file_entry(temporary / name)
            for name in (*output_frames, "selection_manifest.json")
        }
        publish_manifest: dict[str, Any] = {
            "schema_version": MATRIX_SCHEMA_VERSION,
            "artifact_type": MATRIX_ARTIFACT_TYPE,
            "matrix_uid": matrix_uid,
            "immutable_content_identity": matrix.selection_manifest["content_identity"],
            "row_counts": expected_counts,
            "output_files": files,
        }
        manifest_path = temporary / "matrix_manifest.json"
        _write_json(manifest_path, publish_manifest)

        if destination.exists():
            backup = root / f".{matrix_uid}.backup-{os.getpid()}"
            if backup.exists():
                raise FileExistsError(backup)
            os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except BaseException:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
        final_manifest_path = destination / "matrix_manifest.json"
        return CrossModelOpenSetMatrixWriteResult(
            output_dir=destination,
            manifest_path=final_manifest_path,
            manifest=publish_manifest,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
