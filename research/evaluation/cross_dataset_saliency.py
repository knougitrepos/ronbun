from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
from typing import Mapping

import pandas as pd

from research.evaluation.saliency_compression import (
    DEFAULT_PRIMARY_THRESHOLD_EVENT_METRICS,
    DEFAULT_PRIMARY_THRESHOLD_SALIENCY_FEATURES,
    DEFAULT_SALIENCY_FEATURES,
    DEFAULT_THRESHOLD_EVENT_METRICS,
    SALIENCY_THRESHOLD_METRICS_VERSION,
    WEIGHTED_RERANK_STRATEGY,
)
from research.runtime.hashing import sha256_file


_WORKFLOW_SUBDIR = Path("artifacts/step2_workflow")
_GEOMETRY_FILE = "saliency_geometry_associations.csv"
_RETRIEVAL_FILE = "saliency_retrieval_associations.csv"
_THRESHOLD_INSTABILITY_FILE = "saliency_threshold_instability_associations.csv"
_THRESHOLD_POLICY_FILE = "saliency_threshold_policy_comparisons.csv"
_THRESHOLD_POLICY_RHO_FILE = "saliency_threshold_policy_rho_comparisons.csv"
_CANDIDATES_FILE = "representative_case_candidates.csv"
_CASES_FILE = "representative_cases.csv"
_RFW_CALIBRATION_CONTRACT = "rfw_custom_gallery_group_matched_calibration_v2"
_RFW_GALLERY_POLICY = "evaluation_group_matched"
_OPTIONAL_RETRIEVAL_GRAIN_COLUMNS = (
    "protocol_uid",
    "threshold_source_split",
    "evaluation_split",
)
_MATED_ONLY_THRESHOLD_EVENTS = {
    "tpir_at_rank_k_loss",
    "tpir_at_rank_k_gain",
    "tpir_threshold_loss",
    "tpir_rank_loss",
}
_NON_MATED_ONLY_THRESHOLD_EVENTS = {
    "false_accept_gain",
    "false_accept_loss",
}


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


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _phase05_implementation_fingerprint(
    implementation: Mapping[str, object],
    *,
    dataset: str,
) -> tuple[object, ...]:
    version = implementation.get("threshold_metric_derivation_version")
    if version != SALIENCY_THRESHOLD_METRICS_VERSION:
        raise ValueError(f"{dataset}: Phase05 threshold metric version is invalid")
    string_fields = (
        "bootstrap_method",
        "bootstrap_unit",
        "bootstrap_rank_strategy",
    )
    strings: dict[str, str] = {}
    for field in string_fields:
        value = implementation.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{dataset}: Phase05 {field} is missing")
        strings[field] = value
    integer_fields = {
        "bootstrap_repeats": 1,
        "bootstrap_batch_size": 1,
        "bootstrap_seed": 0,
        "paired_minimum_event_count": 1,
    }
    integers: dict[str, int] = {}
    for field, lower_bound in integer_fields.items():
        value = implementation.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < lower_bound
        ):
            raise ValueError(f"{dataset}: Phase05 {field} is invalid")
        integers[field] = value
    sequence_fields = (
        "paired_saliency_features",
        "paired_event_metrics",
    )
    sequences: dict[str, tuple[str, ...]] = {}
    for field in sequence_fields:
        raw_values = implementation.get(field)
        if (
            not isinstance(raw_values, (list, tuple))
            or not raw_values
            or any(not isinstance(value, str) or not value for value in raw_values)
        ):
            raise ValueError(f"{dataset}: Phase05 {field} is missing or invalid")
        values = tuple(raw_values)
        if len(set(values)) != len(values):
            raise ValueError(f"{dataset}: Phase05 {field} contains duplicates")
        allowed = (
            set(DEFAULT_SALIENCY_FEATURES)
            if field == "paired_saliency_features"
            else set(DEFAULT_THRESHOLD_EVENT_METRICS)
        )
        unsupported = set(values) - allowed
        if unsupported:
            raise ValueError(
                f"{dataset}: Phase05 {field} contains unsupported values: "
                f"{sorted(unsupported)}"
            )
        sequences[field] = values
    paired_confidence_level = implementation.get("paired_confidence_level")
    if (
        isinstance(paired_confidence_level, bool)
        or not isinstance(paired_confidence_level, (int, float))
        or not math.isfinite(float(paired_confidence_level))
        or not 0.0 < float(paired_confidence_level) < 1.0
    ):
        raise ValueError(
            f"{dataset}: Phase05 paired_confidence_level is invalid"
        )
    if strings["bootstrap_method"] != "identity_cluster":
        raise ValueError(
            f"{dataset}: Phase05 bootstrap_method must be identity_cluster"
        )
    if strings["bootstrap_unit"] != "identity_id":
        raise ValueError(f"{dataset}: Phase05 bootstrap_unit must be identity_id")
    if strings["bootstrap_rank_strategy"] != WEIGHTED_RERANK_STRATEGY:
        raise ValueError(
            f"{dataset}: Phase05 bootstrap_rank_strategy is invalid"
        )
    commit = implementation.get("source_git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError(f"{dataset}: Phase05 source_git_commit is invalid")
    try:
        int(commit, 16)
    except ValueError as exc:
        raise ValueError(
            f"{dataset}: Phase05 source_git_commit is invalid"
        ) from exc
    raw_source_sha256 = implementation.get("source_sha256")
    if not isinstance(raw_source_sha256, Mapping) or set(raw_source_sha256) != {
        "association",
        "streaming_join",
        "workflow",
    }:
        raise ValueError(f"{dataset}: Phase05 source_sha256 is missing")
    source_sha256: list[tuple[str, str]] = []
    for raw_name, raw_digest in raw_source_sha256.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError(f"{dataset}: Phase05 source_sha256 key is invalid")
        if not _is_sha256(raw_digest):
            raise ValueError(
                f"{dataset}: Phase05 source_sha256[{raw_name!r}] is invalid"
            )
        source_sha256.append((raw_name, str(raw_digest).lower()))
    return (
        str(version),
        strings["bootstrap_method"],
        strings["bootstrap_unit"],
        integers["bootstrap_repeats"],
        strings["bootstrap_rank_strategy"],
        integers["bootstrap_batch_size"],
        integers["bootstrap_seed"],
        sequences["paired_saliency_features"],
        sequences["paired_event_metrics"],
        integers["paired_minimum_event_count"],
        float(paired_confidence_level),
        commit.lower(),
        tuple(sorted(source_sha256)),
    )


def _validate_phase05_output_artifacts(
    phase05_details: Mapping[str, object],
    *,
    run_dir: Path,
    expected_paths: Mapping[str, Path],
    dataset: str,
) -> None:
    raw_artifacts = phase05_details.get("output_artifacts")
    if not isinstance(raw_artifacts, Mapping):
        raise ValueError(f"{dataset}: Phase05 output_artifacts is missing")
    artifact_names = set(raw_artifacts)
    if artifact_names != set(expected_paths):
        raise ValueError(
            f"{dataset}: Phase05 output_artifacts entries mismatch: "
            f"observed={sorted(str(name) for name in artifact_names)}, "
            f"expected={sorted(expected_paths)}"
        )
    for raw_name, raw_entry in raw_artifacts.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError(f"{dataset}: Phase05 output artifact name is invalid")
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                f"{dataset}: Phase05 output artifact {raw_name!r} is invalid"
            )
        raw_path = raw_entry.get("path")
        declared_bytes = raw_entry.get("bytes")
        declared_sha256 = raw_entry.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(
                f"{dataset}: Phase05 output artifact {raw_name!r} path is invalid"
            )
        relative_path = Path(raw_path)
        if relative_path.is_absolute():
            raise ValueError(
                f"{dataset}: Phase05 output artifact {raw_name!r} path "
                "must be relative to the run directory"
            )
        artifact_path = (run_dir / relative_path).resolve()
        try:
            artifact_path.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError(
                f"{dataset}: Phase05 output artifact {raw_name!r} escapes "
                "the run directory"
            ) from exc
        if artifact_path.name != raw_name:
            raise ValueError(
                f"{dataset}: Phase05 output artifact key/path mismatch: "
                f"{raw_name!r} != {artifact_path.name!r}"
            )
        expected_path = expected_paths.get(raw_name)
        if expected_path is not None and artifact_path != expected_path.resolve():
            raise ValueError(
                f"{dataset}: Phase05 output artifact path mismatch for {raw_name}"
            )
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if (
            isinstance(declared_bytes, bool)
            or not isinstance(declared_bytes, int)
            or declared_bytes < 0
        ):
            raise ValueError(
                f"{dataset}: Phase05 output artifact {raw_name!r} bytes is invalid"
            )
        actual_bytes = artifact_path.stat().st_size
        if declared_bytes != actual_bytes:
            raise ValueError(
                f"{dataset}: Phase05 output artifact {raw_name!r} byte mismatch"
            )
        if not _is_sha256(declared_sha256):
            raise ValueError(
                f"{dataset}: Phase05 output artifact {raw_name!r} sha256 is invalid"
            )
        actual_sha256 = sha256_file(artifact_path)
        if str(declared_sha256).lower() != actual_sha256:
            raise ValueError(
                f"{dataset}: Phase05 output artifact {raw_name!r} hash mismatch"
            )


def _validate_target_fpir_coverage_by_grain(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    dataset: str,
    expected_fpirs: set[float],
    grain_columns: tuple[str, ...],
) -> None:
    target_fpirs = pd.to_numeric(frame["target_fpir"], errors="coerce")
    if target_fpirs.isna().any() or not target_fpirs.map(math.isfinite).all():
        raise ValueError(f"{dataset}: {artifact_name} target_fpir is invalid")
    normalized = frame.copy()
    normalized["target_fpir"] = target_fpirs.astype(float)
    duplicate = normalized.duplicated([*grain_columns, "target_fpir"], keep=False)
    if duplicate.any():
        raise ValueError(
            f"{dataset}: {artifact_name} has duplicate target FPIR rows "
            "within a scientific grain"
        )
    grouped = normalized.groupby(list(grain_columns), dropna=False, sort=False)
    for raw_key, group in grouped:
        observed = set(group["target_fpir"].astype(float))
        if observed != expected_fpirs:
            key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            grain = dict(zip(grain_columns, key))
            raise ValueError(
                f"{dataset}: {artifact_name} FPIR coverage mismatch for "
                f"scientific grain {grain}: observed={sorted(observed)}, "
                f"expected={sorted(expected_fpirs)}"
            )


def _applicable_paired_events(
    expected_event_metrics: tuple[str, ...],
    *,
    is_mated: bool,
) -> tuple[str, ...]:
    return tuple(
        event
        for event in expected_event_metrics
        if (
            event not in _NON_MATED_ONLY_THRESHOLD_EVENTS
            if is_mated
            else event not in _MATED_ONLY_THRESHOLD_EVENTS
        )
    )


def _validate_paired_combinations_by_base_grain(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    dataset: str,
    base_grain_columns: tuple[str, ...],
    expected_event_metrics: tuple[str, ...],
    expected_saliency_features: tuple[str, ...] | None = None,
) -> None:
    grouped = frame.groupby(list(base_grain_columns), dropna=False, sort=False)
    for raw_key, group in grouped:
        raw_mated_values = set(group["is_mated"].map(lambda value: str(value).lower()))
        if raw_mated_values == {"true"}:
            is_mated = True
        elif raw_mated_values == {"false"}:
            is_mated = False
        else:
            raise ValueError(f"{artifact_name} is_mated must contain booleans")
        applicable_events = _applicable_paired_events(
            expected_event_metrics,
            is_mated=is_mated,
        )
        if expected_saliency_features is None:
            dimension_columns = ("event_metric",)
            expected = {(event,) for event in applicable_events}
        else:
            dimension_columns = ("saliency_feature", "event_metric")
            expected = set(product(expected_saliency_features, applicable_events))
        observed = {
            tuple(str(value) for value in values)
            for values in group.loc[:, list(dimension_columns)]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
        if observed != expected:
            key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            grain = dict(zip(base_grain_columns, key))
            raise ValueError(
                f"{dataset}: {artifact_name} scientific combinations mismatch "
                f"for base grain {grain}: observed={sorted(observed)}, "
                f"expected={sorted(expected)}"
            )


def _validate_paired_base_grain_coverage(
    reference: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    artifact_name: str,
    dataset: str,
    base_grain_columns: tuple[str, ...],
    expected_event_metrics: tuple[str, ...],
) -> None:
    missing_reference = set(base_grain_columns) - set(reference.columns)
    missing_paired = set(base_grain_columns) - set(paired.columns)
    if missing_reference or missing_paired:
        raise ValueError(
            f"{dataset}: {artifact_name} base grain columns are missing: "
            f"reference={sorted(missing_reference)}, "
            f"paired={sorted(missing_paired)}"
        )

    reference_grains = reference.loc[
        ~reference["search_mode"].astype(str).eq("pq_adc_exhaustive"),
        list(base_grain_columns),
    ].drop_duplicates()
    reference_mated = _strict_boolean_series(
        reference_grains,
        "is_mated",
        artifact_name=_RETRIEVAL_FILE,
    )
    applicable = pd.Series(
        [
            bool(
                _applicable_paired_events(
                    expected_event_metrics,
                    is_mated=bool(is_mated),
                )
            )
            for is_mated in reference_mated
        ],
        index=reference_grains.index,
    )
    reference_grains = reference_grains.loc[applicable]

    def canonical_grains(frame: pd.DataFrame) -> set[tuple[str, ...]]:
        normalized = frame.loc[:, list(base_grain_columns)].drop_duplicates().copy()
        normalized["is_mated"] = _strict_boolean_series(
            normalized,
            "is_mated",
            artifact_name=artifact_name,
        )
        return {
            tuple("<NA>" if pd.isna(value) else str(value) for value in values)
            for values in normalized.itertuples(index=False, name=None)
        }

    expected = canonical_grains(reference_grains)
    observed = canonical_grains(paired)
    if observed != expected:
        raise ValueError(
            f"{dataset}: {artifact_name} paired base grain coverage mismatch: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _nonnegative_integer_series(
    frame: pd.DataFrame,
    column: str,
    *,
    artifact_name: str,
) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    invalid = (
        numeric.isna()
        | ~numeric.map(math.isfinite)
        | numeric.lt(0)
        | numeric.mod(1).ne(0)
    )
    if invalid.any():
        raise ValueError(f"{artifact_name} {column} must be non-negative integers")
    return numeric.astype("int64")


def _strict_boolean_series(
    frame: pd.DataFrame,
    column: str,
    *,
    artifact_name: str,
) -> pd.Series:
    normalized = frame[column].map(
        lambda value: str(value).strip().lower()
        if not pd.isna(value)
        else ""
    )
    invalid = ~normalized.isin({"true", "false"})
    if invalid.any():
        raise ValueError(f"{artifact_name} {column} must contain booleans")
    return normalized.eq("true")


def _required_finite_numeric_series(
    frame: pd.DataFrame,
    column: str,
    *,
    artifact_name: str,
) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any() or not numeric.map(math.isfinite).all():
        raise ValueError(f"{artifact_name} {column} must contain finite numbers")
    return numeric.astype(float)


def _validate_analysis_tier(
    frame: pd.DataFrame,
    *,
    primary: pd.Series,
    primary_label: str,
    artifact_name: str,
) -> None:
    expected = primary.map(
        {True: primary_label, False: "exploratory"}
    )
    if not frame["analysis_tier"].astype(str).equals(expected):
        raise ValueError(
            f"{artifact_name} analysis_tier is inconsistent with metric contract"
        )


def _validate_paired_policy_comparison(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    expected_bootstrap_unit: str,
    expected_bootstrap_repeats: int,
    expected_confidence_level: float,
) -> None:
    primary = frame["event_metric"].astype(str).isin(
        DEFAULT_PRIMARY_THRESHOLD_EVENT_METRICS
    )
    _validate_analysis_tier(
        frame,
        primary=primary,
        primary_label="prespecified_supporting",
        artifact_name=artifact_name,
    )
    if set(frame["bootstrap_unit"].astype(str)) != {expected_bootstrap_unit}:
        raise ValueError(
            f"{artifact_name} bootstrap_unit does not match Phase05 implementation"
        )
    count_columns = (
        "paired_query_count",
        "identity_count",
        "frozen_event_count",
        "recalibrated_event_count",
        "resolved_event_count",
        "introduced_event_count",
        "paired_bootstrap_valid_repeats",
    )
    counts = {
        column: _nonnegative_integer_series(
            frame,
            column,
            artifact_name=artifact_name,
        )
        for column in count_columns
    }
    paired = counts["paired_query_count"]
    frozen = counts["frozen_event_count"]
    recalibrated = counts["recalibrated_event_count"]
    resolved = counts["resolved_event_count"]
    introduced = counts["introduced_event_count"]
    valid_repeats = counts["paired_bootstrap_valid_repeats"]
    if paired.le(0).any():
        raise ValueError(f"{artifact_name} paired_query_count must be positive")
    if (
        counts["identity_count"].le(0).any()
        or counts["identity_count"].gt(paired).any()
    ):
        raise ValueError(
            f"{artifact_name} identity_count must be within paired queries"
        )
    if frozen.gt(paired).any() or recalibrated.gt(paired).any():
        raise ValueError(f"{artifact_name} event counts exceed paired queries")
    if resolved.gt(frozen).any() or introduced.gt(paired - frozen).any():
        raise ValueError(f"{artifact_name} transition counts are impossible")
    if not recalibrated.equals(frozen - resolved + introduced):
        raise ValueError(
            f"{artifact_name} event counts disagree with resolved/introduced counts"
        )
    if valid_repeats.ne(expected_bootstrap_repeats).any():
        raise ValueError(
            f"{artifact_name} valid bootstrap repeats do not match Phase05 contract"
        )

    frozen_rate = _required_finite_numeric_series(
        frame,
        "frozen_event_rate",
        artifact_name=artifact_name,
    )
    recalibrated_rate = _required_finite_numeric_series(
        frame,
        "recalibrated_event_rate",
        artifact_name=artifact_name,
    )
    rate_difference = _required_finite_numeric_series(
        frame,
        "recalibrated_minus_frozen_rate",
        artifact_name=artifact_name,
    )
    if (
        ((frozen_rate - frozen / paired).abs() > 1e-12).any()
        or ((recalibrated_rate - recalibrated / paired).abs() > 1e-12).any()
        or ((rate_difference - (recalibrated_rate - frozen_rate)).abs() > 1e-12).any()
    ):
        raise ValueError(f"{artifact_name} event rates are inconsistent with counts")
    if (
        frozen_rate.lt(0).any()
        or frozen_rate.gt(1).any()
        or recalibrated_rate.lt(0).any()
        or recalibrated_rate.gt(1).any()
        or rate_difference.abs().gt(1).any()
    ):
        raise ValueError(f"{artifact_name} event rates are out of range")

    confidence = _required_finite_numeric_series(
        frame,
        "paired_bootstrap_confidence_level",
        artifact_name=artifact_name,
    )
    if ((confidence - expected_confidence_level).abs() > 1e-12).any():
        raise ValueError(
            f"{artifact_name} confidence level does not match Phase05 implementation"
        )
    ci_low = _required_finite_numeric_series(
        frame,
        "paired_bootstrap_ci_low",
        artifact_name=artifact_name,
    )
    ci_high = _required_finite_numeric_series(
        frame,
        "paired_bootstrap_ci_high",
        artifact_name=artifact_name,
    )
    if (
        ci_low.abs().gt(1).any()
        or ci_high.abs().gt(1).any()
        or (ci_low > ci_high).any()
    ):
        raise ValueError(f"{artifact_name} bootstrap CI is out of range")


def _validate_paired_rho_support(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    expected_minimum_event_count: int,
    expected_bootstrap_unit: str,
    maximum_bootstrap_repeats: int,
    expected_confidence_level: float,
) -> None:
    primary = (
        frame["saliency_feature"].astype(str).isin(
            DEFAULT_PRIMARY_THRESHOLD_SALIENCY_FEATURES
        )
        & frame["event_metric"].astype(str).isin(
            DEFAULT_PRIMARY_THRESHOLD_EVENT_METRICS
        )
    )
    _validate_analysis_tier(
        frame,
        primary=primary,
        primary_label="prespecified_primary",
        artifact_name=artifact_name,
    )
    bootstrap_units = set(frame["bootstrap_unit"].astype(str))
    if bootstrap_units != {expected_bootstrap_unit}:
        raise ValueError(
            f"{artifact_name} bootstrap_unit does not match Phase05 implementation"
        )
    paired = _nonnegative_integer_series(
        frame,
        "paired_query_count",
        artifact_name=artifact_name,
    )
    identities = _nonnegative_integer_series(
        frame,
        "identity_count",
        artifact_name=artifact_name,
    )
    frozen = _nonnegative_integer_series(
        frame,
        "frozen_event_count",
        artifact_name=artifact_name,
    )
    recalibrated = _nonnegative_integer_series(
        frame,
        "recalibrated_event_count",
        artifact_name=artifact_name,
    )
    minimum = _nonnegative_integer_series(
        frame,
        "minimum_event_count",
        artifact_name=artifact_name,
    )
    valid_repeats = _nonnegative_integer_series(
        frame,
        "paired_bootstrap_valid_repeats",
        artifact_name=artifact_name,
    )
    if paired.le(0).any():
        raise ValueError(f"{artifact_name} paired_query_count must be positive")
    if identities.le(0).any() or identities.gt(paired).any():
        raise ValueError(
            f"{artifact_name} identity_count must be within paired queries"
        )
    if minimum.lt(1).any():
        raise ValueError(f"{artifact_name} minimum_event_count must be positive")
    if set(minimum) != {expected_minimum_event_count}:
        raise ValueError(
            f"{artifact_name} minimum_event_count does not match Phase05 "
            "implementation"
        )
    if valid_repeats.gt(maximum_bootstrap_repeats).any():
        raise ValueError(
            f"{artifact_name} valid bootstrap repeats exceed Phase05 contract"
        )
    confidence = _required_finite_numeric_series(
        frame,
        "paired_bootstrap_confidence_level",
        artifact_name=artifact_name,
    )
    if ((confidence - expected_confidence_level).abs() > 1e-12).any():
        raise ValueError(
            f"{artifact_name} confidence level does not match Phase05 implementation"
        )
    if frozen.gt(paired).any() or recalibrated.gt(paired).any():
        raise ValueError(f"{artifact_name} event counts exceed paired_query_count")
    eligible = _strict_boolean_series(
        frame,
        "event_support_eligible",
        artifact_name=artifact_name,
    )
    expected_eligible = pd.concat(
        (
            frozen,
            paired - frozen,
            recalibrated,
            paired - recalibrated,
        ),
        axis=1,
    ).min(axis=1).ge(minimum)
    if not eligible.equals(expected_eligible):
        raise ValueError(
            f"{artifact_name} event_support_eligible is inconsistent with "
            "paired event counts"
        )

    statistic_columns = (
        "frozen_spearman_rho",
        "recalibrated_spearman_rho",
        "recalibrated_minus_frozen_rho",
        "paired_bootstrap_ci_low",
        "paired_bootstrap_ci_high",
    )
    numeric_statistics: dict[str, pd.Series] = {}
    for column in statistic_columns:
        raw_values = frame[column]
        numeric = pd.to_numeric(raw_values, errors="coerce")
        invalid = raw_values.notna() & (
            numeric.isna() | ~numeric.map(math.isfinite)
        )
        if invalid.any():
            raise ValueError(f"{artifact_name} {column} contains invalid values")
        numeric_statistics[column] = numeric
    ineligible = ~eligible
    if any(values.loc[ineligible].notna().any() for values in numeric_statistics.values()):
        raise ValueError(
            f"{artifact_name} unsupported rows must not contain rho or CI values"
        )
    if valid_repeats.loc[ineligible].ne(0).any():
        raise ValueError(
            f"{artifact_name} unsupported rows must have zero valid bootstraps"
        )
    ci_low_present = numeric_statistics["paired_bootstrap_ci_low"].notna()
    ci_high_present = numeric_statistics["paired_bootstrap_ci_high"].notna()
    if (ci_low_present != ci_high_present).any():
        raise ValueError(f"{artifact_name} bootstrap CI must contain both bounds")
    ci_present = ci_low_present & ci_high_present
    if (valid_repeats.gt(0) != ci_present).any():
        raise ValueError(
            f"{artifact_name} bootstrap CI is inconsistent with valid repeats"
        )
    frozen_rho = numeric_statistics["frozen_spearman_rho"]
    recalibrated_rho = numeric_statistics["recalibrated_spearman_rho"]
    rho_difference = numeric_statistics["recalibrated_minus_frozen_rho"]
    ci_low = numeric_statistics["paired_bootstrap_ci_low"]
    ci_high = numeric_statistics["paired_bootstrap_ci_high"]
    if (frozen_rho.notna() != recalibrated_rho.notna()).any():
        raise ValueError(f"{artifact_name} paired rho values must both be present")
    if (
        frozen_rho.dropna().abs().gt(1.0).any()
        or recalibrated_rho.dropna().abs().gt(1.0).any()
        or rho_difference.dropna().abs().gt(2.0).any()
        or ci_low.dropna().abs().gt(2.0).any()
        or ci_high.dropna().abs().gt(2.0).any()
        or (ci_low.loc[ci_present] > ci_high.loc[ci_present]).any()
    ):
        raise ValueError(f"{artifact_name} rho or CI values are out of range")
    both_rho = frozen_rho.notna() & recalibrated_rho.notna()
    if ((~both_rho) & (valid_repeats.ne(0) | ci_present)).any():
        raise ValueError(
            f"{artifact_name} bootstrap evidence requires paired point rho values"
        )
    if (both_rho & valid_repeats.eq(0)).any():
        raise ValueError(
            f"{artifact_name} finite paired rho rows require bootstrap evidence"
        )
    expected_difference = recalibrated_rho - frozen_rho
    inconsistent_difference = both_rho & (
        (rho_difference - expected_difference).abs() > 1e-12
    )
    if inconsistent_difference.any() or rho_difference.loc[~both_rho].notna().any():
        raise ValueError(
            f"{artifact_name} recalibrated_minus_frozen_rho is inconsistent"
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
    if not expected_fpirs or len(expected_fpirs) != len(expected_target_fpirs):
        raise ValueError("expected_target_fpirs must contain unique values")
    if not all(math.isfinite(value) and 0.0 < value < 1.0 for value in expected_fpirs):
        raise ValueError("expected_target_fpirs must be finite values between 0 and 1")
    expected_implementation_fingerprint: tuple[object, ...] | None = None

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
        implementation_fingerprint = _phase05_implementation_fingerprint(
            implementation,
            dataset=dataset,
        )
        if expected_implementation_fingerprint is None:
            expected_implementation_fingerprint = implementation_fingerprint
        elif implementation_fingerprint != expected_implementation_fingerprint:
            raise ValueError(
                f"{dataset}: mixed Phase05 implementation fingerprint across "
                "selected runs"
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
        candidates_path = workflow / _CANDIDATES_FILE
        cases_path = workflow / _CASES_FILE
        for path in (
            geometry_path,
            retrieval_path,
            threshold_instability_path,
            threshold_policy_path,
            threshold_policy_rho_path,
            candidates_path,
            cases_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        if not isinstance(phase05_details, Mapping):
            raise ValueError(f"{dataset}: Phase05 details are missing")
        _validate_phase05_output_artifacts(
            phase05_details,
            run_dir=run_dir,
            expected_paths={
                _GEOMETRY_FILE: geometry_path,
                _RETRIEVAL_FILE: retrieval_path,
                _THRESHOLD_INSTABILITY_FILE: threshold_instability_path,
                _THRESHOLD_POLICY_FILE: threshold_policy_path,
                _THRESHOLD_POLICY_RHO_FILE: threshold_policy_rho_path,
                _CANDIDATES_FILE: candidates_path,
            },
            dataset=dataset,
        )

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
                "analysis_tier",
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
        _validate_analysis_tier(
            retrieval,
            primary=(
                retrieval["saliency_feature"].astype(str).isin(
                    DEFAULT_PRIMARY_THRESHOLD_SALIENCY_FEATURES
                )
                & retrieval["sensitivity_metric"].astype(str).isin(
                    (
                        "absolute_threshold_margin_shift",
                        *DEFAULT_PRIMARY_THRESHOLD_EVENT_METRICS,
                    )
                )
            ),
            primary_label="prespecified_primary",
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
                "analysis_tier",
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
        _validate_analysis_tier(
            threshold_instability,
            primary=(
                threshold_instability["instability_predictor"].astype(str).isin(
                    (
                        "absolute_top1_score_drift",
                        "absolute_threshold_margin_shift",
                    )
                )
                & threshold_instability["sensitivity_metric"].astype(str).isin(
                    DEFAULT_PRIMARY_THRESHOLD_EVENT_METRICS
                )
            ),
            primary_label="prespecified_supporting",
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
                "analysis_tier",
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
                "paired_bootstrap_confidence_level",
                "paired_bootstrap_ci_low",
                "paired_bootstrap_ci_high",
                "paired_bootstrap_valid_repeats",
                "bootstrap_unit",
            },
            artifact_name=_THRESHOLD_POLICY_FILE,
        )
        _validate_threshold_metric_version(
            threshold_policy,
            artifact_name=_THRESHOLD_POLICY_FILE,
        )
        _validate_paired_policy_comparison(
            threshold_policy,
            artifact_name=_THRESHOLD_POLICY_FILE,
            expected_bootstrap_unit=str(implementation["bootstrap_unit"]),
            expected_bootstrap_repeats=int(implementation["bootstrap_repeats"]),
            expected_confidence_level=float(
                implementation["paired_confidence_level"]
            ),
        )
        _validate_frame(
            threshold_policy_rho,
            dataset=dataset,
            model_uid=model_uid,
            analysis_scope="threshold_policy_saliency_rho",
            required_columns={
                "analysis_scope",
                "threshold_metric_derivation_version",
                "analysis_tier",
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
                "event_support_eligible",
                "minimum_event_count",
                "paired_bootstrap_confidence_level",
                "paired_bootstrap_ci_low",
                "paired_bootstrap_ci_high",
                "paired_bootstrap_valid_repeats",
                "bootstrap_unit",
            },
            artifact_name=_THRESHOLD_POLICY_RHO_FILE,
        )
        _validate_threshold_metric_version(
            threshold_policy_rho,
            artifact_name=_THRESHOLD_POLICY_RHO_FILE,
        )
        _validate_paired_rho_support(
            threshold_policy_rho,
            artifact_name=_THRESHOLD_POLICY_RHO_FILE,
            expected_minimum_event_count=int(
                implementation["paired_minimum_event_count"]
            ),
            expected_bootstrap_unit=str(implementation["bootstrap_unit"]),
            maximum_bootstrap_repeats=int(implementation["bootstrap_repeats"]),
            expected_confidence_level=float(
                implementation["paired_confidence_level"]
            ),
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
        optional_retrieval_grain = tuple(
            column
            for column in _OPTIONAL_RETRIEVAL_GRAIN_COLUMNS
            if column in retrieval
        )
        for artifact_name, frame in (
            (_THRESHOLD_INSTABILITY_FILE, threshold_instability),
            (_THRESHOLD_POLICY_FILE, threshold_policy),
            (_THRESHOLD_POLICY_RHO_FILE, threshold_policy_rho),
        ):
            observed_optional = tuple(
                column
                for column in _OPTIONAL_RETRIEVAL_GRAIN_COLUMNS
                if column in frame
            )
            if observed_optional != optional_retrieval_grain:
                raise ValueError(
                    f"{dataset}: {artifact_name} optional scientific grain "
                    "columns do not match retrieval"
                )
        _validate_target_fpir_coverage_by_grain(
            retrieval,
            artifact_name=_RETRIEVAL_FILE,
            dataset=dataset,
            expected_fpirs=expected_fpirs,
            grain_columns=(
                "compression_family",
                "compression_profile",
                "search_mode",
                *optional_retrieval_grain,
                "threshold_policy",
                "is_mated",
                "saliency_feature",
                "sensitivity_metric",
            ),
        )
        _validate_target_fpir_coverage_by_grain(
            threshold_instability,
            artifact_name=_THRESHOLD_INSTABILITY_FILE,
            dataset=dataset,
            expected_fpirs=expected_fpirs,
            grain_columns=(
                "compression_family",
                "compression_profile",
                "search_mode",
                *optional_retrieval_grain,
                "threshold_policy",
                "is_mated",
                "instability_predictor",
                "sensitivity_metric",
            ),
        )
        policy_base_grain = (
            "compression_family",
            "compression_profile",
            "search_mode",
            *optional_retrieval_grain,
            "is_mated",
        )
        paired_event_metrics = tuple(
            str(value) for value in implementation["paired_event_metrics"]
        )
        _validate_paired_base_grain_coverage(
            retrieval,
            threshold_policy,
            artifact_name=_THRESHOLD_POLICY_FILE,
            dataset=dataset,
            base_grain_columns=policy_base_grain,
            expected_event_metrics=paired_event_metrics,
        )
        _validate_paired_combinations_by_base_grain(
            threshold_policy,
            artifact_name=_THRESHOLD_POLICY_FILE,
            dataset=dataset,
            base_grain_columns=policy_base_grain,
            expected_event_metrics=paired_event_metrics,
        )
        _validate_target_fpir_coverage_by_grain(
            threshold_policy,
            artifact_name=_THRESHOLD_POLICY_FILE,
            dataset=dataset,
            expected_fpirs=expected_fpirs,
            grain_columns=(
                "compression_family",
                "compression_profile",
                "search_mode",
                *optional_retrieval_grain,
                "is_mated",
                "event_metric",
            ),
        )
        policy_rho_base_grain = (
            "compression_family",
            "compression_profile",
            "search_mode",
            *optional_retrieval_grain,
            "is_mated",
        )
        paired_saliency_features = tuple(
            str(value) for value in implementation["paired_saliency_features"]
        )
        _validate_paired_base_grain_coverage(
            retrieval,
            threshold_policy_rho,
            artifact_name=_THRESHOLD_POLICY_RHO_FILE,
            dataset=dataset,
            base_grain_columns=policy_rho_base_grain,
            expected_event_metrics=paired_event_metrics,
        )
        _validate_paired_combinations_by_base_grain(
            threshold_policy_rho,
            artifact_name=_THRESHOLD_POLICY_RHO_FILE,
            dataset=dataset,
            base_grain_columns=policy_rho_base_grain,
            expected_event_metrics=paired_event_metrics,
            expected_saliency_features=paired_saliency_features,
        )
        _validate_target_fpir_coverage_by_grain(
            threshold_policy_rho,
            artifact_name=_THRESHOLD_POLICY_RHO_FILE,
            dataset=dataset,
            expected_fpirs=expected_fpirs,
            grain_columns=(
                "compression_family",
                "compression_profile",
                "search_mode",
                *optional_retrieval_grain,
                "is_mated",
                "saliency_feature",
                "event_metric",
            ),
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
            _CANDIDATES_FILE: _file_entry(candidates_path, run_dir=run_dir),
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
