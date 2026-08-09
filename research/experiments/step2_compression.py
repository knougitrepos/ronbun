"""PyTorch Step 2 compression characterization for one frozen population."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from research.calibration.rejection import (
    choose_non_mated_fpir_threshold,
    choose_threshold,
)
from research.compression import (
    ORIGIN_512,
    PQCompressor,
    fit_pca_family,
    pq_profile_name,
)
from research.evaluation import (
    PAIRED_BOOTSTRAP_RANDOM_SEED,
    PAIRED_BOOTSTRAP_RESAMPLES,
    apply_retrieval_thresholds,
    compare_cosine_retrieval,
    compare_pq_adc_retrieval,
    origin_cosine_retrieval,
    paired_binary_rate_difference_bootstrap_interval,
    paired_embedding_metrics,
    wilson_score_interval,
)
from research.explainability.gradcam.extraction import PreparedPopulationInputs
from research.datasets.rfw_custom import (
    adapt_rfw_custom_manifest_to_open_set_protocol,
)
from research.protocols import (
    OpenSetProtocol,
    build_calibration_protocol,
    build_open_set_protocol,
    build_survface_official_protocol,
    build_survface_matched_calibration_protocol,
    rebase_survface_protocol_subset_indexes,
    validate_identity_disjoint_splits,
)
from research.templates.aggregation import aggregate_templates


ProgressCallback = Callable[[str, dict[str, object]], None]


def _emit(progress: ProgressCallback | None, message: str, **details: object) -> None:
    if progress is not None:
        progress(message, details)


@dataclass(frozen=True)
class Step2CompressionResult:
    paired_metrics: pd.DataFrame
    retrieval_metrics: pd.DataFrame
    origin_score_audit: pd.DataFrame
    calibration_diagnostics: dict[str, object]
    summary: pd.DataFrame
    fitted_codecs: tuple[tuple[str, str, Any], ...]
    calibration_diagnostics_by_target: dict[str, dict[str, object]]
    demographic_summary: pd.DataFrame


@dataclass(frozen=True)
class SurvFaceOriginCalibrationSweepResult:
    score_audit: pd.DataFrame
    condition_summary: pd.DataFrame
    diagnostics: dict[str, object]


def _stable_protocol_key(value: str, *, seed: int, namespace: str) -> str:
    payload = f"{namespace}:{seed}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_matched_survface_test_protocol(
    official_protocol: OpenSetProtocol,
    *,
    gallery_identity_count: int,
    enrollment_count: int,
    seed: int,
) -> OpenSetProtocol:
    gallery = official_protocol.gallery.copy()
    sizes = gallery.groupby("identity_id")["image_id"].nunique()
    eligible = [
        str(identity_id)
        for identity_id, count in sizes.items()
        if int(count) >= enrollment_count
    ]
    eligible.sort(
        key=lambda value: _stable_protocol_key(
            value,
            seed=seed,
            namespace="test:gallery-identity",
        )
    )
    if len(eligible) < gallery_identity_count:
        raise ValueError(
            "official test gallery has too few identities for matched condition: "
            f"gallery={gallery_identity_count}, enrollment={enrollment_count}, "
            f"eligible={len(eligible)}"
        )
    selected_identities = set(eligible[:gallery_identity_count])
    selected_gallery = gallery.loc[
        gallery["identity_id"].astype(str).isin(selected_identities)
    ].copy()
    enrollment_parts: list[pd.DataFrame] = []
    for identity_id, group in selected_gallery.groupby("identity_id", sort=True):
        ordered = group.assign(
            _stable_order=group["image_id"].astype(str).map(
                lambda value: _stable_protocol_key(
                    value,
                    seed=seed,
                    namespace=f"test:{identity_id}:enrollment",
                )
            )
        ).sort_values("_stable_order")
        enrollment_parts.append(
            ordered.head(enrollment_count).drop(columns="_stable_order")
        )
    matched_gallery = pd.concat(enrollment_parts, ignore_index=True)
    registered = official_protocol.registered_probes.loc[
        official_protocol.registered_probes["identity_id"]
        .astype(str)
        .isin(selected_identities)
    ].copy()
    empty_known = official_protocol.known_unknown_probes.iloc[0:0].copy()
    return OpenSetProtocol(
        gallery=matched_gallery.sort_values(
            ["identity_id", "image_id"], kind="stable"
        ).reset_index(drop=True),
        registered_probes=registered.reset_index(drop=True),
        known_unknown_probes=empty_known.reset_index(drop=True),
        unknown_unknown_probes=official_protocol.unknown_unknown_probes.copy().reset_index(
            drop=True
        ),
    )


def _identity_tuple(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values if str(value).strip())
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _population_frame(
    prepared: PreparedPopulationInputs,
    selected_manifest: pd.DataFrame,
) -> pd.DataFrame:
    required = {"sample_id", "identity_id", "split"}
    missing = sorted(required.difference(selected_manifest.columns))
    if missing:
        raise ValueError(f"selected_manifest is missing columns: {missing}")
    selected = selected_manifest.copy()
    selected["sample_id"] = selected["sample_id"].astype(str)
    selected["identity_id"] = selected["identity_id"].astype(str)
    if selected["sample_id"].duplicated().any():
        raise ValueError("selected_manifest sample_id values must be unique")
    if not np.array_equal(
        selected["sample_id"].to_numpy(),
        prepared.sample_ids.astype(str),
    ):
        raise ValueError(
            "selected_manifest row order must match the prepared population"
        )
    embeddings = np.asarray(prepared.normalized_embeddings, dtype=np.float32)
    if embeddings.shape != (len(selected), 512):
        raise ValueError(
            f"Step 2 compression requires [N, 512] embeddings, got {embeddings.shape}"
        )
    raw_norms = np.asarray(prepared.raw_norms, dtype=np.float32)
    if raw_norms.shape != (len(selected),):
        raise ValueError(
            "Step 2 compression requires one raw embedding norm per sample"
        )
    if not np.all(np.isfinite(raw_norms)) or np.any(raw_norms <= 0.0):
        raise ValueError("raw embedding norms must be positive and finite")
    selected = selected.rename(columns={"sample_id": "image_id"})
    selected["origin_embedding"] = list(embeddings)
    selected["raw_embedding_norm"] = raw_norms
    protocol_columns = ["image_id", "identity_id", "split"]
    if "image_path" in selected:
        protocol_columns.append("image_path")
    else:
        selected["image_path"] = ""
        protocol_columns.append("image_path")
    validate_identity_disjoint_splits(selected.loc[:, protocol_columns])
    return selected


def _protocol_arrays(
    protocol: OpenSetProtocol,
    population: pd.DataFrame,
) -> dict[str, np.ndarray]:
    lookup = population[
        ["image_id", "origin_embedding", "raw_embedding_norm"]
    ]

    def attach(frame: pd.DataFrame) -> pd.DataFrame:
        if "origin_embedding" in frame:
            merged = frame.copy()
        else:
            merged = frame.merge(
                lookup,
                on="image_id",
                how="left",
                validate="one_to_one",
            )
        if merged["origin_embedding"].isna().any():
            raise ValueError("protocol row is missing an origin embedding")
        return merged

    gallery = attach(protocol.gallery)
    templates = aggregate_templates(
        gallery,
        "mean",
        identity_col="identity_id",
        image_col="image_id",
        embedding_col="origin_embedding",
    )
    query_parts = []
    for probe_type, frame in (
        ("registered", protocol.registered_probes),
        ("known_unknown", protocol.known_unknown_probes),
        ("unknown_unknown", protocol.unknown_unknown_probes),
    ):
        if frame.empty:
            continue
        part = attach(frame)
        part["probe_type"] = probe_type
        query_parts.append(part)
    if not query_parts:
        raise ValueError("protocol contains no probe rows")
    queries = pd.concat(query_parts, ignore_index=True)
    return {
        "query_ids": queries["image_id"].astype(str).to_numpy(),
        "query_identity_ids": queries["identity_id"].astype(str).to_numpy(),
        "query_probe_types": queries["probe_type"].astype(str).to_numpy(),
        "query_raw_norms": queries["raw_embedding_norm"].to_numpy(
            dtype=np.float32
        ),
        "queries": np.stack(queries["origin_embedding"]).astype(np.float32),
        "gallery_ids": templates["identity_id"].astype(str).to_numpy(),
        "gallery_identity_ids": templates["identity_id"].astype(str).to_numpy(),
        "gallery": np.stack(templates["embedding"]).astype(np.float32),
    }


def _distribution_summary(values: np.ndarray, *, name: str) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    quantiles = np.quantile(array, [0.50, 0.90, 0.95, 0.99])
    return {
        "count": int(len(array)),
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "maximum": float(np.max(array)),
    }


def _wilson_interval_95(successes: int, total: int) -> tuple[float, float]:
    """Compatibility wrapper for calibration-validation callers."""

    return wilson_score_interval(successes, total, confidence_level=0.95)


def _template_count_summary(protocol: OpenSetProtocol) -> dict[str, object]:
    counts = (
        protocol.gallery.assign(
            identity_id=protocol.gallery["identity_id"].astype(str),
            image_id=protocol.gallery["image_id"].astype(str),
        )
        .groupby("identity_id", sort=True)["image_id"]
        .nunique()
        .to_numpy(dtype=np.int64)
    )
    if len(counts) == 0 or np.any(counts <= 0):
        raise ValueError("gallery must contain at least one image per template")
    return {
        "template_count": int(len(counts)),
        "source_image_count": int(np.sum(counts)),
        "images_per_template": _distribution_summary(
            counts.astype(np.float64),
            name="gallery images per template",
        ),
    }


def _assert_same_origin_comparison(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    split_name: str,
) -> None:
    exact_columns = (
        "query_id",
        "query_identity_id",
        "is_mated",
        "origin_top1_gallery_id",
        "origin_top1_identity_id",
        "origin_rank1_correct",
        "origin_top_k_correct",
    )
    if len(reference) != len(candidate):
        raise RuntimeError(f"{split_name} origin retrieval row count drifted")
    for column in exact_columns:
        if not np.array_equal(
            reference[column].to_numpy(),
            candidate[column].to_numpy(),
        ):
            raise RuntimeError(
                f"{split_name} origin retrieval column drifted: {column}"
            )
    if not np.allclose(
        reference["origin_top1_score"].to_numpy(dtype=np.float64),
        candidate["origin_top1_score"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError(f"{split_name} origin top-1 scores drifted")


def _origin_score_rows(
    comparison: pd.DataFrame,
    protocol_arrays: dict[str, np.ndarray],
    *,
    dataset_id: str,
    model_uid: str,
    protocol_uid: str,
    evaluation_split: str,
    decision_threshold: float,
    target_fpir: float,
) -> pd.DataFrame:
    query_ids = protocol_arrays["query_ids"].astype(str)
    if not np.array_equal(
        comparison["query_id"].astype(str).to_numpy(),
        query_ids,
    ):
        raise RuntimeError(
            f"{evaluation_split} origin audit query order differs from protocol"
        )
    scores = comparison["origin_top1_score"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(scores)) or np.any(scores < -1.000001) or np.any(
        scores > 1.000001
    ):
        raise ValueError("origin cosine scores must be finite and inside [-1, 1]")
    is_mated = comparison["is_mated"].to_numpy(dtype=bool)
    rank1_correct = comparison["origin_rank1_correct"].to_numpy(dtype=bool)
    accepted = scores >= float(decision_threshold)
    result = pd.DataFrame(
        {
            "dataset": str(dataset_id),
            "model_uid": str(model_uid),
            "protocol_uid": str(protocol_uid),
            "evaluation_split": str(evaluation_split),
            "threshold_source_split": "calibration",
            "query_id": query_ids,
            "query_identity_id": protocol_arrays["query_identity_ids"].astype(
                str
            ),
            "probe_type": protocol_arrays["query_probe_types"].astype(str),
            "is_mated": is_mated,
            "origin_top1_gallery_id": comparison[
                "origin_top1_gallery_id"
            ].astype(str),
            "origin_top1_identity_id": comparison[
                "origin_top1_identity_id"
            ].astype(str),
            "origin_top1_score": scores,
            "origin_rank1_correct": rank1_correct,
            "origin_top_k_correct": comparison[
                "origin_top_k_correct"
            ].to_numpy(dtype=bool),
            "query_raw_embedding_norm": protocol_arrays[
                "query_raw_norms"
            ].astype(np.float32),
            "target_fpir": float(target_fpir),
            "origin_decision_threshold": float(decision_threshold),
            "threshold_comparator": ">=",
            "origin_accepted": accepted,
            "origin_false_accept": accepted & ~is_mated,
            "origin_true_identification": accepted & is_mated & rank1_correct,
        }
    )
    if result["query_id"].duplicated().any():
        raise RuntimeError(f"{evaluation_split} origin audit query IDs repeat")
    return result


def _split_diagnostics(
    rows: pd.DataFrame,
    protocol: OpenSetProtocol,
    *,
    target_fpir: float,
    require_target_bound: bool,
) -> dict[str, object]:
    is_mated = rows["is_mated"].to_numpy(dtype=bool)
    accepted = rows["origin_accepted"].to_numpy(dtype=bool)
    correct = rows["origin_rank1_correct"].to_numpy(dtype=bool)
    non_mated_count = int((~is_mated).sum())
    if non_mated_count <= 0:
        raise ValueError("origin calibration audit requires non-mated probes")
    false_accept_count = int((accepted & ~is_mated).sum())
    realized_fpir = float(false_accept_count / non_mated_count)
    fpir_ci_low, fpir_ci_high = _wilson_interval_95(
        false_accept_count,
        non_mated_count,
    )
    if require_target_bound and realized_fpir > float(target_fpir) + 1e-12:
        raise RuntimeError(
            "calibration FPIR exceeds its target under the persisted comparator"
        )
    mated_count = int(is_mated.sum())
    true_identification_count = int((accepted & is_mated & correct).sum())
    probe_type_counts = {
        str(key): int(value)
        for key, value in rows["probe_type"].value_counts(sort=False).items()
    }
    return {
        **_template_count_summary(protocol),
        "query_count": int(len(rows)),
        "mated_count": mated_count,
        "non_mated_count": non_mated_count,
        "probe_type_counts": probe_type_counts,
        "origin_decision_threshold": float(
            rows["origin_decision_threshold"].iloc[0]
        ),
        "threshold_comparator": ">=",
        "target_fpir": float(target_fpir),
        "origin_false_accept_count": false_accept_count,
        "origin_fpir": realized_fpir,
        "origin_fpir_wilson95_low": fpir_ci_low,
        "origin_fpir_wilson95_high": fpir_ci_high,
        "origin_true_identification_count": true_identification_count,
        "origin_dir_rank1": (
            float(true_identification_count / mated_count)
            if mated_count
            else None
        ),
        "non_mated_top1_score": _distribution_summary(
            rows.loc[~rows["is_mated"], "origin_top1_score"].to_numpy(
                dtype=np.float64
            ),
            name="non-mated origin top-1 scores",
        ),
        "query_raw_embedding_norm": _distribution_summary(
            rows["query_raw_embedding_norm"].to_numpy(dtype=np.float64),
            name="query raw embedding norms",
        ),
    }


def _independent_threshold(
    comparison: pd.DataFrame,
    *,
    target_fpir: float,
    threshold_selection: str = "maximize_dir",
) -> float:
    if threshold_selection == "non_mated_only":
        return choose_non_mated_fpir_threshold(
            comparison["origin_top1_score"].to_numpy(dtype=np.float64),
            comparison["is_mated"].to_numpy(dtype=bool),
            target_fpir,
        )
    if threshold_selection != "maximize_dir":
        raise ValueError(f"unsupported threshold_selection: {threshold_selection!r}")
    scores = comparison["origin_top1_score"].to_numpy(dtype=np.float64)
    is_mated = comparison["is_mated"].to_numpy(dtype=bool)
    correct = comparison["origin_rank1_correct"].to_numpy(dtype=bool)
    if not (len(scores) == len(is_mated) == len(correct)) or len(scores) == 0:
        raise ValueError("origin threshold audit arrays must have equal non-zero length")
    if not np.all(np.isfinite(scores)):
        raise ValueError("origin threshold audit scores must be finite")
    non_mated_count = int((~is_mated).sum())
    if non_mated_count <= 0:
        raise ValueError("origin threshold audit requires non-mated probes")

    grouped = pd.DataFrame(
        {
            "score": scores,
            "false_accept": (~is_mated).astype(np.int64),
            "true_identification": (is_mated & correct).astype(np.int64),
        }
    ).groupby("score", sort=False, as_index=False).sum()
    grouped = grouped.sort_values("score", ascending=False, kind="stable")
    grouped["false_accept"] = grouped["false_accept"].cumsum()
    grouped["true_identification"] = grouped["true_identification"].cumsum()
    candidates = pd.concat(
        [
            pd.DataFrame(
                {
                    "score": [np.inf],
                    "false_accept": [0],
                    "true_identification": [0],
                }
            ),
            grouped,
        ],
        ignore_index=True,
    )
    feasible = candidates.loc[
        candidates["false_accept"] / non_mated_count
        <= float(target_fpir) + 1e-15
    ]
    if feasible.empty:
        raise RuntimeError("origin threshold audit found no feasible threshold")
    best_true_identifications = int(feasible["true_identification"].max())
    best = feasible.loc[
        feasible["true_identification"].eq(best_true_identifications)
    ].iloc[0]
    return float(best["score"])


def _build_origin_calibration_audit(
    calibration_comparison: pd.DataFrame,
    evaluation_comparison: pd.DataFrame,
    calibration_arrays: dict[str, np.ndarray],
    evaluation_arrays: dict[str, np.ndarray],
    calibration_protocol: OpenSetProtocol,
    evaluation_protocol: OpenSetProtocol,
    *,
    dataset_id: str,
    model_uid: str,
    protocol_uid: str,
    decision_threshold: float,
    target_fpir: float,
    threshold_selection: str = "maximize_dir",
) -> tuple[pd.DataFrame, dict[str, object]]:
    independently_calculated_threshold = _independent_threshold(
        calibration_comparison,
        target_fpir=target_fpir,
        threshold_selection=threshold_selection,
    )
    thresholds_match = (
        np.isinf(decision_threshold)
        and np.isinf(independently_calculated_threshold)
        or np.isclose(
            decision_threshold,
            independently_calculated_threshold,
            rtol=0.0,
            atol=1e-12,
        )
    )
    if not thresholds_match:
        raise RuntimeError(
            "calibration threshold differs from the independent grouped-score audit"
        )
    calibration_rows = _origin_score_rows(
        calibration_comparison,
        calibration_arrays,
        dataset_id=dataset_id,
        model_uid=model_uid,
        protocol_uid=protocol_uid,
        evaluation_split="calibration",
        decision_threshold=decision_threshold,
        target_fpir=target_fpir,
    )
    test_rows = _origin_score_rows(
        evaluation_comparison,
        evaluation_arrays,
        dataset_id=dataset_id,
        model_uid=model_uid,
        protocol_uid=protocol_uid,
        evaluation_split="test",
        decision_threshold=decision_threshold,
        target_fpir=target_fpir,
    )
    audit = pd.concat([calibration_rows, test_rows], ignore_index=True)
    summary: dict[str, object] = {
        "schema_version": 2,
        "artifact_type": "origin_open_set_calibration_diagnostics",
        "dataset_id": str(dataset_id),
        "model_uid": str(model_uid),
        "protocol_uid": str(protocol_uid),
        "score_definition": "maximum cosine similarity over gallery templates",
        "threshold_source_split": "calibration",
        "threshold_selection": (
            "non-mated maximum-score empirical FPIR only; most permissive "
            "tie-preserving threshold within target"
            if threshold_selection == "non_mated_only"
            else "maximize calibration DIR subject to empirical FPIR <= target"
        ),
        "threshold_comparator": ">=",
        "target_fpir": float(target_fpir),
        "origin_decision_threshold": float(decision_threshold),
        "independent_threshold_verification": {
            "algorithm": "descending grouped-score cumulative-count audit",
            "origin_decision_threshold": float(decision_threshold),
            "independently_calculated_threshold": float(
                independently_calculated_threshold
            ),
            "matches": True,
        },
        "splits": {
            "calibration": _split_diagnostics(
                calibration_rows,
                calibration_protocol,
                target_fpir=target_fpir,
                require_target_bound=True,
            ),
            "test": _split_diagnostics(
                test_rows,
                evaluation_protocol,
                target_fpir=target_fpir,
                require_target_bound=False,
            ),
        },
    }
    test_fpir = float(summary["splits"]["test"]["origin_fpir"])
    target_met = test_fpir <= float(target_fpir) + 1e-12
    summary["calibration_transfer_assessment"] = {
        "status": "target_met" if target_met else "failed_target_fpir",
        "target_met": target_met,
        "threshold_dependent_results_valid_for_target_operating_point": target_met,
        "test_used_for_threshold_selection": False,
        "observed_test_fpir": test_fpir,
        "target_fpir": float(target_fpir),
    }
    return audit, summary


def _origin_protocol_comparison(
    arrays: dict[str, np.ndarray],
    *,
    top_k: int,
    progress: ProgressCallback | None,
    progress_message: str,
    progress_details: dict[str, object],
) -> pd.DataFrame:
    return origin_cosine_retrieval(
        arrays["queries"],
        arrays["gallery"],
        query_ids=arrays["query_ids"],
        gallery_ids=arrays["gallery_ids"],
        query_identity_ids=arrays["query_identity_ids"],
        gallery_identity_ids=arrays["gallery_identity_ids"],
        top_k=top_k,
        progress=progress,
        progress_message=progress_message,
        progress_details=progress_details,
    )


def diagnose_step2_survface_origin_calibration(
    prepared: PreparedPopulationInputs,
    selected_manifest: pd.DataFrame,
    *,
    calibration_conditions: Sequence[tuple[int, int]] = (
        (100, 1),
        (200, 1),
        (500, 1),
        (1000, 1),
        (200, 5),
        (200, 20),
    ),
    seed: int = 42,
    target_fpir: float = 0.10,
    top_k: int = 1,
    progress: ProgressCallback | None = None,
) -> SurvFaceOriginCalibrationSweepResult:
    """Diagnose SurvFace origin FPIR without fitting or searching compression.

    Each calibration condition varies the number of gallery identities and
    source images averaged per template. The official test search is performed
    once and each frozen calibration threshold is then applied to those same
    test scores. This isolates gallery-size and template-aggregation effects
    from PCA/PQ behavior.
    """

    normalized_conditions: list[tuple[int, int]] = []
    for gallery_identity_count, enrollment_count in calibration_conditions:
        gallery_value = int(gallery_identity_count)
        enrollment_value = int(enrollment_count)
        if gallery_value <= 0 or enrollment_value <= 0:
            raise ValueError("calibration sweep counts must be positive")
        condition = (gallery_value, enrollment_value)
        if condition in normalized_conditions:
            raise ValueError(f"duplicate calibration condition: {condition}")
        normalized_conditions.append(condition)
    if not normalized_conditions:
        raise ValueError("calibration_conditions must not be empty")
    target_value = float(target_fpir)
    if not 0.0 <= target_value <= 1.0:
        raise ValueError("target_fpir must be inside [0, 1]")

    population = _population_frame(prepared, selected_manifest)
    required_protocol_columns = {
        "protocol_role",
        "probe_type",
        "protocol_index",
    }
    missing = sorted(required_protocol_columns.difference(population.columns))
    if missing:
        raise ValueError(
            "SurvFace selected manifest is missing official protocol columns: "
            f"{missing}"
        )
    official_roles = {
        "gallery",
        "registered_probe",
        "unknown_unknown_probe",
    }
    official = population.loc[
        population["protocol_role"].astype(str).isin(official_roles)
    ].copy()
    if official.empty:
        raise ValueError("SurvFace official evaluation rows are missing")
    official = rebase_survface_protocol_subset_indexes(official)
    evaluation_protocol = build_survface_official_protocol(official)
    evaluation_arrays = _protocol_arrays(evaluation_protocol, population)
    evaluation_comparison = _origin_protocol_comparison(
        evaluation_arrays,
        top_k=top_k,
        progress=progress,
        progress_message="SurvFace official origin retrieval",
        progress_details={"evaluation_split": "test"},
    )

    calibration_sizes = (
        population.loc[population["split"].eq("calibration")]
        .groupby("identity_id")["image_id"]
        .nunique()
    )
    audit_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    condition_diagnostics: list[dict[str, object]] = []
    protocol_uid = "qmul-survface-v1-official-open-set-identification"
    for condition_index, (gallery_identity_count, enrollment_count) in enumerate(
        normalized_conditions
    ):
        eligible_identity_count = int((calibration_sizes > enrollment_count).sum())
        if gallery_identity_count >= eligible_identity_count:
            raise ValueError(
                "calibration condition must reserve at least one eligible "
                "known-unknown identity: "
                f"gallery={gallery_identity_count}, enrollment={enrollment_count}, "
                f"eligible={eligible_identity_count}"
            )
        condition_uid = (
            f"gallery-{gallery_identity_count}_enrollment-{enrollment_count}"
        )
        calibration_protocol = build_calibration_protocol(
            population,
            split_name="calibration",
            gallery_identity_count=gallery_identity_count,
            enrollment_count=enrollment_count,
            seed=seed,
        )
        calibration_arrays = _protocol_arrays(calibration_protocol, population)
        calibration_comparison = _origin_protocol_comparison(
            calibration_arrays,
            top_k=top_k,
            progress=progress,
            progress_message="SurvFace calibration origin retrieval",
            progress_details={
                "condition_index": int(condition_index),
                "condition_count": int(len(normalized_conditions)),
                "calibration_condition_uid": condition_uid,
            },
        )
        threshold = _threshold(
            calibration_comparison,
            score_column="origin_top1_score",
            correct_column="origin_rank1_correct",
            target_fpir=target_value,
        )
        matched_test_protocol = _build_matched_survface_test_protocol(
            evaluation_protocol,
            gallery_identity_count=gallery_identity_count,
            enrollment_count=enrollment_count,
            seed=seed,
        )
        matched_test_arrays = _protocol_arrays(matched_test_protocol, population)
        matched_test_comparison = _origin_protocol_comparison(
            matched_test_arrays,
            top_k=top_k,
            progress=progress,
            progress_message="SurvFace matched-test origin retrieval",
            progress_details={
                "condition_index": int(condition_index),
                "condition_count": int(len(normalized_conditions)),
                "calibration_condition_uid": condition_uid,
            },
        )
        audit, diagnostics = _build_origin_calibration_audit(
            calibration_comparison,
            evaluation_comparison,
            calibration_arrays,
            evaluation_arrays,
            calibration_protocol,
            evaluation_protocol,
            dataset_id=prepared.dataset_id,
            model_uid=prepared.model_uid,
            protocol_uid=protocol_uid,
            decision_threshold=threshold,
            target_fpir=target_value,
        )
        matched_test_rows = _origin_score_rows(
            matched_test_comparison,
            matched_test_arrays,
            dataset_id=prepared.dataset_id,
            model_uid=prepared.model_uid,
            protocol_uid=protocol_uid,
            evaluation_split="test_matched",
            decision_threshold=threshold,
            target_fpir=target_value,
        )
        matched_test_diagnostics = _split_diagnostics(
            matched_test_rows,
            matched_test_protocol,
            target_fpir=target_value,
            require_target_bound=False,
        )
        diagnostics["splits"]["test_matched"] = matched_test_diagnostics
        audit = pd.concat([audit, matched_test_rows], ignore_index=True)
        audit.insert(4, "calibration_condition_uid", condition_uid)
        audit.insert(
            5,
            "calibration_gallery_identity_count",
            gallery_identity_count,
        )
        audit.insert(6, "calibration_enrollment_count", enrollment_count)
        audit_frames.append(audit)

        calibration_split = diagnostics["splits"]["calibration"]
        test_split = diagnostics["splits"]["test"]
        matched_test_split = diagnostics["splits"]["test_matched"]
        summary_rows.append(
            {
                "dataset_id": str(prepared.dataset_id),
                "model_uid": str(prepared.model_uid),
                "protocol_uid": protocol_uid,
                "calibration_condition_uid": condition_uid,
                "seed": int(seed),
                "target_fpir": target_value,
                "calibration_gallery_identity_count": gallery_identity_count,
                "calibration_enrollment_count": enrollment_count,
                "eligible_calibration_identity_count": eligible_identity_count,
                "calibration_source_image_count": int(
                    calibration_split["source_image_count"]
                ),
                "calibration_non_mated_count": int(
                    calibration_split["non_mated_count"]
                ),
                "origin_decision_threshold": float(threshold),
                "calibration_false_accept_count": int(
                    calibration_split["origin_false_accept_count"]
                ),
                "calibration_fpir": float(calibration_split["origin_fpir"]),
                "calibration_fpir_wilson95_low": float(
                    calibration_split["origin_fpir_wilson95_low"]
                ),
                "calibration_fpir_wilson95_high": float(
                    calibration_split["origin_fpir_wilson95_high"]
                ),
                "calibration_non_mated_score_p50": float(
                    calibration_split["non_mated_top1_score"]["p50"]
                ),
                "calibration_non_mated_score_p90": float(
                    calibration_split["non_mated_top1_score"]["p90"]
                ),
                "calibration_non_mated_score_p95": float(
                    calibration_split["non_mated_top1_score"]["p95"]
                ),
                "calibration_non_mated_score_p99": float(
                    calibration_split["non_mated_top1_score"]["p99"]
                ),
                "test_gallery_template_count": int(test_split["template_count"]),
                "test_gallery_source_image_count": int(
                    test_split["source_image_count"]
                ),
                "test_non_mated_count": int(test_split["non_mated_count"]),
                "test_false_accept_count": int(
                    test_split["origin_false_accept_count"]
                ),
                "test_fpir": float(test_split["origin_fpir"]),
                "test_fpir_wilson95_low": float(
                    test_split["origin_fpir_wilson95_low"]
                ),
                "test_fpir_wilson95_high": float(
                    test_split["origin_fpir_wilson95_high"]
                ),
                "test_to_target_fpir_ratio": float(
                    test_split["origin_fpir"] / target_value
                )
                if target_value > 0.0
                else None,
                "test_non_mated_score_p50": float(
                    test_split["non_mated_top1_score"]["p50"]
                ),
                "test_non_mated_score_p90": float(
                    test_split["non_mated_top1_score"]["p90"]
                ),
                "test_non_mated_score_p95": float(
                    test_split["non_mated_top1_score"]["p95"]
                ),
                "test_non_mated_score_p99": float(
                    test_split["non_mated_top1_score"]["p99"]
                ),
                "matched_test_gallery_template_count": int(
                    matched_test_split["template_count"]
                ),
                "matched_test_gallery_source_image_count": int(
                    matched_test_split["source_image_count"]
                ),
                "matched_test_non_mated_count": int(
                    matched_test_split["non_mated_count"]
                ),
                "matched_test_false_accept_count": int(
                    matched_test_split["origin_false_accept_count"]
                ),
                "matched_test_fpir": float(matched_test_split["origin_fpir"]),
                "matched_test_fpir_wilson95_low": float(
                    matched_test_split["origin_fpir_wilson95_low"]
                ),
                "matched_test_fpir_wilson95_high": float(
                    matched_test_split["origin_fpir_wilson95_high"]
                ),
                "matched_test_to_target_fpir_ratio": float(
                    matched_test_split["origin_fpir"] / target_value
                )
                if target_value > 0.0
                else None,
                "matched_test_non_mated_score_p50": float(
                    matched_test_split["non_mated_top1_score"]["p50"]
                ),
                "matched_test_non_mated_score_p90": float(
                    matched_test_split["non_mated_top1_score"]["p90"]
                ),
                "matched_test_non_mated_score_p95": float(
                    matched_test_split["non_mated_top1_score"]["p95"]
                ),
                "matched_test_non_mated_score_p99": float(
                    matched_test_split["non_mated_top1_score"]["p99"]
                ),
            }
        )
        condition_diagnostics.append(
            {
                "calibration_condition_uid": condition_uid,
                "calibration_gallery_identity_count": gallery_identity_count,
                "calibration_enrollment_count": enrollment_count,
                "eligible_calibration_identity_count": eligible_identity_count,
                **diagnostics,
            }
        )

    score_audit = pd.concat(audit_frames, ignore_index=True)
    condition_summary = pd.DataFrame(summary_rows)
    diagnostics_payload: dict[str, object] = {
        "schema_version": 2,
        "artifact_type": "survface_origin_fpir_calibration_sweep",
        "dataset_id": str(prepared.dataset_id),
        "model_uid": str(prepared.model_uid),
        "protocol_uid": protocol_uid,
        "seed": int(seed),
        "target_fpir": target_value,
        "score_definition": "maximum cosine similarity over gallery templates",
        "threshold_source_split": "calibration",
        "test_threshold_recalibration": False,
        "matched_test_definition": (
            "deterministic official-test gallery subset with the same gallery "
            "identity and enrollment counts as each calibration condition"
        ),
        "calibration_identity_count": int(len(calibration_sizes)),
        "conditions": condition_diagnostics,
    }
    return SurvFaceOriginCalibrationSweepResult(
        score_audit=score_audit,
        condition_summary=condition_summary,
        diagnostics=diagnostics_payload,
    )


def _compressed_matrix(family: str, compressor: Any, matrix: np.ndarray) -> np.ndarray:
    if family == "pca":
        return compressor.transform(matrix)
    if family == "pq":
        return compressor.transform_profile(matrix).vectors
    raise ValueError(f"unsupported compression family: {family}")


def _threshold(
    comparison: pd.DataFrame,
    *,
    score_column: str,
    correct_column: str,
    target_fpir: float,
    threshold_selection: str = "maximize_dir",
) -> float:
    if threshold_selection == "non_mated_only":
        value = float(
            choose_non_mated_fpir_threshold(
                comparison[score_column].to_numpy(dtype=float),
                comparison["is_mated"].to_numpy(dtype=bool),
                target_fpir,
            )
        )
    elif threshold_selection == "maximize_dir":
        value = float(
            choose_threshold(
                comparison[score_column].to_numpy(dtype=float),
                comparison["is_mated"].to_numpy(dtype=bool),
                comparison[correct_column].to_numpy(dtype=bool),
                target_fpir,
            )
        )
    else:
        raise ValueError(f"unsupported threshold_selection: {threshold_selection!r}")
    if not np.isfinite(value):
        raise ValueError(
            "threshold calibration was non-finite; verify calibration coverage"
        )
    return value


def _storage_metadata(family: str, profile_result: Any) -> dict[str, object]:
    if family == "pca":
        storage_bytes = int(profile_result.metadata["storage_bytes_per_vector"])
        codebook_bytes = 0
        codebook_source = "not_applicable"
        codec_parameter_bytes = int(
            profile_result.metadata.get("codec_parameter_bytes", 0)
        )
        codec_parameter_source = str(
            profile_result.metadata.get(
                "codec_parameter_bytes_source",
                "not_recorded",
            )
        )
    else:
        storage_bytes = int(profile_result.metadata["code_bytes"])
        codebook_bytes = int(profile_result.metadata["codebook_bytes"])
        codebook_source = str(profile_result.metadata["codebook_bytes_source"])
        codec_parameter_bytes = int(
            profile_result.metadata.get("codec_parameter_bytes", codebook_bytes)
        )
        codec_parameter_source = str(
            profile_result.metadata.get(
                "codec_parameter_bytes_source",
                codebook_source,
            )
        )
    if storage_bytes <= 0 or codebook_bytes < 0 or codec_parameter_bytes < 0:
        raise ValueError("compression storage metadata must be non-negative")
    return {
        "storage_bytes_per_embedding": storage_bytes,
        "codebook_bytes": codebook_bytes,
        "codebook_bytes_source": codebook_source,
        "codec_parameter_bytes": codec_parameter_bytes,
        "codec_parameter_bytes_source": codec_parameter_source,
    }


def _normalize_target_fpirs(
    primary_target_fpir: float,
    target_fpirs: Sequence[float] | None,
) -> tuple[float, ...]:
    primary = float(primary_target_fpir)
    requested = (
        (primary,)
        if target_fpirs is None
        else (primary, *(float(value) for value in target_fpirs))
    )
    normalized = tuple(dict.fromkeys(requested))
    if any(not 0.0 <= value <= 1.0 for value in normalized):
        raise ValueError("target FPIR values must be between 0 and 1")
    return normalized


def _summarize(paired: pd.DataFrame, retrieval: pd.DataFrame) -> pd.DataFrame:
    paired_summary = (
        paired.groupby(
            ["compression_family", "compression_profile"],
            sort=True,
        )
        .agg(
            sample_count=("sample_id", "size"),
            mean_angular_error_rad=("angular_error_rad", "mean"),
            p95_angular_error_rad=(
                "angular_error_rad",
                lambda values: values.quantile(0.95),
            ),
            mean_reconstruction_mse=("reconstruction_mse", "mean"),
            storage_bytes_per_embedding=(
                "storage_bytes_per_embedding",
                "first",
            ),
            codebook_bytes=("codebook_bytes", "first"),
            codec_parameter_bytes=("codec_parameter_bytes", "first"),
        )
        .reset_index()
    )
    records = []
    retrieval_group_columns = [
        "compression_family",
        "compression_profile",
        "search_mode",
        "threshold_policy",
    ]
    if "target_fpir" in retrieval:
        retrieval_group_columns.append("target_fpir")
    for keys, group in retrieval.groupby(
        retrieval_group_columns,
        sort=True,
    ):
        family, profile, search_mode, policy = keys[:4]
        operating_target = float(keys[4]) if len(keys) == 5 else np.nan
        mated = group["is_mated"].astype(bool).to_numpy()
        non_mated = ~mated
        origin_accepted = group["origin_accepted"].astype(bool).to_numpy()
        compressed_accepted = group["compressed_accepted"].astype(bool).to_numpy()
        origin_correct = group["origin_rank1_correct"].astype(bool).to_numpy()
        compressed_correct = group["compressed_rank1_correct"].astype(
            bool
        ).to_numpy()
        origin_dir = origin_accepted & origin_correct & mated
        compressed_dir = compressed_accepted & compressed_correct & mated
        origin_false_accept = origin_accepted & non_mated
        compressed_false_accept = compressed_accepted & non_mated
        mated_count = int(mated.sum())
        non_mated_count = int(non_mated.sum())
        origin_dir_count = int(origin_dir.sum())
        compressed_dir_count = int(compressed_dir.sum())
        both_dir_count = int((origin_dir & compressed_dir).sum())
        origin_false_accept_count = int(origin_false_accept.sum())
        compressed_false_accept_count = int(compressed_false_accept.sum())
        both_false_accept_count = int(
            (origin_false_accept & compressed_false_accept).sum()
        )
        if mated_count:
            origin_dir_ci = wilson_score_interval(
                origin_dir_count,
                mated_count,
            )
            compressed_dir_ci = wilson_score_interval(
                compressed_dir_count,
                mated_count,
            )
            dir_delta_ci = paired_binary_rate_difference_bootstrap_interval(
                origin_dir_count,
                compressed_dir_count,
                both_dir_count,
                mated_count,
            )
        else:
            origin_dir_ci = (np.nan, np.nan)
            compressed_dir_ci = (np.nan, np.nan)
            dir_delta_ci = (np.nan, np.nan)
        if non_mated_count:
            origin_fpir_ci = wilson_score_interval(
                origin_false_accept_count,
                non_mated_count,
            )
            compressed_fpir_ci = wilson_score_interval(
                compressed_false_accept_count,
                non_mated_count,
            )
            fpir_delta_ci = paired_binary_rate_difference_bootstrap_interval(
                origin_false_accept_count,
                compressed_false_accept_count,
                both_false_accept_count,
                non_mated_count,
            )
        else:
            origin_fpir_ci = (np.nan, np.nan)
            compressed_fpir_ci = (np.nan, np.nan)
            fpir_delta_ci = (np.nan, np.nan)
        origin_dir_rate = (
            float(origin_dir_count / mated_count) if mated_count else np.nan
        )
        compressed_dir_rate = (
            float(compressed_dir_count / mated_count) if mated_count else np.nan
        )
        origin_fpir = (
            float(origin_false_accept_count / non_mated_count)
            if non_mated_count
            else np.nan
        )
        compressed_fpir = (
            float(compressed_false_accept_count / non_mated_count)
            if non_mated_count
            else np.nan
        )
        records.append(
            {
                "compression_family": family,
                "compression_profile": profile,
                "search_mode": search_mode,
                "threshold_policy": policy,
                "target_fpir": operating_target,
                "query_count": int(len(group)),
                "mated_count": mated_count,
                "non_mated_count": non_mated_count,
                "origin_dir_rank1_count": origin_dir_count,
                "compressed_dir_rank1_count": compressed_dir_count,
                "both_dir_rank1_count": both_dir_count,
                "origin_dir_rank1_denominator": mated_count,
                "compressed_dir_rank1_denominator": mated_count,
                "origin_dir_rank1": origin_dir_rate,
                "compressed_dir_rank1": compressed_dir_rate,
                "dir_rank1": compressed_dir_rate,
                "origin_dir_rank1_wilson95_low": origin_dir_ci[0],
                "origin_dir_rank1_wilson95_high": origin_dir_ci[1],
                "compressed_dir_rank1_wilson95_low": compressed_dir_ci[0],
                "compressed_dir_rank1_wilson95_high": compressed_dir_ci[1],
                "compressed_minus_origin_dir_rank1": (
                    compressed_dir_rate - origin_dir_rate
                ),
                "compressed_minus_origin_dir_rank1_paired_bootstrap95_low": (
                    dir_delta_ci[0]
                ),
                "compressed_minus_origin_dir_rank1_paired_bootstrap95_high": (
                    dir_delta_ci[1]
                ),
                "origin_false_accept_count": origin_false_accept_count,
                "compressed_false_accept_count": compressed_false_accept_count,
                "both_false_accept_count": both_false_accept_count,
                "origin_fpir_denominator": non_mated_count,
                "compressed_fpir_denominator": non_mated_count,
                "origin_fpir": origin_fpir,
                "origin_realized_fpir": origin_fpir,
                "compressed_fpir": compressed_fpir,
                "compressed_realized_fpir": compressed_fpir,
                "fpir": compressed_fpir,
                "origin_fpir_wilson95_low": origin_fpir_ci[0],
                "origin_fpir_wilson95_high": origin_fpir_ci[1],
                "compressed_fpir_wilson95_low": compressed_fpir_ci[0],
                "compressed_fpir_wilson95_high": compressed_fpir_ci[1],
                "compressed_minus_origin_fpir": compressed_fpir - origin_fpir,
                "compressed_minus_origin_fpir_paired_bootstrap95_low": (
                    fpir_delta_ci[0]
                ),
                "compressed_minus_origin_fpir_paired_bootstrap95_high": (
                    fpir_delta_ci[1]
                ),
                "confidence_interval_unit": "probe",
                "rate_confidence_interval_method": "wilson_score",
                "difference_confidence_interval_method": (
                    "paired_nonparametric_bootstrap_percentile"
                ),
                "difference_confidence_interval_resamples": (
                    PAIRED_BOOTSTRAP_RESAMPLES
                ),
                "difference_confidence_interval_random_seed": (
                    PAIRED_BOOTSTRAP_RANDOM_SEED
                ),
                "agreement_with_origin": float(
                    group["agreement_with_origin"].mean()
                ),
                "threshold_crossing_rate": float(
                    group["threshold_crossing"].mean()
                ),
                "gallery_template_count": int(
                    group["gallery_template_count"].iloc[0]
                ),
                "origin_storage_bytes_per_embedding": 512 * 4,
                "origin_gallery_storage_bytes": int(
                    group["gallery_template_count"].iloc[0]
                )
                * 512
                * 4,
                "compressed_gallery_storage_bytes": int(
                    group["gallery_template_count"].iloc[0]
                )
                * int(group["storage_bytes_per_embedding"].iloc[0])
                + int(group["codec_parameter_bytes"].iloc[0]),
                "amortized_storage_bytes_per_gallery_template": float(
                    int(group["storage_bytes_per_embedding"].iloc[0])
                    + int(group["codec_parameter_bytes"].iloc[0])
                    / int(group["gallery_template_count"].iloc[0])
                ),
                "compressed_search_latency_ms_total": float(
                    group["compressed_search_latency_ms_total"].iloc[0]
                ),
                "compressed_search_latency_ms_per_query": float(
                    group["compressed_search_latency_ms_per_query"].iloc[0]
                ),
                "compressed_search_queries_per_second": float(
                    group["compressed_search_queries_per_second"].iloc[0]
                ),
            }
        )
    return pd.DataFrame.from_records(records).merge(
        paired_summary,
        on=["compression_family", "compression_profile"],
        how="left",
        validate="many_to_one",
    )


def _annotate_rfw_custom_query_boundaries(
    frame: pd.DataFrame,
    population: pd.DataFrame,
) -> pd.DataFrame:
    """Retain RFW demographic and checkpoint-overlap boundaries per probe."""

    if "rfw_group" not in population.columns:
        return frame
    lookup = population.loc[:, ["image_id", "rfw_group"]].copy()
    conflicts = lookup.groupby("image_id")["rfw_group"].nunique(dropna=False)
    if (conflicts != 1).any():
        raise ValueError("RFW custom image_id must map to exactly one rfw_group")
    group_by_image = (
        lookup.drop_duplicates("image_id").set_index("image_id")["rfw_group"]
    )
    annotated = frame.copy()
    annotated["rfw_group"] = annotated["query_id"].astype(str).map(
        group_by_image
    )
    if annotated["rfw_group"].isna().any():
        missing = annotated.loc[
            annotated["rfw_group"].isna(), "query_id"
        ].astype(str)
        raise ValueError(
            "RFW custom retrieval query is missing demographic lineage: "
            f"{missing.iloc[0]!r}"
        )
    annotated["checkpoint_training_identity_overlap_status"] = "UNKNOWN"
    annotated["strict_unseen_identity_evidence"] = False
    return annotated


def _summarize_rfw_custom_demographics(
    paired: pd.DataFrame,
    retrieval: pd.DataFrame,
) -> pd.DataFrame:
    if "rfw_group" not in retrieval.columns:
        return pd.DataFrame()
    summaries: list[pd.DataFrame] = []
    for group_name, group in retrieval.groupby("rfw_group", sort=True):
        summary = _summarize(paired, group)
        summary.insert(0, "rfw_group", str(group_name))
        summary["checkpoint_training_identity_overlap_status"] = "UNKNOWN"
        summary["strict_unseen_identity_evidence"] = False
        summaries.append(summary)
    if not summaries:
        raise ValueError("RFW custom demographic summary has no groups")
    return pd.concat(summaries, ignore_index=True)


def _characterize_population_with_protocols(
    prepared: PreparedPopulationInputs,
    population: pd.DataFrame,
    *,
    development_matrix: np.ndarray,
    calibration_protocol: OpenSetProtocol,
    evaluation_protocol: OpenSetProtocol,
    protocol_uid: str,
    pca_dimensions: Sequence[int],
    pq_settings: Sequence[tuple[int, int]],
    seed: int,
    target_fpir: float,
    target_fpirs: Sequence[float] | None = None,
    top_k: int,
    threshold_selection: str = "maximize_dir",
    calibration_contract: dict[str, object] | None = None,
    progress: ProgressCallback | None = None,
) -> Step2CompressionResult:
    operating_targets = _normalize_target_fpirs(target_fpir, target_fpirs)
    requested_pca = tuple(int(value) for value in pca_dimensions)
    requested_pq = tuple((int(m), int(bits)) for m, bits in pq_settings)
    if not requested_pca and not requested_pq:
        raise ValueError("at least one PCA or PQ profile is required")
    development_values = np.asarray(development_matrix, dtype=np.float32)
    if development_values.ndim != 2 or development_values.shape[1] != 512:
        raise ValueError("development_matrix must have shape [N, 512]")

    pca_models = (
        fit_pca_family(
            development_values,
            dimensions=requested_pca,
            random_state=seed,
        )
        if requested_pca
        else {}
    )
    compressors: list[tuple[str, str, Any]] = [
        ("pca", profile, compressor)
        for profile, compressor in pca_models.items()
    ]
    compressor_total = len(requested_pca) + len(requested_pq)
    if pca_models:
        _emit(
            progress,
            f"{prepared.dataset_id} compressor fit",
            processed=len(pca_models),
            total=compressor_total,
            family="pca",
        )
    for m, nbits in requested_pq:
        compressor = PQCompressor(
            source_dim=512,
            m=m,
            nbits=nbits,
            source_profile=ORIGIN_512,
            random_state=int(seed),
        ).fit(development_values)
        compressors.append(("pq", pq_profile_name(m, nbits), compressor))
        _emit(
            progress,
            f"{prepared.dataset_id} compressor fit",
            processed=len(compressors),
            total=compressor_total,
            family="pq",
            profile=pq_profile_name(m, nbits),
        )

    calibration = _protocol_arrays(calibration_protocol, population)
    evaluation = _protocol_arrays(evaluation_protocol, population)
    work_per_cosine_mode = 2 * (
        len(calibration["queries"]) + len(evaluation["queries"])
    )
    cosine_mode_count = sum(
        2 if family == "pca" else 1 for family, _, _ in compressors
    )
    retrieval_work_total = cosine_mode_count * work_per_cosine_mode
    full_matrix = np.stack(population["origin_embedding"]).astype(np.float32)
    paired_frames = []
    retrieval_frames = []
    origin_calibration_reference: pd.DataFrame | None = None
    origin_evaluation_reference: pd.DataFrame | None = None
    origin_thresholds: dict[float, float] | None = None
    progress_offset = 0
    for family, profile, compressor in compressors:
        full_result = compressor.transform_profile(full_matrix)
        storage = _storage_metadata(family, full_result)
        paired = paired_embedding_metrics(
            full_matrix,
            full_result.vectors,
            reconstructed_embeddings=full_result.reconstructed_vectors,
            sample_ids=population["image_id"].astype(str).to_numpy(),
            compression_family=family,
            compression_profile=profile,
        )
        paired.insert(0, "dataset", prepared.dataset_id)
        paired.insert(1, "model_uid", prepared.model_uid)
        for column, value in storage.items():
            paired[column] = value
        paired_frames.append(paired)

        calibration_comparison = compare_cosine_retrieval(
            calibration["queries"],
            calibration["gallery"],
            _compressed_matrix(family, compressor, calibration["queries"]),
            _compressed_matrix(family, compressor, calibration["gallery"]),
            query_ids=calibration["query_ids"],
            gallery_ids=calibration["gallery_ids"],
            query_identity_ids=calibration["query_identity_ids"],
            gallery_identity_ids=calibration["gallery_identity_ids"],
            compression_family=family,
            compression_profile=profile,
            search_mode=(
                "pca_direct_cosine"
                if family == "pca"
                else "pq_reconstruction_cosine"
            ),
            top_k=min(int(top_k), len(calibration["gallery"])),
            progress=progress,
            progress_message=(
                f"{prepared.dataset_id} compression retrieval"
            ),
            progress_offset=progress_offset,
            progress_total=retrieval_work_total,
            progress_details={
                "profile": profile,
                "split": "calibration",
            },
        )
        if origin_calibration_reference is None:
            origin_calibration_reference = calibration_comparison.copy()
            origin_thresholds = {
                target: _threshold(
                    calibration_comparison,
                    score_column="origin_top1_score",
                    correct_column="origin_rank1_correct",
                    target_fpir=target,
                    threshold_selection=threshold_selection,
                )
                for target in operating_targets
            }
        else:
            _assert_same_origin_comparison(
                origin_calibration_reference,
                calibration_comparison,
                split_name="calibration",
            )
        compressed_thresholds = {
            target: _threshold(
                calibration_comparison,
                score_column="compressed_top1_score",
                correct_column="compressed_rank1_correct",
                target_fpir=target,
                threshold_selection=threshold_selection,
            )
            for target in operating_targets
        }
        adc_calibration_thresholds: dict[float, float] | None = None
        if family == "pq":
            calibration_gallery_codes = compressor.encode(calibration["gallery"])
            (
                adc_calibration_distances,
                adc_calibration_indices,
                _,
            ) = compressor.search_adc_with_metrics(
                calibration["queries"],
                calibration_gallery_codes,
                top_k=min(int(top_k), len(calibration["gallery"])),
            )
            adc_calibration_comparison = compare_pq_adc_retrieval(
                calibration_comparison,
                calibration["queries"],
                calibration["gallery"],
                adc_calibration_distances,
                adc_calibration_indices,
                query_ids=calibration["query_ids"],
                gallery_ids=calibration["gallery_ids"],
                query_identity_ids=calibration["query_identity_ids"],
                gallery_identity_ids=calibration["gallery_identity_ids"],
                compression_profile=profile,
            )
            adc_calibration_thresholds = {
                target: _threshold(
                    adc_calibration_comparison,
                    score_column="compressed_top1_score",
                    correct_column="compressed_rank1_correct",
                    target_fpir=target,
                    threshold_selection=threshold_selection,
                )
                for target in operating_targets
            }
        evaluation_queries = _compressed_matrix(
            family,
            compressor,
            evaluation["queries"],
        )
        evaluation_gallery = _compressed_matrix(
            family,
            compressor,
            evaluation["gallery"],
        )
        evaluation_comparison = compare_cosine_retrieval(
            evaluation["queries"],
            evaluation["gallery"],
            evaluation_queries,
            evaluation_gallery,
            query_ids=evaluation["query_ids"],
            gallery_ids=evaluation["gallery_ids"],
            query_identity_ids=evaluation["query_identity_ids"],
            gallery_identity_ids=evaluation["gallery_identity_ids"],
            compression_family=family,
            compression_profile=profile,
            search_mode=(
                "pca_direct_cosine"
                if family == "pca"
                else "pq_reconstruction_cosine"
            ),
            top_k=min(int(top_k), len(evaluation["gallery"])),
            progress=progress,
            progress_message=(
                f"{prepared.dataset_id} compression retrieval"
            ),
            progress_offset=(
                progress_offset
                + 2 * len(calibration["queries"])
            ),
            progress_total=retrieval_work_total,
            progress_details={
                "profile": profile,
                "split": "evaluation",
            },
        )
        if origin_evaluation_reference is None:
            origin_evaluation_reference = evaluation_comparison.copy()
        else:
            _assert_same_origin_comparison(
                origin_evaluation_reference,
                evaluation_comparison,
                split_name="test",
            )
        if origin_thresholds is None:
            raise RuntimeError("origin calibration threshold was not initialized")
        for operating_target in operating_targets:
            origin_threshold = origin_thresholds[operating_target]
            for threshold_policy, operating_threshold in (
                ("frozen_origin", origin_threshold),
                (
                    "recalibrated_compressed",
                    compressed_thresholds[operating_target],
                ),
            ):
                compared = apply_retrieval_thresholds(
                    evaluation_comparison,
                    origin_threshold=origin_threshold,
                    compressed_threshold=operating_threshold,
                )
                compared.insert(0, "dataset", prepared.dataset_id)
                compared.insert(1, "model_uid", prepared.model_uid)
                compared["protocol_uid"] = str(protocol_uid)
                compared["target_fpir"] = float(operating_target)
                compared["threshold_policy"] = threshold_policy
                compared["threshold_source_split"] = "calibration"
                compared["evaluation_split"] = "test"
                for column, value in storage.items():
                    compared[column] = value
                compared["gallery_template_count"] = len(evaluation["gallery"])
                retrieval_frames.append(
                    _annotate_rfw_custom_query_boundaries(compared, population)
                )
        progress_offset += work_per_cosine_mode

        if family == "pca":
            reconstructed_calibration = compressor.transform_profile(
                calibration["queries"]
            ).reconstructed_vectors
            reconstructed_calibration_gallery = compressor.transform_profile(
                calibration["gallery"]
            ).reconstructed_vectors
            reconstructed_calibration_comparison = compare_cosine_retrieval(
                calibration["queries"],
                calibration["gallery"],
                reconstructed_calibration,
                reconstructed_calibration_gallery,
                query_ids=calibration["query_ids"],
                gallery_ids=calibration["gallery_ids"],
                query_identity_ids=calibration["query_identity_ids"],
                gallery_identity_ids=calibration["gallery_identity_ids"],
                compression_family=family,
                compression_profile=profile,
                search_mode="pca_reconstruction_cosine",
                top_k=min(int(top_k), len(calibration["gallery"])),
                progress=progress,
                progress_message=(
                    f"{prepared.dataset_id} compression retrieval"
                ),
                progress_offset=progress_offset,
                progress_total=retrieval_work_total,
                progress_details={
                    "profile": profile,
                    "split": "calibration",
                    "search_mode": "pca_reconstruction_cosine",
                },
            )
            _assert_same_origin_comparison(
                origin_calibration_reference,
                reconstructed_calibration_comparison,
                split_name="calibration",
            )
            reconstructed_thresholds = {
                target: _threshold(
                    reconstructed_calibration_comparison,
                    score_column="compressed_top1_score",
                    correct_column="compressed_rank1_correct",
                    target_fpir=target,
                    threshold_selection=threshold_selection,
                )
                for target in operating_targets
            }
            reconstructed_evaluation = compressor.transform_profile(
                evaluation["queries"]
            ).reconstructed_vectors
            reconstructed_evaluation_gallery = compressor.transform_profile(
                evaluation["gallery"]
            ).reconstructed_vectors
            reconstructed_evaluation_comparison = compare_cosine_retrieval(
                evaluation["queries"],
                evaluation["gallery"],
                reconstructed_evaluation,
                reconstructed_evaluation_gallery,
                query_ids=evaluation["query_ids"],
                gallery_ids=evaluation["gallery_ids"],
                query_identity_ids=evaluation["query_identity_ids"],
                gallery_identity_ids=evaluation["gallery_identity_ids"],
                compression_family=family,
                compression_profile=profile,
                search_mode="pca_reconstruction_cosine",
                top_k=min(int(top_k), len(evaluation["gallery"])),
                progress=progress,
                progress_message=(
                    f"{prepared.dataset_id} compression retrieval"
                ),
                progress_offset=(
                    progress_offset + 2 * len(calibration["queries"])
                ),
                progress_total=retrieval_work_total,
                progress_details={
                    "profile": profile,
                    "split": "evaluation",
                    "search_mode": "pca_reconstruction_cosine",
                },
            )
            _assert_same_origin_comparison(
                origin_evaluation_reference,
                reconstructed_evaluation_comparison,
                split_name="test",
            )
            for operating_target in operating_targets:
                origin_threshold = origin_thresholds[operating_target]
                for threshold_policy, operating_threshold in (
                    ("frozen_origin", origin_threshold),
                    (
                        "recalibrated_compressed",
                        reconstructed_thresholds[operating_target],
                    ),
                ):
                    reconstructed_compared = apply_retrieval_thresholds(
                        reconstructed_evaluation_comparison,
                        origin_threshold=origin_threshold,
                        compressed_threshold=operating_threshold,
                    )
                    reconstructed_compared.insert(
                        0, "dataset", prepared.dataset_id
                    )
                    reconstructed_compared.insert(
                        1, "model_uid", prepared.model_uid
                    )
                    reconstructed_compared["protocol_uid"] = str(protocol_uid)
                    reconstructed_compared["target_fpir"] = float(
                        operating_target
                    )
                    reconstructed_compared["threshold_policy"] = threshold_policy
                    reconstructed_compared[
                        "threshold_source_split"
                    ] = "calibration"
                    reconstructed_compared["evaluation_split"] = "test"
                    for column, value in storage.items():
                        reconstructed_compared[column] = value
                    reconstructed_compared["gallery_template_count"] = len(
                        evaluation["gallery"]
                    )
                    retrieval_frames.append(
                        _annotate_rfw_custom_query_boundaries(
                            reconstructed_compared,
                            population,
                        )
                    )
            progress_offset += work_per_cosine_mode
        if family == "pq":
            if adc_calibration_thresholds is None:
                raise RuntimeError("PQ ADC calibration threshold was not initialized")
            gallery_encode_started = perf_counter()
            evaluation_gallery_codes = compressor.encode(evaluation["gallery"])
            gallery_encode_elapsed = perf_counter() - gallery_encode_started
            (
                adc_evaluation_distances,
                adc_evaluation_indices,
                adc_search_metrics,
            ) = compressor.search_adc_with_metrics(
                evaluation["queries"],
                evaluation_gallery_codes,
                top_k=min(int(top_k), len(evaluation["gallery"])),
            )
            adc_search_metrics["compressed_gallery_encode_latency_ms"] = float(
                gallery_encode_elapsed * 1000.0
            )
            adc_evaluation_comparison = compare_pq_adc_retrieval(
                evaluation_comparison,
                evaluation["queries"],
                evaluation["gallery"],
                adc_evaluation_distances,
                adc_evaluation_indices,
                query_ids=evaluation["query_ids"],
                gallery_ids=evaluation["gallery_ids"],
                query_identity_ids=evaluation["query_identity_ids"],
                gallery_identity_ids=evaluation["gallery_identity_ids"],
                compression_profile=profile,
                search_metrics=adc_search_metrics,
            )
            for operating_target in operating_targets:
                adc_compared = apply_retrieval_thresholds(
                    adc_evaluation_comparison,
                    origin_threshold=origin_thresholds[operating_target],
                    compressed_threshold=adc_calibration_thresholds[
                        operating_target
                    ],
                )
                adc_compared.insert(0, "dataset", prepared.dataset_id)
                adc_compared.insert(1, "model_uid", prepared.model_uid)
                adc_compared["protocol_uid"] = str(protocol_uid)
                adc_compared["target_fpir"] = float(operating_target)
                adc_compared["threshold_policy"] = "recalibrated_compressed"
                adc_compared["threshold_source_split"] = "calibration"
                adc_compared["evaluation_split"] = "test"
                for column, value in storage.items():
                    adc_compared[column] = value
                adc_compared["gallery_template_count"] = len(
                    evaluation["gallery"]
                )
                retrieval_frames.append(
                    _annotate_rfw_custom_query_boundaries(
                        adc_compared,
                        population,
                    )
                )

    paired_metrics = pd.concat(paired_frames, ignore_index=True)
    retrieval_metrics = pd.concat(retrieval_frames, ignore_index=True)
    if paired_metrics["origin_fallback_used"].astype(bool).any():
        raise RuntimeError("paired metrics contain origin fallback rows")
    if retrieval_metrics["origin_fallback_used"].astype(bool).any():
        raise RuntimeError("retrieval metrics contain origin fallback rows")
    if (
        origin_calibration_reference is None
        or origin_evaluation_reference is None
        or origin_thresholds is None
    ):
        raise RuntimeError("origin calibration diagnostics were not initialized")
    audit_frames: list[pd.DataFrame] = []
    diagnostics_by_target: dict[str, dict[str, object]] = {}
    for operating_target in operating_targets:
        audit, diagnostics = _build_origin_calibration_audit(
            origin_calibration_reference,
            origin_evaluation_reference,
            calibration,
            evaluation,
            calibration_protocol,
            evaluation_protocol,
            dataset_id=prepared.dataset_id,
            model_uid=prepared.model_uid,
            protocol_uid=protocol_uid,
            decision_threshold=origin_thresholds[operating_target],
            target_fpir=operating_target,
            threshold_selection=threshold_selection,
        )
        if calibration_contract is not None:
            diagnostics["calibration_contract"] = dict(calibration_contract)
        audit_frames.append(audit)
        diagnostics_by_target[f"{operating_target:.12g}"] = diagnostics
    origin_score_audit = pd.concat(audit_frames, ignore_index=True)
    calibration_diagnostics = diagnostics_by_target[f"{float(target_fpir):.12g}"]
    return Step2CompressionResult(
        paired_metrics=paired_metrics,
        retrieval_metrics=retrieval_metrics,
        origin_score_audit=origin_score_audit,
        calibration_diagnostics=calibration_diagnostics,
        summary=_summarize(paired_metrics, retrieval_metrics),
        fitted_codecs=tuple(compressors),
        calibration_diagnostics_by_target=diagnostics_by_target,
        demographic_summary=_summarize_rfw_custom_demographics(
            paired_metrics,
            retrieval_metrics,
        ),
    )


def characterize_step2_compression(
    prepared: PreparedPopulationInputs,
    selected_manifest: pd.DataFrame,
    *,
    gallery_identities: Sequence[str],
    unknown_unknown_identities: Sequence[str],
    pca_dimensions: Sequence[int],
    pq_settings: Sequence[tuple[int, int]],
    seed: int = 42,
    target_fpir: float = 0.01,
    target_fpirs: Sequence[float] | None = None,
    enrollment_count: int = 5,
    calibration_gallery_identities: int = 20,
    top_k: int = 20,
    progress: ProgressCallback | None = None,
) -> Step2CompressionResult:
    """Fit on development, calibrate on calibration, and evaluate on test."""

    population = _population_frame(prepared, selected_manifest)
    gallery_ids = _identity_tuple(gallery_identities, name="gallery_identities")
    unknown_ids = _identity_tuple(
        unknown_unknown_identities,
        name="unknown_unknown_identities",
    )
    if set(gallery_ids).intersection(unknown_ids):
        raise ValueError("gallery and unknown-unknown identities must be disjoint")

    development = population.loc[population["split"].eq("development")]
    if development.empty:
        raise ValueError("development split is required to fit compressors")
    development_matrix = np.stack(development["origin_embedding"]).astype(
        np.float32
    )
    calibration_sizes = (
        population.loc[population["split"].eq("calibration")]
        .groupby("identity_id")["image_id"]
        .nunique()
    )
    eligible_calibration = int((calibration_sizes > 1).sum())
    calibration_gallery_count = min(
        int(calibration_gallery_identities),
        max(1, eligible_calibration),
    )
    calibration_protocol = build_calibration_protocol(
        population,
        split_name="calibration",
        gallery_identity_count=calibration_gallery_count,
        enrollment_count=1,
        seed=seed,
    )
    test_protocol = build_open_set_protocol(
        population,
        gallery_ids,
        unknown_ids,
        enrollment_count=enrollment_count,
        seed=seed,
    )
    return _characterize_population_with_protocols(
        prepared,
        population,
        development_matrix=development_matrix,
        calibration_protocol=calibration_protocol,
        evaluation_protocol=test_protocol,
        protocol_uid="lfw-identity-disjoint-open-set-v1",
        pca_dimensions=pca_dimensions,
        pq_settings=pq_settings,
        seed=seed,
        target_fpir=target_fpir,
        target_fpirs=target_fpirs,
        top_k=top_k,
        progress=progress,
    )


def characterize_step2_survface_compression(
    prepared: PreparedPopulationInputs,
    selected_manifest: pd.DataFrame,
    *,
    pca_dimensions: Sequence[int],
    pq_settings: Sequence[tuple[int, int]],
    seed: int = 42,
    target_fpir: float = 0.10,
    target_fpirs: Sequence[float] | None = None,
    calibration_gallery_identities: int = 3000,
    top_k: int = 20,
    progress: ProgressCallback | None = None,
) -> Step2CompressionResult:
    """Fit on a matched training watch-list and evaluate the official protocol.

    The selected population must contain the identity-disjoint training
    development/calibration rows and the official gallery, registered-probe,
    and unknown-unknown rows. Training identities are deterministically
    repartitioned into a half-gallery watch-list and non-mated probes. PCA/PQ
    sees only watch-list enrollment images and thresholds use only non-mated
    maximum scores. Official protocol order is preserved by
    ``build_survface_official_protocol``; the official test set is never used
    to fit PCA/PQ or thresholds.
    """

    population = _population_frame(prepared, selected_manifest)
    calibration_protocol = build_survface_matched_calibration_protocol(
        population,
        gallery_identity_count=int(calibration_gallery_identities),
        seed=seed,
    )
    fit_image_ids = set(calibration_protocol.gallery["image_id"].astype(str))
    fit_rows = population.loc[
        population["image_id"].astype(str).isin(fit_image_ids)
    ]
    if len(fit_rows) != len(fit_image_ids):
        raise ValueError("SurvFace watch-list fit rows are incomplete")
    development_matrix = np.stack(fit_rows["origin_embedding"]).astype(
        np.float32
    )

    required_protocol_columns = {
        "protocol_role",
        "probe_type",
        "protocol_index",
    }
    missing = sorted(required_protocol_columns.difference(population.columns))
    if missing:
        raise ValueError(
            f"SurvFace selected manifest is missing official protocol columns: "
            f"{missing}"
        )
    official_roles = {
        "gallery",
        "registered_probe",
        "unknown_unknown_probe",
    }
    official = population.loc[
        population["protocol_role"].astype(str).isin(official_roles)
    ].copy()
    if official.empty:
        raise ValueError("SurvFace official evaluation rows are missing")
    official = rebase_survface_protocol_subset_indexes(official)
    evaluation_protocol = build_survface_official_protocol(official)
    if not evaluation_protocol.known_unknown_probes.empty:
        raise ValueError("SurvFace official protocol must not contain known-unknown")

    return _characterize_population_with_protocols(
        prepared,
        population,
        development_matrix=development_matrix,
        calibration_protocol=calibration_protocol,
        evaluation_protocol=evaluation_protocol,
        protocol_uid=(
            "qmul-survface-v1-training-derived-"
            f"{int(calibration_gallery_identities)}-watchlist-calibration-v2"
        ),
        pca_dimensions=pca_dimensions,
        pq_settings=pq_settings,
        seed=seed,
        target_fpir=target_fpir,
        target_fpirs=target_fpirs,
        top_k=top_k,
        threshold_selection="non_mated_only",
        calibration_contract={
            "name": (
                f"training_{int(calibration_gallery_identities)}_"
                "half_gallery_v2"
            ),
            "seed": int(seed),
            "source_identity_count": int(
                population.loc[
                    population["protocol_role"].astype(str).eq("training"),
                    "identity_id",
                ].nunique()
            ),
            "gallery_identity_count": int(
                calibration_protocol.gallery["identity_id"].nunique()
            ),
            "gallery_source_image_count": int(len(calibration_protocol.gallery)),
            "gallery_enrollment_policy": "half_floor",
            "registered_probe_count": int(
                len(calibration_protocol.registered_probes)
            ),
            "non_mated_identity_count": int(
                calibration_protocol.known_unknown_probes[
                    "identity_id"
                ].nunique()
            ),
            "non_mated_probe_count": int(
                len(calibration_protocol.known_unknown_probes)
            ),
            "compressor_fit_source": "watchlist_enrollment_images_only",
            "compressor_fit_image_count": int(len(fit_rows)),
            "official_test_used_for_threshold": False,
        },
        progress=progress,
    )


def characterize_step2_rfw_custom_compression(
    prepared: PreparedPopulationInputs,
    selected_manifest: pd.DataFrame,
    *,
    pca_dimensions: Sequence[int],
    pq_settings: Sequence[tuple[int, int]],
    seed: int = 42,
    target_fpir: float = 0.10,
    target_fpirs: Sequence[float] | None = None,
    calibration_gallery_identities: int = 80,
    top_k: int = 20,
    progress: ProgressCallback | None = None,
) -> Step2CompressionResult:
    """Evaluate the identity-disjoint RFW-Custom 1:N open-set protocol.

    PCA/PQ fitting uses only ``development_pool`` rows. Threshold calibration
    uses only ``calibration_pool`` rows, while the persisted custom gallery and
    probe roles define the test protocol without resampling. RFW Official
    pairs/folds are never consumed by this path. The EdgeFace/RFW training
    identity-overlap status remains ``UNKNOWN`` and this result is therefore
    checkpoint-level same-dataset evidence, not strict unseen-identity proof.
    """

    population = _population_frame(prepared, selected_manifest)
    evaluation_protocol = adapt_rfw_custom_manifest_to_open_set_protocol(
        population
    )
    protocol_uids = set(population["protocol_uid"].astype(str))
    if len(protocol_uids) != 1:
        raise ValueError("RFW custom population must contain one protocol_uid")
    protocol_uid = next(iter(protocol_uids))

    development = population.loc[
        population["protocol_role"].astype(str).eq("development_pool")
    ]
    if development.empty or set(development["split"].astype(str)) != {
        "development"
    }:
        raise ValueError("RFW custom development_pool is required")
    development_matrix = np.stack(development["origin_embedding"]).astype(
        np.float32
    )

    calibration = population.loc[
        population["protocol_role"].astype(str).eq("calibration_pool")
    ]
    if calibration.empty or set(calibration["split"].astype(str)) != {
        "calibration"
    }:
        raise ValueError("RFW custom calibration_pool is required")
    calibration_sizes = calibration.groupby("identity_id")["image_id"].nunique()
    eligible_calibration = int((calibration_sizes > 1).sum())
    requested_gallery_count = int(calibration_gallery_identities)
    if requested_gallery_count < 1 or requested_gallery_count >= eligible_calibration:
        raise ValueError(
            "RFW custom calibration_gallery_identities must be positive and "
            "reserve at least one non-mated identity: "
            f"requested={requested_gallery_count}, eligible={eligible_calibration}"
        )
    calibration_protocol = build_calibration_protocol(
        population,
        split_name="calibration",
        gallery_identity_count=requested_gallery_count,
        enrollment_count=1,
        seed=seed,
    )
    group_counts = {
        str(group): int(count)
        for group, count in population.groupby("rfw_group")["identity_id"]
        .nunique()
        .to_dict()
        .items()
    }
    return _characterize_population_with_protocols(
        prepared,
        population,
        development_matrix=development_matrix,
        calibration_protocol=calibration_protocol,
        evaluation_protocol=evaluation_protocol,
        protocol_uid=protocol_uid,
        pca_dimensions=pca_dimensions,
        pq_settings=pq_settings,
        seed=seed,
        target_fpir=target_fpir,
        target_fpirs=target_fpirs,
        top_k=top_k,
        threshold_selection="non_mated_only",
        calibration_contract={
            "name": "rfw_custom_identity_disjoint_calibration_v1",
            "seed": int(seed),
            "gallery_identity_count": requested_gallery_count,
            "enrollment_count": 1,
            "compressor_fit_source": "development_pool_only",
            "compressor_fit_image_count": int(len(development)),
            "threshold_source": "calibration_pool_non_mated_only",
            "official_pairs_or_folds_used": False,
            "checkpoint_overlap_status": "UNKNOWN",
            "strict_unseen_identity_evidence": False,
            "evaluation_identity_counts_by_group": group_counts,
        },
        progress=progress,
    )
