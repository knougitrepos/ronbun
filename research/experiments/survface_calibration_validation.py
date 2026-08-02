from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.calibration import (
    choose_effective_gallery_threshold,
    choose_sidak_gallery_threshold,
    estimate_effective_gallery_ratio,
)
from research.experiments.step2_compression import (
    _origin_protocol_comparison,
    _population_frame,
    _protocol_arrays,
    _stable_protocol_key,
    _wilson_interval_95,
)
from research.explainability.gradcam.extraction import PreparedPopulationInputs
from research.protocols import build_calibration_protocol


ProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class SurvFaceSidakHoldoutResult:
    fold_results: pd.DataFrame
    summary: pd.DataFrame


def _non_mated_pair_scores(
    arrays: dict[str, np.ndarray],
    *,
    query_batch_size: int = 512,
) -> np.ndarray:
    non_mated = arrays["query_probe_types"].astype(str) == "known_unknown"
    queries = np.asarray(arrays["queries"][non_mated], dtype=np.float32)
    gallery = np.asarray(arrays["gallery"], dtype=np.float32)
    if len(queries) == 0 or len(gallery) == 0:
        raise ValueError("Sidak calibration requires non-mated queries and gallery")
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    blocks: list[np.ndarray] = []
    for start in range(0, len(queries), query_batch_size):
        stop = min(start + query_batch_size, len(queries))
        blocks.append((queries[start:stop] @ gallery.T).reshape(-1))
    scores = np.concatenate(blocks).astype(np.float64, copy=False)
    if len(scores) != len(queries) * len(gallery):
        raise RuntimeError("impostor pair score count is inconsistent")
    return scores


def validate_survface_sidak_calibration_holdout(
    prepared: PreparedPopulationInputs,
    selected_manifest: pd.DataFrame,
    *,
    conditions: Sequence[tuple[int, int, int]] = (
        (50, 100, 1),
        (100, 200, 1),
        (50, 100, 5),
        (100, 200, 5),
        (50, 100, 20),
    ),
    target_fpir: float = 0.10,
    seed: int = 42,
    progress: ProgressCallback | None = None,
) -> SurvFaceSidakHoldoutResult:
    """Validate empirical Sidak calibration on identity-disjoint calibration folds."""

    population = _population_frame(prepared, selected_manifest)
    calibration = population.loc[population["split"].eq("calibration")].copy()
    identities = sorted(
        calibration["identity_id"].astype(str).unique().tolist(),
        key=lambda value: _stable_protocol_key(
            value,
            seed=seed,
            namespace="calibration:sidak-holdout-fold",
        ),
    )
    if len(identities) < 4:
        raise ValueError("Sidak holdout validation requires at least four identities")
    fold_ids = {
        "fold_a": set(identities[0::2]),
        "fold_b": set(identities[1::2]),
    }
    fold_populations = {
        name: calibration.loc[
            calibration["identity_id"].astype(str).isin(identity_set)
        ].copy()
        for name, identity_set in fold_ids.items()
    }

    records: list[dict[str, object]] = []
    for (
        fit_gallery_identity_count,
        validation_gallery_identity_count,
        enrollment_count,
    ) in conditions:
        fit_gallery_count = int(fit_gallery_identity_count)
        validation_gallery_count = int(validation_gallery_identity_count)
        enrollment = int(enrollment_count)
        if fit_gallery_count <= 0 or validation_gallery_count <= 0 or enrollment <= 0:
            raise ValueError("Sidak holdout condition counts must be positive")
        for fit_fold, validation_fold in (
            ("fold_a", "fold_b"),
            ("fold_b", "fold_a"),
        ):
            fit_population = fold_populations[fit_fold]
            validation_population = fold_populations[validation_fold]
            fit_protocol = build_calibration_protocol(
                fit_population,
                split_name="calibration",
                gallery_identity_count=fit_gallery_count,
                enrollment_count=enrollment,
                seed=seed,
            )
            validation_protocol = build_calibration_protocol(
                validation_population,
                split_name="calibration",
                gallery_identity_count=validation_gallery_count,
                enrollment_count=enrollment,
                seed=seed,
            )
            fit_arrays = _protocol_arrays(fit_protocol, fit_population)
            validation_arrays = _protocol_arrays(
                validation_protocol,
                validation_population,
            )
            pair_scores = _non_mated_pair_scores(fit_arrays)
            fit_comparison = _origin_protocol_comparison(
                fit_arrays,
                top_k=1,
                progress=progress,
                progress_message="SurvFace effective-gallery fit retrieval",
                progress_details={
                    "fit_fold": fit_fold,
                    "fit_gallery_identity_count": fit_gallery_count,
                    "validation_gallery_identity_count": validation_gallery_count,
                    "enrollment_count": enrollment,
                },
            )
            fit_non_mated = ~fit_comparison["is_mated"].to_numpy(dtype=bool)
            ratio_estimate = estimate_effective_gallery_ratio(
                pair_scores,
                fit_comparison.loc[fit_non_mated, "origin_top1_score"].to_numpy(
                    dtype=np.float64
                ),
                target_fpir=target_fpir,
                gallery_size=fit_gallery_count,
            )
            calibrated = choose_sidak_gallery_threshold(
                pair_scores,
                target_fpir=target_fpir,
                target_gallery_size=validation_gallery_count,
            )
            adjusted = choose_effective_gallery_threshold(
                pair_scores,
                target_fpir=target_fpir,
                target_gallery_size=validation_gallery_count,
                effective_gallery_ratio=ratio_estimate.effective_gallery_ratio,
            )
            comparison = _origin_protocol_comparison(
                validation_arrays,
                top_k=1,
                progress=progress,
                progress_message="SurvFace Sidak holdout retrieval",
                progress_details={
                    "fit_fold": fit_fold,
                    "validation_fold": validation_fold,
                    "fit_gallery_identity_count": fit_gallery_count,
                    "validation_gallery_identity_count": validation_gallery_count,
                    "enrollment_count": enrollment,
                },
            )
            non_mated = ~comparison["is_mated"].to_numpy(dtype=bool)
            scores = comparison["origin_top1_score"].to_numpy(dtype=np.float64)
            non_mated_count = int(non_mated.sum())
            false_accept_count = int(
                (non_mated & (scores >= calibrated.threshold)).sum()
            )
            realized_fpir = float(false_accept_count / non_mated_count)
            ci_low, ci_high = _wilson_interval_95(
                false_accept_count,
                non_mated_count,
            )
            adjusted_false_accept_count = int(
                (non_mated & (scores >= adjusted.threshold)).sum()
            )
            adjusted_fpir = float(adjusted_false_accept_count / non_mated_count)
            adjusted_ci_low, adjusted_ci_high = _wilson_interval_95(
                adjusted_false_accept_count,
                non_mated_count,
            )
            records.append(
                {
                    "fit_fold": fit_fold,
                    "validation_fold": validation_fold,
                    "fit_gallery_identity_count": fit_gallery_count,
                    "validation_gallery_identity_count": validation_gallery_count,
                    "enrollment_count": enrollment,
                    "target_fpir": float(target_fpir),
                    "target_pair_false_match_rate": (
                        calibrated.target_pair_false_match_rate
                    ),
                    "impostor_pair_count": calibrated.impostor_pair_count,
                    "threshold": calibrated.threshold,
                    "empirical_pair_false_match_rate": (
                        calibrated.empirical_pair_false_match_rate
                    ),
                    "validation_non_mated_count": non_mated_count,
                    "validation_false_accept_count": false_accept_count,
                    "validation_fpir": realized_fpir,
                    "validation_fpir_wilson95_low": ci_low,
                    "validation_fpir_wilson95_high": ci_high,
                    "absolute_target_error": abs(realized_fpir - target_fpir),
                    "effective_gallery_ratio": (
                        ratio_estimate.effective_gallery_ratio
                    ),
                    "effective_gallery_size": ratio_estimate.effective_gallery_size,
                    "fit_maximum_fpir": ratio_estimate.realized_maximum_fpir,
                    "adjusted_threshold": adjusted.threshold,
                    "adjusted_validation_false_accept_count": (
                        adjusted_false_accept_count
                    ),
                    "adjusted_validation_fpir": adjusted_fpir,
                    "adjusted_validation_fpir_wilson95_low": adjusted_ci_low,
                    "adjusted_validation_fpir_wilson95_high": adjusted_ci_high,
                    "adjusted_absolute_target_error": abs(
                        adjusted_fpir - target_fpir
                    ),
                }
            )
    fold_results = pd.DataFrame(records)
    summary = (
        fold_results.groupby(
            [
                "fit_gallery_identity_count",
                "validation_gallery_identity_count",
                "enrollment_count",
            ],
            sort=True,
            as_index=False,
        )
        .agg(
            fold_count=("validation_fold", "size"),
            mean_threshold=("threshold", "mean"),
            mean_validation_fpir=("validation_fpir", "mean"),
            maximum_validation_fpir=("validation_fpir", "max"),
            mean_absolute_target_error=("absolute_target_error", "mean"),
            mean_effective_gallery_ratio=("effective_gallery_ratio", "mean"),
            mean_adjusted_threshold=("adjusted_threshold", "mean"),
            mean_adjusted_validation_fpir=("adjusted_validation_fpir", "mean"),
            maximum_adjusted_validation_fpir=(
                "adjusted_validation_fpir",
                "max",
            ),
            mean_adjusted_absolute_target_error=(
                "adjusted_absolute_target_error",
                "mean",
            ),
        )
    )
    return SurvFaceSidakHoldoutResult(
        fold_results=fold_results,
        summary=summary,
    )
