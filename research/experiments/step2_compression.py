"""PyTorch Step 2 compression characterization for one frozen population."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.calibration.rejection import choose_threshold
from research.compression import (
    ORIGIN_512,
    PQCompressor,
    fit_pca_family,
    pq_profile_name,
)
from research.evaluation import (
    apply_retrieval_thresholds,
    compare_cosine_retrieval,
    paired_embedding_metrics,
)
from research.explainability.gradcam.extraction import PreparedPopulationInputs
from research.protocols import (
    OpenSetProtocol,
    build_calibration_protocol,
    build_open_set_protocol,
    build_survface_official_protocol,
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
    summary: pd.DataFrame


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
    selected = selected.rename(columns={"sample_id": "image_id"})
    selected["origin_embedding"] = list(embeddings)
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
    lookup = population[["image_id", "origin_embedding"]]

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
        "queries": np.stack(queries["origin_embedding"]).astype(np.float32),
        "gallery_ids": templates["identity_id"].astype(str).to_numpy(),
        "gallery_identity_ids": templates["identity_id"].astype(str).to_numpy(),
        "gallery": np.stack(templates["embedding"]).astype(np.float32),
    }


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
) -> float:
    value = float(
        choose_threshold(
            comparison[score_column].to_numpy(dtype=float),
            comparison["is_mated"].to_numpy(dtype=bool),
            comparison[correct_column].to_numpy(dtype=bool),
            target_fpir,
        )
    )
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
    else:
        storage_bytes = int(profile_result.metadata["code_bytes"])
        codebook_bytes = int(profile_result.metadata["codebook_bytes"])
        codebook_source = str(profile_result.metadata["codebook_bytes_source"])
    if storage_bytes <= 0 or codebook_bytes < 0:
        raise ValueError("compression storage metadata must be non-negative")
    return {
        "storage_bytes_per_embedding": storage_bytes,
        "codebook_bytes": codebook_bytes,
        "codebook_bytes_source": codebook_source,
    }


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
        )
        .reset_index()
    )
    records = []
    for keys, group in retrieval.groupby(
        [
            "compression_family",
            "compression_profile",
            "threshold_policy",
        ],
        sort=True,
    ):
        family, profile, policy = keys
        mated = group["is_mated"].astype(bool)
        accepted = group["compressed_accepted"].astype(bool)
        correct = group["compressed_rank1_correct"].astype(bool)
        records.append(
            {
                "compression_family": family,
                "compression_profile": profile,
                "threshold_policy": policy,
                "query_count": int(len(group)),
                "dir_rank1": (
                    float((accepted & correct & mated).sum() / mated.sum())
                    if mated.any()
                    else np.nan
                ),
                "fpir": (
                    float((accepted & ~mated).sum() / (~mated).sum())
                    if (~mated).any()
                    else np.nan
                ),
                "agreement_with_origin": float(
                    group["agreement_with_origin"].mean()
                ),
                "threshold_crossing_rate": float(
                    group["threshold_crossing"].mean()
                ),
            }
        )
    return pd.DataFrame.from_records(records).merge(
        paired_summary,
        on=["compression_family", "compression_profile"],
        how="left",
        validate="many_to_one",
    )


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
    top_k: int,
    progress: ProgressCallback | None = None,
) -> Step2CompressionResult:
    requested_pca = tuple(int(value) for value in pca_dimensions)
    requested_pq = tuple((int(m), int(bits)) for m, bits in pq_settings)
    if not requested_pca or not requested_pq:
        raise ValueError("both PCA and PQ profile sets are required")
    development_values = np.asarray(development_matrix, dtype=np.float32)
    if development_values.ndim != 2 or development_values.shape[1] != 512:
        raise ValueError("development_matrix must have shape [N, 512]")

    pca_models = fit_pca_family(
        development_values,
        dimensions=requested_pca,
        random_state=seed,
    )
    compressors: list[tuple[str, str, Any]] = [
        ("pca", profile, compressor)
        for profile, compressor in pca_models.items()
    ]
    compressor_total = len(requested_pca) + len(requested_pq)
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
    work_per_profile = 2 * (
        len(calibration["queries"]) + len(evaluation["queries"])
    )
    retrieval_work_total = len(compressors) * work_per_profile
    full_matrix = np.stack(population["origin_embedding"]).astype(np.float32)
    paired_frames = []
    retrieval_frames = []
    for profile_index, (family, profile, compressor) in enumerate(compressors):
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
            top_k=min(int(top_k), len(calibration["gallery"])),
            progress=progress,
            progress_message=(
                f"{prepared.dataset_id} compression retrieval"
            ),
            progress_offset=profile_index * work_per_profile,
            progress_total=retrieval_work_total,
            progress_details={
                "profile": profile,
                "split": "calibration",
            },
        )
        origin_threshold = _threshold(
            calibration_comparison,
            score_column="origin_top1_score",
            correct_column="origin_rank1_correct",
            target_fpir=target_fpir,
        )
        compressed_threshold = _threshold(
            calibration_comparison,
            score_column="compressed_top1_score",
            correct_column="compressed_rank1_correct",
            target_fpir=target_fpir,
        )
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
            top_k=min(int(top_k), len(evaluation["gallery"])),
            progress=progress,
            progress_message=(
                f"{prepared.dataset_id} compression retrieval"
            ),
            progress_offset=(
                profile_index * work_per_profile
                + 2 * len(calibration["queries"])
            ),
            progress_total=retrieval_work_total,
            progress_details={
                "profile": profile,
                "split": "evaluation",
            },
        )
        for threshold_policy, operating_threshold in (
            ("frozen_origin", origin_threshold),
            ("recalibrated_compressed", compressed_threshold),
        ):
            compared = apply_retrieval_thresholds(
                evaluation_comparison,
                origin_threshold=origin_threshold,
                compressed_threshold=operating_threshold,
            )
            compared.insert(0, "dataset", prepared.dataset_id)
            compared.insert(1, "model_uid", prepared.model_uid)
            compared["protocol_uid"] = str(protocol_uid)
            compared["threshold_policy"] = threshold_policy
            compared["threshold_source_split"] = "calibration"
            compared["evaluation_split"] = "test"
            for column, value in storage.items():
                compared[column] = value
            retrieval_frames.append(compared)

    paired_metrics = pd.concat(paired_frames, ignore_index=True)
    retrieval_metrics = pd.concat(retrieval_frames, ignore_index=True)
    if paired_metrics["origin_fallback_used"].astype(bool).any():
        raise RuntimeError("paired metrics contain origin fallback rows")
    if retrieval_metrics["origin_fallback_used"].astype(bool).any():
        raise RuntimeError("retrieval metrics contain origin fallback rows")
    return Step2CompressionResult(
        paired_metrics=paired_metrics,
        retrieval_metrics=retrieval_metrics,
        summary=_summarize(paired_metrics, retrieval_metrics),
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
    calibration_gallery_identities: int = 200,
    top_k: int = 20,
    progress: ProgressCallback | None = None,
) -> Step2CompressionResult:
    """Fit on SurvFace training development and evaluate the official protocol.

    The selected population must contain the identity-disjoint training
    development/calibration rows and the official gallery, registered-probe,
    and unknown-unknown rows. Official protocol order is preserved by
    ``build_survface_official_protocol``; the official test set is never used
    to fit PCA/PQ or thresholds.
    """

    population = _population_frame(prepared, selected_manifest)
    development = population.loc[population["split"].eq("development")]
    if development.empty:
        raise ValueError("SurvFace development split is required to fit compressors")
    development_matrix = np.stack(development["origin_embedding"]).astype(
        np.float32
    )

    calibration_sizes = (
        population.loc[population["split"].eq("calibration")]
        .groupby("identity_id")["image_id"]
        .nunique()
    )
    eligible_calibration = int((calibration_sizes > 1).sum())
    if eligible_calibration < 2:
        raise ValueError(
            "SurvFace calibration requires at least two identities with "
            "multiple samples"
        )
    calibration_gallery_count = min(
        int(calibration_gallery_identities),
        eligible_calibration - 1,
    )
    calibration_protocol = build_calibration_protocol(
        population,
        split_name="calibration",
        gallery_identity_count=calibration_gallery_count,
        enrollment_count=1,
        seed=seed,
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
        protocol_uid="qmul-survface-v1-official-open-set-identification",
        pca_dimensions=pca_dimensions,
        pq_settings=pq_settings,
        seed=seed,
        target_fpir=target_fpir,
        top_k=top_k,
        progress=progress,
    )
