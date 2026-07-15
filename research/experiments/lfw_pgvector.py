from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from research.calibration.rejection import GlobalThresholdCalibrator
from research.compression import (
    ORIGIN_512,
    PCA_256,
    PCA_PROFILE_DIMENSIONS,
    pca_profile_dimension,
)
from research.database.connection import ensure_vector_indexes, session_scope
from research.database.repository import VectorRepository
from research.experiments.lfw_certification import LFWCertificationInputs
from research.protocols.open_set import OpenSetProtocol
from research.search.open_set import (
    build_certified_search_features,
    summarize_certified_search_features,
)


ProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class LFWTemplateScope:
    run_uid: str
    protocol_name: str
    model_uid: str
    aggregation_method: str = "mean"
    enrollment_policy: str = "fixed"
    enrollment_target: int = 1

    def repository_kwargs(self) -> dict[str, object]:
        return asdict(self)


def protocol_frames(protocol: OpenSetProtocol) -> dict[str, pd.DataFrame]:
    return {
        "gallery": protocol.gallery,
        "registered_probes": protocol.registered_probes,
        "known_unknown_probes": protocol.known_unknown_probes,
        "unknown_unknown_probes": protocol.unknown_unknown_probes,
    }


def _emit(progress: ProgressCallback | None, message: str, **details: object) -> None:
    if progress is not None:
        progress(message, details)


def materialize_lfw_templates(
    engine: Engine,
    *,
    bundle: LFWCertificationInputs,
    scope: LFWTemplateScope,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    required = {
        "identity_id",
        "retrieval_embedding",
        "embedding",
        "fallback_embedding",
        "source_image_ids",
        "enrollment_count",
        "quality",
        "variance",
        "angular_error",
        "reconstruction_error_norm",
    }
    missing = required.difference(bundle.templates.columns)
    if missing:
        raise ValueError(f"template bundle is missing columns: {sorted(missing)}")

    compression_profile = str(bundle.coverage.get("compression_profile", PCA_256))
    if compression_profile not in PCA_PROFILE_DIMENSIONS:
        raise ValueError("LFW compressed template materialization requires a PCA profile")
    pca_dimension = pca_profile_dimension(compression_profile)
    actions = {
        "template_512_inserted": 0,
        "template_512_skipped": 0,
        f"template_{pca_dimension}_inserted": 0,
        f"template_{pca_dimension}_skipped": 0,
    }
    scope_kwargs = scope.repository_kwargs()
    with session_scope(engine) as session:
        repository = VectorRepository(session)
        for index, row in enumerate(bundle.templates.itertuples(index=False), start=1):
            common = {
                **scope_kwargs,
                "enrollment_count": int(row.enrollment_count),
                "identity_id": str(row.identity_id),
                "source_image_ids": list(row.source_image_ids),
                "quality": float(row.quality),
                "variance": float(row.variance),
                "parameters": {
                    "source_run_uid": scope.run_uid,
                    "certificate_space": bundle.certificate_space,
                    "compression_profile": compression_profile,
                },
            }
            _, action_512 = repository.upsert_template_512(
                **common,
                vector_type=ORIGIN_512,
                embedding=np.asarray(row.fallback_embedding, dtype=np.float32).tolist(),
                angular_error=0.0,
                reconstruction_error_norm=0.0,
            )
            _, action_pca = repository.upsert_pca_template(
                pca_dimension,
                **common,
                vector_type=compression_profile,
                embedding=np.asarray(row.retrieval_embedding, dtype=np.float32).tolist(),
                angular_error=float(row.angular_error),
                reconstruction_error_norm=float(row.reconstruction_error_norm),
            )
            actions[f"template_512_{action_512}"] += 1
            actions[f"template_{pca_dimension}_{action_pca}"] += 1
            if index % 100 == 0:
                _emit(progress, "identity templates materialized", processed=index)

    ensure_vector_indexes(engine)
    _emit(
        progress,
        "identity template materialization completed",
        templates=len(bundle.templates),
        protocol_name=scope.protocol_name,
    )
    return {
        "scope": asdict(scope),
        "template_count": int(len(bundle.templates)),
        "compression_profile": compression_profile,
        "dimension": pca_dimension,
        "actions": actions,
    }


def _exact_feature_frame(
    engine: Engine,
    *,
    probes: pd.DataFrame,
    scope: LFWTemplateScope,
    compression_profile: str,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    with session_scope(engine) as session:
        repository = VectorRepository(session)
        for index, probe in enumerate(probes.itertuples(index=False), start=1):
            if compression_profile in PCA_PROFILE_DIMENSIONS:
                ranked = repository.find_similar_pca_templates(
                    pca_profile_dimension(compression_profile),
                    probe.retrieval_embedding,
                    **scope.repository_kwargs(),
                    vector_type=compression_profile,
                    top_k=2,
                    search_mode="exact",
                )
            elif compression_profile == ORIGIN_512:
                ranked = repository.find_similar_templates_512(
                    probe.fallback_embedding,
                    **scope.repository_kwargs(),
                    vector_type=ORIGIN_512,
                    top_k=2,
                    search_mode="exact",
                )
            else:
                raise ValueError(
                    "compression_profile must be origin_512 or a supported PCA profile"
                )
            if len(ranked) < 2:
                raise RuntimeError("calibration requires at least two gallery templates")
            probe_type = str(probe.probe_type)
            query_identity = str(probe.identity_id)
            top1_identity = str(ranked[0]["identity_id"])
            top1_score = float(ranked[0]["similarity"])
            top2_score = float(ranked[1]["similarity"])
            records.append(
                {
                    "query_id": str(probe.image_id),
                    "query_identity_id": query_identity,
                    "probe_type": probe_type,
                    "compression_profile": compression_profile,
                    "top1_identity": top1_identity,
                    "top1_score": top1_score,
                    "top2_score": top2_score,
                    "score_margin": top1_score - top2_score,
                    "probe_quality": float(probe.quality),
                    "template_quality": float(ranked[0]["quality"] or 0.0),
                    "template_variance": float(ranked[0]["variance"] or 0.0),
                    "enrollment_count": int(ranked[0]["enrollment_count"]),
                    "reconstruction_error_norm": float(
                        probe.reconstruction_error_norm
                    ),
                    "is_mated": int(probe_type == "registered"),
                    "top1_correct": int(
                        probe_type == "registered" and top1_identity == query_identity
                    ),
                    "y_true_accept": int(
                        probe_type == "registered" and top1_identity == query_identity
                    ),
                    "exact_latency_ms": float(ranked[0]["query_elapsed_ms"]),
                }
            )
            if index % 100 == 0:
                _emit(
                    progress,
                    "calibration pgvector exact search",
                    processed=index,
                    total=len(probes),
                    compression_profile=compression_profile,
                )
    return pd.DataFrame.from_records(records)


def calibrate_lfw_pgvector_threshold(
    engine: Engine,
    *,
    probes: pd.DataFrame,
    scope: LFWTemplateScope,
    target_fpir: float,
    compression_profile: str = PCA_256,
    progress: ProgressCallback | None = None,
) -> tuple[float, pd.DataFrame, dict[str, object]]:
    features = _exact_feature_frame(
        engine,
        probes=probes,
        scope=scope,
        compression_profile=compression_profile,
        progress=progress,
    )
    calibrator = GlobalThresholdCalibrator(target_fpir=target_fpir).fit(features)
    threshold = float(calibrator.threshold)
    features["accepted"] = calibrator.predict(features).astype(bool)
    mated = features["is_mated"].astype(bool)
    non_mated = ~mated
    achieved_fpir = (
        float(features.loc[non_mated, "accepted"].mean()) if non_mated.any() else 0.0
    )
    achieved_dir = (
        float(
            (
                features.loc[mated, "accepted"]
                & features.loc[mated, "top1_correct"].astype(bool)
            ).mean()
        )
        if mated.any()
        else 0.0
    )
    summary = {
        "model": "global_threshold",
        "compression_profile": compression_profile,
        "target_fpir": float(target_fpir),
        "threshold": threshold,
        "calibration_rows": int(len(features)),
        "mated_rows": int(mated.sum()),
        "non_mated_rows": int(non_mated.sum()),
        "achieved_fpir": achieved_fpir,
        "achieved_dir_rank1": achieved_dir,
        "exact_latency_ms_p50": float(features["exact_latency_ms"].median()),
        "exact_latency_ms_p95": float(
            features["exact_latency_ms"].quantile(0.95)
        ),
    }
    return threshold, features, summary


def _identity_rows(templates: pd.DataFrame) -> dict[str, dict[str, object]]:
    if templates["identity_id"].astype(str).duplicated().any():
        raise ValueError("template identities must be unique within a protocol scope")
    return {
        str(row["identity_id"]): row.to_dict()
        for _, row in templates.reset_index(drop=True).iterrows()
    }


def run_lfw_pgvector_search(
    engine: Engine,
    *,
    probes: pd.DataFrame,
    templates: pd.DataFrame,
    scope: LFWTemplateScope,
    origin_threshold: float,
    pca_threshold: float,
    candidate_k: int,
    ef_search: int,
    compression_profile: str = PCA_256,
    progress: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if candidate_k < 2:
        raise ValueError("candidate_k must be at least 2")
    gallery_size = int(len(templates))
    if candidate_k > gallery_size:
        raise ValueError("candidate_k must not exceed gallery size")
    if compression_profile not in PCA_PROFILE_DIMENSIONS:
        raise ValueError("compression_profile must be a supported PCA profile")
    pca_dimension = pca_profile_dimension(compression_profile)
    mean_template_angular_error = float(
        pd.to_numeric(templates["angular_error"], errors="raise").mean()
    )
    max_template_angular_error = float(
        pd.to_numeric(templates["angular_error"], errors="raise").max()
    )
    by_identity = _identity_rows(templates)
    records: list[dict[str, object]] = []

    with session_scope(engine) as session:
        repository = VectorRepository(session)
        for index, probe in enumerate(probes.itertuples(index=False), start=1):
            origin_exact = repository.find_similar_templates_512(
                probe.fallback_embedding,
                **scope.repository_kwargs(),
                vector_type=ORIGIN_512,
                top_k=candidate_k,
                search_mode="exact",
            )
            pca_exact = repository.find_similar_pca_templates(
                pca_dimension,
                probe.retrieval_embedding,
                **scope.repository_kwargs(),
                vector_type=compression_profile,
                top_k=candidate_k,
                search_mode="exact",
            )
            pca_hnsw = repository.find_similar_pca_templates(
                pca_dimension,
                probe.retrieval_embedding,
                **scope.repository_kwargs(),
                vector_type=compression_profile,
                top_k=candidate_k,
                search_mode="hnsw",
                ef_search=ef_search,
            )
            if min(len(origin_exact), len(pca_exact), len(pca_hnsw)) < 2:
                raise RuntimeError("pgvector search returned fewer than two candidates")

            hnsw_ids = [str(row["identity_id"]) for row in pca_hnsw]
            missing = [identity for identity in hnsw_ids if identity not in by_identity]
            if missing:
                raise RuntimeError(f"DB candidate is missing from frozen templates: {missing}")
            candidate_templates = pd.DataFrame(
                [by_identity[identity] for identity in hnsw_ids]
            ).drop(columns=["fallback_embedding"], errors="ignore")
            certificate_probe = pd.DataFrame.from_records(
                [
                    {
                        "image_id": str(probe.image_id),
                        "identity_id": str(probe.identity_id),
                        "probe_type": str(probe.probe_type),
                        "embedding": np.asarray(
                            probe.fallback_embedding, dtype=np.float32
                        ),
                        "quality": float(probe.quality),
                        "angular_error": 0.0,
                        "reconstruction_error_norm": float(
                            probe.reconstruction_error_norm
                        ),
                    }
                ]
            )
            certificate_started = perf_counter()
            certified = build_certified_search_features(
                certificate_probe,
                candidate_templates,
                compression_profile=compression_profile,
                threshold=origin_threshold,
                top_k=min(candidate_k, len(candidate_templates)),
                candidate_scope="candidate_set",
                gallery_size=gallery_size,
            ).iloc[0].to_dict()
            certificate_elapsed_ms = (perf_counter() - certificate_started) * 1000.0

            origin_top1 = origin_exact[0]
            pca_exact_top1 = pca_exact[0]
            pca_hnsw_top1 = pca_hnsw[0]
            origin_accept = float(origin_top1["similarity"]) >= origin_threshold
            pca_accept = float(pca_exact_top1["similarity"]) >= pca_threshold
            origin_decision = "accept" if origin_accept else "reject"
            origin_identity = str(origin_top1["identity_id"]) if origin_accept else None
            defer = certified["certified_decision"] == "defer"
            if defer:
                final_decision = origin_decision
                final_identity = origin_identity
                final_source = "origin_512_db_exact_fallback"
            else:
                final_decision = str(certified["certified_decision"])
                final_identity = certified["certified_identity"]
                final_source = "candidate_certificate"

            pca_exact_ids = [str(row["identity_id"]) for row in pca_exact]
            true_identity = str(probe.identity_id)
            registered = str(probe.probe_type) == "registered"
            non_mated = not registered
            origin_top1_correct = registered and str(origin_top1["identity_id"]) == true_identity
            pca_top1_correct = registered and str(pca_exact_top1["identity_id"]) == true_identity
            candidate_contains_origin_top1 = str(origin_top1["identity_id"]) in hnsw_ids
            compressed_candidates_contain_origin_top1 = (
                str(origin_top1["identity_id"]) in pca_exact_ids
            )
            candidate_contains_pca_top1 = str(pca_exact_top1["identity_id"]) in hnsw_ids
            fallback_latency = float(origin_top1["query_elapsed_ms"]) if defer else 0.0
            certified.update(
                {
                    "certified_query_angular_error": 0.0,
                    "fallback_used": bool(defer),
                    "fallback_query_source": "origin_512_db_exact" if defer else None,
                    "fallback_template_source": "template_embedding_512" if defer else None,
                    "fallback_decision": origin_decision if defer else None,
                    "fallback_identity": origin_identity if defer else None,
                    "fallback_top1_identity": str(origin_top1["identity_id"]) if defer else None,
                    "fallback_top1_score": float(origin_top1["similarity"]) if defer else np.nan,
                    "final_decision": final_decision,
                    "final_identity": final_identity,
                    "final_decision_source": final_source,
                    "compression_profile": compression_profile,
                    "compression_dimension": pca_dimension,
                    "mean_gallery_template_angular_error": mean_template_angular_error,
                    "max_gallery_template_angular_error": max_template_angular_error,
                    "origin_exact_top1_identity": str(origin_top1["identity_id"]),
                    "origin_exact_top1_score": float(origin_top1["similarity"]),
                    "pca_exact_top1_identity": str(pca_exact_top1["identity_id"]),
                    "pca_exact_top1_score": float(pca_exact_top1["similarity"]),
                    "pca_hnsw_top1_identity": str(pca_hnsw_top1["identity_id"]),
                    "pca_hnsw_top1_score": float(pca_hnsw_top1["similarity"]),
                    "pca_exact_ranked_identities": pca_exact_ids,
                    "pca_hnsw_ranked_identities": hnsw_ids,
                    "candidate_contains_origin_top1": candidate_contains_origin_top1,
                    "pca_exact_candidates_contain_origin_top1": (
                        compressed_candidates_contain_origin_top1
                    ),
                    "candidate_contains_pca_exact_top1": candidate_contains_pca_top1,
                    "candidate_contains_true_identity": true_identity in hnsw_ids if registered else None,
                    "hnsw_recall_at_k_vs_pca_exact": len(set(hnsw_ids).intersection(pca_exact_ids)) / len(pca_exact_ids),
                    "compressed_rank_inversion": str(pca_exact_top1["identity_id"]) != str(origin_top1["identity_id"]),
                    "hnsw_rank_inversion": str(pca_hnsw_top1["identity_id"]) != str(pca_exact_top1["identity_id"]),
                    "registered_compressed_rank_inversion": (
                        str(pca_exact_top1["identity_id"])
                        != str(origin_top1["identity_id"])
                        if registered
                        else None
                    ),
                    "registered_identity_loss": (
                        origin_top1_correct and not pca_top1_correct
                        if registered
                        else None
                    ),
                    "registered_identity_gain": (
                        not origin_top1_correct and pca_top1_correct
                        if registered
                        else None
                    ),
                    "non_mated_top1_change": (
                        str(pca_exact_top1["identity_id"])
                        != str(origin_top1["identity_id"])
                        if non_mated
                        else None
                    ),
                    "candidate_miss_caused_by_compression": (
                        not compressed_candidates_contain_origin_top1
                    ),
                    "candidate_miss_caused_by_hnsw": not candidate_contains_pca_top1,
                    "origin_calibrated_threshold": float(origin_threshold),
                    "pca_calibrated_threshold": float(pca_threshold),
                    "pca_calibrated_decision": "accept" if pca_accept else "reject",
                    "origin_calibrated_decision": origin_decision,
                    "threshold_crossing": pca_accept != origin_accept,
                    "candidate_k": candidate_k,
                    "hnsw_ef_search": ef_search,
                    "origin_exact_baseline_latency_ms": float(origin_top1["query_elapsed_ms"]),
                    "pca_exact_baseline_latency_ms": float(pca_exact_top1["query_elapsed_ms"]),
                    "pca_hnsw_latency_ms": float(pca_hnsw_top1["query_elapsed_ms"]),
                    "certificate_latency_ms": float(certificate_elapsed_ms),
                    "fallback_latency_ms": fallback_latency,
                    "simulated_system_latency_ms": float(pca_hnsw_top1["query_elapsed_ms"]) + certificate_elapsed_ms + fallback_latency,
                    "final_matches_origin_exact": final_decision == origin_decision and (final_decision == "reject" or str(final_identity) == str(origin_identity)),
                    "certified_accept_correct": (
                        origin_decision == "accept"
                        and str(certified["certified_identity"]) == str(origin_identity)
                        if certified["certified_decision"] == "accept"
                        else None
                    ),
                    "certified_reject_correct": (
                        origin_decision == "reject"
                        if certified["certified_decision"] == "reject"
                        else None
                    ),
                }
            )
            records.append(certified)
            if index % 100 == 0:
                _emit(
                    progress,
                    "LFW pgvector candidate search",
                    processed=index,
                    total=len(probes),
                )

    features = pd.DataFrame.from_records(records)
    summary = summarize_lfw_pgvector_search(features)
    return features, summary


def summarize_lfw_pgvector_search(features: pd.DataFrame) -> dict[str, object]:
    if features.empty:
        raise ValueError("search features must not be empty")

    def rate(column: str, frame: pd.DataFrame = features) -> float:
        values = frame[column].dropna().astype(bool)
        return float(values.mean()) if len(values) else 0.0

    def latency(column: str) -> dict[str, float]:
        values = pd.to_numeric(features[column], errors="raise")
        return {
            "p50": float(values.quantile(0.50)),
            "p95": float(values.quantile(0.95)),
            "mean": float(values.mean()),
        }

    def optional_rate(column: str, frame: pd.DataFrame) -> float | None:
        if column not in frame.columns:
            return None
        values = frame[column].dropna().astype(bool)
        return float(values.mean()) if len(values) else None

    def correctness(column: str, frame: pd.DataFrame) -> dict[str, float | int | None]:
        if column not in frame.columns:
            return {"count": 0, "correct": 0, "rate": None}
        values = frame[column].dropna().astype(bool)
        return {
            "count": int(len(values)),
            "correct": int(values.sum()),
            "rate": float(values.mean()) if len(values) else None,
        }

    def open_set_metrics(
        frame: pd.DataFrame,
        *,
        decision_column: str,
        identity_column: str,
    ) -> dict[str, float | int]:
        registered_mask = frame["probe_type"].eq("registered")
        non_mated_mask = ~registered_mask
        accepted = frame[decision_column].eq("accept")
        top1_correct = (
            frame[identity_column].astype(str)
            == frame["query_identity_id"].astype(str)
        )
        dir_rank1 = (
            float((accepted[registered_mask] & top1_correct[registered_mask]).mean())
            if registered_mask.any()
            else 0.0
        )
        fpir = (
            float(accepted[non_mated_mask].mean()) if non_mated_mask.any() else 0.0
        )
        return {
            "mated_count": int(registered_mask.sum()),
            "non_mated_count": int(non_mated_mask.sum()),
            "dir_rank1": dir_rank1,
            "fnir_rank1": 1.0 - dir_rank1,
            "fpir": fpir,
        }

    def detailed_metrics(frame: pd.DataFrame) -> dict[str, object]:
        return {
            "rows": int(len(frame)),
            "registered_compressed_rank_inversion_rate": optional_rate(
                "registered_compressed_rank_inversion", frame
            ),
            "registered_identity_loss_rate": optional_rate(
                "registered_identity_loss", frame
            ),
            "registered_identity_gain_rate": optional_rate(
                "registered_identity_gain", frame
            ),
            "non_mated_top1_change_rate": optional_rate(
                "non_mated_top1_change", frame
            ),
            "candidate_miss_caused_by_compression_rate": optional_rate(
                "candidate_miss_caused_by_compression", frame
            ),
            "candidate_miss_caused_by_hnsw_rate": optional_rate(
                "candidate_miss_caused_by_hnsw", frame
            ),
            "certified_accept_correctness": correctness(
                "certified_accept_correct", frame
            ),
            "certified_reject_correctness": correctness(
                "certified_reject_correct", frame
            ),
            "exact_fallback_rate": optional_rate("fallback_used", frame),
        }

    registered = features.loc[features["probe_type"].eq("registered")]
    compression_profile = str(features["compression_profile"].iloc[0])
    compression_dimension = int(features["compression_dimension"].iloc[0])
    certification = summarize_certified_search_features(features)
    result = {
        "rows": int(len(features)),
        "compression_profile": compression_profile,
        "compression_dimension": compression_dimension,
        "mean_gallery_template_angular_error": float(
            features["mean_gallery_template_angular_error"].iloc[0]
        ),
        "max_gallery_template_angular_error": float(
            features["max_gallery_template_angular_error"].iloc[0]
        ),
        "candidate_k": int(features["candidate_k"].iloc[0]),
        "hnsw_ef_search": int(features["hnsw_ef_search"].iloc[0]),
        "candidate_contains_origin_top1_rate": rate("candidate_contains_origin_top1"),
        "pca_exact_candidates_contain_origin_top1_rate": rate(
            "pca_exact_candidates_contain_origin_top1"
        ),
        "candidate_contains_pca_exact_top1_rate": rate("candidate_contains_pca_exact_top1"),
        "candidate_contains_true_identity_rate_registered": rate(
            "candidate_contains_true_identity", registered
        ),
        "mean_hnsw_recall_at_k_vs_pca_exact": float(
            features["hnsw_recall_at_k_vs_pca_exact"].mean()
        ),
        "compressed_rank_inversion_rate": rate("compressed_rank_inversion"),
        "registered_compressed_rank_inversion_rate": optional_rate(
            "registered_compressed_rank_inversion", registered
        ),
        "registered_identity_loss_rate": optional_rate(
            "registered_identity_loss", registered
        ),
        "registered_identity_gain_rate": optional_rate(
            "registered_identity_gain", registered
        ),
        "non_mated_top1_change_rate": optional_rate(
            "non_mated_top1_change",
            features.loc[~features["probe_type"].eq("registered")],
        ),
        "candidate_miss_caused_by_compression_rate": rate(
            "candidate_miss_caused_by_compression"
        ),
        "candidate_miss_caused_by_hnsw_rate": rate(
            "candidate_miss_caused_by_hnsw"
        ),
        "hnsw_rank_inversion_rate": rate("hnsw_rank_inversion"),
        "threshold_crossing_rate": rate("threshold_crossing"),
        "final_matches_origin_exact_rate": rate("final_matches_origin_exact"),
        "latency_ms": {
            "origin_exact_baseline": latency("origin_exact_baseline_latency_ms"),
            "pca_exact_baseline": latency("pca_exact_baseline_latency_ms"),
            "pca_hnsw": latency("pca_hnsw_latency_ms"),
            "certificate": latency("certificate_latency_ms"),
            "simulated_system": latency("simulated_system_latency_ms"),
        },
        "storage": {
            "vector_payload_bytes_per_template": compression_dimension * 4,
            "gallery_vector_payload_bytes": (
                int(features["certification_gallery_size"].iloc[0])
                * compression_dimension
                * 4
            ),
        },
        "origin_open_set": open_set_metrics(
            features,
            decision_column="origin_calibrated_decision",
            identity_column="origin_exact_top1_identity",
        ),
        "compressed_open_set": open_set_metrics(
            features,
            decision_column="pca_calibrated_decision",
            identity_column="pca_exact_top1_identity",
        ),
        "final_open_set": open_set_metrics(
            features,
            decision_column="final_decision",
            identity_column="final_identity",
        ),
        "certified_accept_correctness": correctness(
            "certified_accept_correct", features
        ),
        "certified_reject_correctness": correctness(
            "certified_reject_correct", features
        ),
        "certification": certification,
        "by_probe_type": {
            str(probe_type): detailed_metrics(group)
            for probe_type, group in features.groupby("probe_type", sort=True)
        },
    }
    return result
