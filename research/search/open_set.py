from __future__ import annotations

import numpy as np
import pandas as pd

from research.search.certification import certify_open_set_decision, exact_open_set_decision

CALIBRATION_FEATURE_COLUMNS = [
    "top1_score",
    "score_margin",
    "probe_quality",
    "template_quality",
    "template_variance",
    "enrollment_count",
    "reconstruction_error_norm",
]

CERTIFICATION_CANDIDATE_SCOPES = {"exhaustive", "candidate_set"}
EXACT_FALLBACK_SOURCES = {
    "exact_fallback",
    "origin_512_db_exact_fallback",
}


def _l2(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _cosine_scores(query: np.ndarray, gallery: list[np.ndarray]) -> np.ndarray:
    q = _l2(query)
    matrix = np.stack([_l2(value) for value in gallery])
    return matrix @ q


def _is_vector_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.size > 0 and np.all(np.isfinite(array)))


def _positive_integer_value(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(number) or number <= 0.0 or number % 1 != 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def build_search_features(
    probes: pd.DataFrame,
    templates: pd.DataFrame,
    *,
    compression_profile: str,
    top_k: int = 2,
) -> pd.DataFrame:
    if top_k < 2:
        raise ValueError("top_k must be at least 2 to compute score margin")
    required_probe = {"image_id", "identity_id", "probe_type", "embedding"}
    required_template = {"identity_id", "embedding", "quality", "variance", "enrollment_count"}
    missing_probe = required_probe.difference(probes.columns)
    missing_template = required_template.difference(templates.columns)
    if missing_probe or missing_template:
        raise ValueError(f"missing columns: probes={sorted(missing_probe)}, templates={sorted(missing_template)}")

    template_rows = templates.reset_index(drop=True)
    template_embeddings = template_rows["embedding"].tolist()
    records = []
    for _, probe in probes.reset_index(drop=True).iterrows():
        scores = _cosine_scores(probe["embedding"], template_embeddings)
        order = np.argsort(scores)[::-1][:top_k]
        top1_idx = int(order[0])
        top2_idx = int(order[1]) if len(order) > 1 else top1_idx
        top1 = template_rows.iloc[top1_idx]
        top2_score = float(scores[top2_idx]) if top2_idx != top1_idx else float("nan")
        top1_score = float(scores[top1_idx])
        probe_type = str(probe["probe_type"])
        query_identity = str(probe["identity_id"])
        top1_identity = str(top1["identity_id"])
        records.append(
            {
                "query_id": probe["image_id"],
                "query_identity_id": query_identity,
                "probe_type": probe_type,
                "compression_profile": compression_profile,
                "top1_identity": top1_identity,
                "top1_score": top1_score,
                "top2_score": top2_score,
                "score_margin": top1_score - top2_score,
                "probe_quality": float(probe.get("quality", 0.0)),
                "template_quality": float(top1["quality"]),
                "template_variance": float(top1["variance"]),
                "enrollment_count": int(top1["enrollment_count"]),
                "reconstruction_error_norm": float(probe.get("reconstruction_error_norm", 0.0)),
                "ranked_identities": template_rows.iloc[order]["identity_id"].astype(str).tolist(),
                "ranked_scores": [float(scores[idx]) for idx in order],
                "is_mated": int(probe_type == "registered"),
                "top1_correct": int(
                    probe_type == "registered" and top1_identity == query_identity
                ),
                "y_true_accept": int(
                    probe_type == "registered" and top1_identity == query_identity
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def build_certified_search_features(
    probes: pd.DataFrame,
    templates: pd.DataFrame,
    *,
    compression_profile: str,
    threshold: float,
    top_k: int = 2,
    candidate_scope: str = "exhaustive",
    gallery_size: int | None = None,
) -> pd.DataFrame:
    required_template = {"identity_id", "embedding", "angular_error"}
    missing_template = required_template.difference(templates.columns)
    if missing_template:
        raise ValueError(f"missing columns: templates={sorted(missing_template)}")
    if candidate_scope not in CERTIFICATION_CANDIDATE_SCOPES:
        raise ValueError(
            f"candidate_scope must be one of {sorted(CERTIFICATION_CANDIDATE_SCOPES)}"
        )
    if candidate_scope == "candidate_set" and gallery_size is None:
        raise ValueError("gallery_size is required when candidate_scope is candidate_set")

    features = build_search_features(
        probes,
        templates,
        compression_profile=compression_profile,
        top_k=top_k,
    )
    template_rows = templates.reset_index(drop=True)
    candidate_count = int(len(template_rows))
    gallery_size_value = (
        _positive_integer_value(gallery_size, name="gallery_size")
        if gallery_size is not None
        else candidate_count
    )
    if gallery_size_value < candidate_count:
        raise ValueError("gallery_size must be at least the number of supplied candidates")
    if candidate_scope == "exhaustive" and gallery_size_value != candidate_count:
        raise ValueError(
            "gallery_size must equal the supplied candidate count when "
            "candidate_scope is exhaustive"
        )
    global_claim = candidate_scope == "exhaustive" and gallery_size_value == candidate_count
    template_embeddings = np.stack(template_rows["embedding"].tolist())
    template_identities = template_rows["identity_id"].astype(str).tolist()
    template_errors = template_rows["angular_error"].astype(float).to_numpy()
    fallback_template_embeddings = None
    if "fallback_embedding" in template_rows.columns:
        fallback_template_embeddings = np.stack(template_rows["fallback_embedding"].tolist())

    records = []
    for _, probe in probes.reset_index(drop=True).iterrows():
        query_angular_error = float(probe.get("angular_error", 0.0))
        decision = certify_open_set_decision(
            query=probe["embedding"],
            compressed_templates=template_embeddings,
            template_identities=template_identities,
            template_angular_errors=template_errors,
            threshold=threshold,
            query_angular_error=query_angular_error,
        )
        top1_index = int(np.argmax(decision.bounds.approximate_scores))
        top1_template_angular_error = float(template_errors[top1_index])
        top1_total_angular_error = min(
            query_angular_error + top1_template_angular_error,
            float(np.pi),
        )
        top1_approximate_angle = float(decision.bounds.approximate_angles[top1_index])
        top1_lower_bound = float(decision.bounds.lower_bounds[top1_index])
        top1_upper_bound = float(decision.bounds.upper_bounds[top1_index])
        top1_bound_width = top1_upper_bound - top1_lower_bound
        other_upper_bounds = np.delete(decision.bounds.upper_bounds, top1_index)
        max_other_upper_bound = (
            float(np.max(other_upper_bounds)) if len(other_upper_bounds) > 0 else float("nan")
        )
        max_upper_bound = float(np.max(decision.bounds.upper_bounds))
        top1_threshold_margin = top1_lower_bound - float(threshold)
        rank_margin = (
            top1_lower_bound - max_other_upper_bound
            if np.isfinite(max_other_upper_bound)
            else float("nan")
        )
        reject_margin = float(threshold) - max_upper_bound
        if decision.decision == "accept":
            margin_candidates = [top1_threshold_margin]
            if np.isfinite(rank_margin):
                margin_candidates.append(rank_margin)
            decision_margin = min(margin_candidates)
        elif decision.decision == "reject":
            decision_margin = reject_margin
        else:
            decision_margin = float("nan")
        fallback_used = False
        fallback_decision = None
        fallback_identity = None
        fallback_top1_identity = None
        fallback_top1_score = float("nan")
        fallback_query_source = None
        fallback_template_source = None
        final_decision = decision.decision
        final_identity = decision.selected_identity if decision.decision == "accept" else None
        final_decision_source = "certified_bound" if decision.decision != "defer" else "defer_unresolved"
        if decision.decision == "defer" and fallback_template_embeddings is not None:
            probe_fallback = probe["fallback_embedding"] if "fallback_embedding" in probe else None
            if _is_vector_value(probe_fallback):
                query_embedding = probe_fallback
                fallback_query_source = "fallback_embedding"
            elif query_angular_error == 0.0:
                query_embedding = probe["embedding"]
                fallback_query_source = "embedding"
            else:
                query_embedding = None

            if query_embedding is not None:
                exact = exact_open_set_decision(
                    query=query_embedding,
                    templates=fallback_template_embeddings,
                    template_identities=template_identities,
                    threshold=threshold,
                )
                fallback_used = True
                fallback_template_source = "fallback_embedding"
                fallback_decision = exact.decision
                fallback_identity = exact.selected_identity
                fallback_top1_identity = exact.top1_identity
                fallback_top1_score = exact.top1_score
                final_decision = exact.decision
                final_identity = exact.selected_identity
                final_decision_source = "exact_fallback"
        records.append(
            {
                "certified_decision": decision.decision,
                "certified_identity": decision.selected_identity if decision.decision == "accept" else None,
                "certified_fallback_required": decision.decision == "defer",
                "certified_rank": bool(decision.rank_certified),
                "certification_threshold": float(threshold),
                "certified_query_angular_error": query_angular_error,
                "certified_top1_template_angular_error": top1_template_angular_error,
                "certified_top1_total_angular_error": top1_total_angular_error,
                "certified_top1_approximate_angle": top1_approximate_angle,
                "certified_top1_lower_bound": top1_lower_bound,
                "certified_top1_upper_bound": top1_upper_bound,
                "certified_top1_bound_width": top1_bound_width,
                "certified_max_upper_bound": max_upper_bound,
                "certified_max_other_upper_bound": max_other_upper_bound,
                "certified_top1_threshold_margin": top1_threshold_margin,
                "certified_rank_margin": rank_margin,
                "certified_reject_margin": reject_margin,
                "certified_decision_margin": decision_margin,
                "certification_candidate_scope": candidate_scope,
                "certification_candidate_count": candidate_count,
                "certification_gallery_size": gallery_size_value,
                "certification_global_claim": global_claim,
                "fallback_used": fallback_used,
                "fallback_query_source": fallback_query_source,
                "fallback_template_source": fallback_template_source,
                "fallback_decision": fallback_decision,
                "fallback_identity": fallback_identity,
                "fallback_top1_identity": fallback_top1_identity,
                "fallback_top1_score": fallback_top1_score,
                "final_decision": final_decision,
                "final_identity": final_identity,
                "final_decision_source": final_decision_source,
            }
        )

    certified = pd.DataFrame.from_records(records)
    for identity_column in [
        "certified_identity",
        "fallback_identity",
        "fallback_top1_identity",
        "final_identity",
    ]:
        certified[identity_column] = pd.Series(
            [record[identity_column] for record in records],
            dtype=object,
        )
    return pd.concat([features, certified], axis=1)


def summarize_certified_search_features(features: pd.DataFrame) -> dict:
    required = {"certified_decision", "certified_fallback_required"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"missing certified feature columns: {sorted(missing)}")
    if features.empty:
        raise ValueError("certified feature frame must not be empty")

    def _summarize(frame: pd.DataFrame) -> dict:
        def _finite_mean(column: str) -> float | None:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                return None
            return float(values.mean())

        def _finite_max(column: str) -> float | None:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                return None
            return float(values.max())

        counts = {
            decision: int((frame["certified_decision"] == decision).sum())
            for decision in ["accept", "reject", "defer"]
        }
        total = float(len(frame))
        result = {
            "total": int(total),
            "decision_counts": counts,
            "certification_coverage": (counts["accept"] + counts["reject"]) / total,
            "defer_rate": counts["defer"] / total,
            "fallback_rate": float(frame["certified_fallback_required"].astype(bool).mean()),
        }
        if "final_decision" in frame.columns:
            result["final_decision_counts"] = {
                decision: int((frame["final_decision"] == decision).sum())
                for decision in ["accept", "reject", "defer"]
            }
        if "fallback_used" in frame.columns or "final_decision_source" in frame.columns:
            if "fallback_used" in frame.columns:
                exact_fallback_mask = frame["fallback_used"].fillna(False).astype(bool)
            else:
                exact_fallback_mask = frame["final_decision_source"].isin(
                    EXACT_FALLBACK_SOURCES
                )
            result["exact_fallback_rate"] = float(exact_fallback_mask.mean())
            fallback_required_mask = frame["certified_fallback_required"].astype(bool)
            if fallback_required_mask.any():
                result["fallback_resolution_rate"] = float(
                    exact_fallback_mask[fallback_required_mask].mean()
                )
            else:
                result["fallback_resolution_rate"] = None
        if "certified_top1_bound_width" in frame.columns:
            result["mean_top1_bound_width"] = _finite_mean("certified_top1_bound_width")
            result["max_top1_bound_width"] = _finite_max("certified_top1_bound_width")
        if "certified_decision_margin" in frame.columns:
            result["mean_certified_decision_margin"] = _finite_mean(
                "certified_decision_margin"
            )
        if "certified_query_angular_error" in frame.columns:
            result["mean_query_angular_error"] = _finite_mean(
                "certified_query_angular_error"
            )
        if "certified_top1_template_angular_error" in frame.columns:
            result["mean_top1_template_angular_error"] = _finite_mean(
                "certified_top1_template_angular_error"
            )
        if "certified_top1_total_angular_error" in frame.columns:
            result["mean_top1_total_angular_error"] = _finite_mean(
                "certified_top1_total_angular_error"
            )
        if "certification_candidate_scope" in frame.columns:
            result["candidate_scope_counts"] = {
                str(scope): int(count)
                for scope, count in frame["certification_candidate_scope"].value_counts(sort=False).items()
            }
        return result

    summary = _summarize(features)
    if "probe_type" in features.columns:
        summary["by_probe_type"] = {
            str(probe_type): _summarize(group)
            for probe_type, group in features.groupby("probe_type", sort=True)
        }
    else:
        summary["by_probe_type"] = {}
    return summary
