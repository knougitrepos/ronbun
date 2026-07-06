from __future__ import annotations

import numpy as np
import pandas as pd

CALIBRATION_FEATURE_COLUMNS = [
    "top1_score",
    "score_margin",
    "probe_quality",
    "template_quality",
    "template_variance",
    "enrollment_count",
    "reconstruction_error_norm",
]


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
                "y_true_accept": int(probe_type == "registered" and top1_identity == query_identity),
            }
        )

    return pd.DataFrame.from_records(records)
