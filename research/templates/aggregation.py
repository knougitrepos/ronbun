from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

AggregationMethod = Literal["single", "mean", "outlier_mean", "quality_weighted"]


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _stack_embeddings(rows: pd.DataFrame, embedding_col: str) -> np.ndarray:
    return np.stack([l2_normalize(value) for value in rows[embedding_col].to_numpy()])


def _keep_non_outliers(vectors: np.ndarray, threshold: float) -> np.ndarray:
    if len(vectors) <= 2:
        return np.ones(len(vectors), dtype=bool)
    center = l2_normalize(vectors.mean(axis=0))
    distances = 1.0 - np.clip(vectors @ center, -1.0, 1.0)
    return distances <= threshold


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("quality_temperature must be positive")
    scaled = values / temperature
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    return exp / exp.sum()


def aggregate_templates(
    rows: pd.DataFrame,
    method: AggregationMethod,
    *,
    identity_col: str = "identity_id",
    image_col: str = "image_id",
    embedding_col: str = "embedding",
    quality_col: str = "quality",
    outlier_threshold: float = 0.35,
    quality_temperature: float = 1.0,
) -> pd.DataFrame:
    required = {identity_col, image_col, embedding_col}
    if method == "quality_weighted":
        required.add(quality_col)
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"missing template columns: {sorted(missing)}")

    records = []
    for identity_id, group in rows.sort_values([identity_col, image_col]).groupby(identity_col):
        vectors = _stack_embeddings(group, embedding_col)
        if method == "single":
            keep = np.zeros(len(group), dtype=bool)
            keep[0] = True
            weights = np.array([1.0])
        else:
            keep = _keep_non_outliers(vectors, outlier_threshold) if method in {"outlier_mean", "quality_weighted"} else np.ones(len(group), dtype=bool)
            if not keep.any():
                keep = np.ones(len(group), dtype=bool)
            kept_group = group.iloc[np.flatnonzero(keep)]
            if method == "quality_weighted":
                weights = _softmax(kept_group[quality_col].astype(float).to_numpy(), quality_temperature)
            else:
                weights = np.full(keep.sum(), 1.0 / keep.sum())

        kept_vectors = vectors[keep]
        template = l2_normalize(np.average(kept_vectors, axis=0, weights=weights))
        distances = 1.0 - np.clip(kept_vectors @ template, -1.0, 1.0)
        kept_group = group.iloc[np.flatnonzero(keep)]
        quality = (
            float(np.average(kept_group[quality_col].astype(float).to_numpy(), weights=weights))
            if quality_col in kept_group.columns
            else float("nan")
        )
        records.append(
            {
                "identity_id": identity_id,
                "embedding": template,
                "quality": quality,
                "variance": float(np.mean(distances**2)) if len(distances) else 0.0,
                "enrollment_count": int(len(kept_group)),
                "aggregation_method": method,
                "source_image_ids": kept_group[image_col].tolist(),
            }
        )

    return pd.DataFrame.from_records(records).sort_values("identity_id").reset_index(drop=True)
