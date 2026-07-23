from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd


OcclusionStrategy = Literal["high_saliency", "low_saliency", "random"]


def _heatmap_batch(heatmaps: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmaps, dtype=np.float64)
    if values.ndim == 2:
        values = values[np.newaxis, ...]
    if values.ndim != 3 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError("heatmaps must have shape [H, W] or [batch, H, W]")
    if not np.all(np.isfinite(values)):
        raise ValueError("heatmaps must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("heatmaps must be non-negative")
    return values


def saliency_entropy(
    heatmaps: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Return spatial entropy per heatmap; zero-mass maps remain undefined."""

    values = _heatmap_batch(heatmaps)
    flat = values.reshape(len(values), -1)
    mass = flat.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        flat,
        mass,
        out=np.zeros_like(flat),
        where=mass > 0.0,
    )
    log_probabilities = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    log_probabilities[positive] = np.log(probabilities[positive])
    entropy = -np.sum(probabilities * log_probabilities, axis=1)
    if normalize and flat.shape[1] > 1:
        entropy = entropy / np.log(float(flat.shape[1]))
    entropy[mass[:, 0] <= 0.0] = np.nan
    return entropy


def saliency_concentration(
    heatmaps: np.ndarray,
    region_masks: np.ndarray,
) -> np.ndarray:
    """Return the fraction of saliency mass falling inside each region mask."""

    values = _heatmap_batch(heatmaps)
    masks = np.asarray(region_masks)
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    if masks.ndim != 3 or masks.shape[1:] != values.shape[1:]:
        raise ValueError("region_masks must match the heatmap spatial shape")
    if masks.shape[0] == 1 and len(values) > 1:
        masks = np.broadcast_to(masks, values.shape)
    if masks.shape[0] != len(values):
        raise ValueError("region_masks batch must be one or match heatmaps")
    masks = masks.astype(bool, copy=False)
    total = values.sum(axis=(1, 2))
    inside = np.where(masks, values, 0.0).sum(axis=(1, 2))
    return np.divide(
        inside,
        total,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=total > 0.0,
    )


def central_region_concentration(
    heatmaps: np.ndarray,
    *,
    height_fraction: float = 0.5,
    width_fraction: float = 0.5,
) -> np.ndarray:
    """Return saliency concentration in a centered rectangular region."""

    values = _heatmap_batch(heatmaps)
    height_fraction = float(height_fraction)
    width_fraction = float(width_fraction)
    if not 0.0 < height_fraction <= 1.0:
        raise ValueError("height_fraction must be in (0, 1]")
    if not 0.0 < width_fraction <= 1.0:
        raise ValueError("width_fraction must be in (0, 1]")

    height, width = values.shape[1:]
    region_height = max(1, int(round(height * height_fraction)))
    region_width = max(1, int(round(width * width_fraction)))
    top = (height - region_height) // 2
    left = (width - region_width) // 2
    mask = np.zeros((height, width), dtype=bool)
    mask[top : top + region_height, left : left + region_width] = True
    return saliency_concentration(values, mask)


def occlude_by_saliency(
    images: np.ndarray,
    heatmaps: np.ndarray,
    *,
    fraction: float,
    strategy: OcclusionStrategy,
    fill_value: float | Sequence[float] = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Occlude an exact pixel fraction using high/low/random spatial ranks.

    Heatmaps are resized to the image grid with deterministic nearest-neighbor
    indexing. Random tie keys avoid a fixed top-left bias when low-resolution
    Grad-CAM maps contain many equal values.
    """

    values = np.asarray(images)
    single_image = values.ndim == 3
    if single_image:
        values = values[np.newaxis, ...]
    if values.ndim != 4 or values.shape[-1] == 0:
        raise ValueError("images must have shape [H, W, C] or [batch, H, W, C]")
    maps = _heatmap_batch(heatmaps)
    if len(maps) == 1 and len(values) > 1:
        maps = np.broadcast_to(maps, (len(values), *maps.shape[1:]))
    if len(maps) != len(values):
        raise ValueError("heatmaps batch must be one or match images")
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if strategy not in {"high_saliency", "low_saliency", "random"}:
        raise ValueError(
            "strategy must be high_saliency, low_saliency, or random"
        )
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    seed = int(seed)

    fill = np.asarray(fill_value, dtype=np.float64)
    if fill.ndim == 0:
        fill = np.repeat(fill, values.shape[-1])
    if fill.shape != (values.shape[-1],) or not np.isfinite(fill).all():
        raise ValueError("fill_value must be finite and match the channel count")
    if np.issubdtype(values.dtype, np.integer):
        limits = np.iinfo(values.dtype)
        fill = np.clip(np.rint(fill), limits.min, limits.max).astype(values.dtype)
    else:
        fill = fill.astype(values.dtype)

    output = values.copy()
    height, width = values.shape[1:3]
    map_height, map_width = maps.shape[1:]
    row_index = np.minimum(
        (np.arange(height) * map_height // height),
        map_height - 1,
    )
    column_index = np.minimum(
        (np.arange(width) * map_width // width),
        map_width - 1,
    )
    pixel_count = height * width
    occlusion_count = max(1, int(np.ceil(pixel_count * fraction)))
    for index in range(len(output)):
        resized = maps[index][row_index[:, None], column_index[None, :]].reshape(-1)
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        if strategy == "random":
            order = rng.permutation(pixel_count)
        else:
            tie_key = rng.random(pixel_count)
            primary = -resized if strategy == "high_saliency" else resized
            order = np.lexsort((tie_key, primary))
        flat_output = output[index].reshape(pixel_count, values.shape[-1])
        flat_output[order[:occlusion_count]] = fill
    return output[0] if single_image else output


def occlusion_faithfulness(
    origin_scores: np.ndarray,
    saliency_occluded_scores: np.ndarray,
    *,
    control_occluded_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    """Measure cosine-score drop after saliency-guided versus control occlusion."""

    origin = np.asarray(origin_scores, dtype=np.float64)
    saliency = np.asarray(saliency_occluded_scores, dtype=np.float64)
    if origin.ndim != 1 or saliency.ndim != 1 or origin.shape != saliency.shape:
        raise ValueError("origin_scores and saliency_occluded_scores must align")
    if len(origin) == 0:
        raise ValueError("score arrays must not be empty")
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(saliency)):
        raise ValueError("score arrays must contain only finite values")
    if np.any(np.abs(origin) > 1.0) or np.any(np.abs(saliency) > 1.0):
        raise ValueError("scores must be cosine values in [-1, 1]")

    saliency_drop = origin - saliency
    result: dict[str, np.ndarray] = {
        "origin_score": origin,
        "saliency_occluded_score": saliency,
        "saliency_score_drop": saliency_drop,
    }
    if control_occluded_scores is not None:
        control = np.asarray(control_occluded_scores, dtype=np.float64)
        if control.ndim != 1 or control.shape != origin.shape:
            raise ValueError("control_occluded_scores must align with origin_scores")
        if not np.all(np.isfinite(control)) or np.any(np.abs(control) > 1.0):
            raise ValueError(
                "control_occluded_scores must be finite cosine values in [-1, 1]"
            )
        control_drop = origin - control
        result.update(
            {
                "control_occluded_score": control,
                "control_score_drop": control_drop,
                "faithfulness_gain_over_control": saliency_drop - control_drop,
            }
        )
    return pd.DataFrame(result)
