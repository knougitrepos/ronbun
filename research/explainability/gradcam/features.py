from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from research.explainability.gradcam.metrics import (
    _heatmap_batch,
    saliency_concentration,
    saliency_entropy,
)


SEMANTIC_REGION_NAMES = (
    "left_eye",
    "right_eye",
    "nose",
    "mouth",
    "left_cheek",
    "right_cheek",
    "jaw",
)
FACE_MASK_NAME = "face"
QUADRANT_COLUMNS = (
    "quadrant_top_left",
    "quadrant_top_right",
    "quadrant_bottom_left",
    "quadrant_bottom_right",
)
SEMANTIC_ATTENTION_COLUMNS = tuple(
    f"{name}_attention" for name in SEMANTIC_REGION_NAMES
)


def _nearest_resize_masks(
    masks: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    source_height, source_width = masks.shape[-2:]
    if (source_height, source_width) == (height, width):
        return masks.astype(bool, copy=False)
    row_index = np.minimum(
        np.arange(height) * source_height // height,
        source_height - 1,
    )
    column_index = np.minimum(
        np.arange(width) * source_width // width,
        source_width - 1,
    )
    return masks[:, row_index[:, None], column_index[None, :]].astype(
        bool,
        copy=False,
    )


def _region_mask_batch(
    mask: np.ndarray,
    *,
    sample_count: int,
    height: int,
    width: int,
    name: str,
) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim == 2:
        values = values[np.newaxis, ...]
    if values.ndim != 3 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError(f"region mask {name!r} must have shape [H, W] or [N, H, W]")
    if values.shape[0] == 1 and sample_count > 1:
        values = np.broadcast_to(
            values,
            (sample_count, *values.shape[1:]),
        )
    if values.shape[0] != sample_count:
        raise ValueError(f"region mask {name!r} batch must be one or match heatmaps")
    return _nearest_resize_masks(values, height=height, width=width)


def quadrant_saliency_concentration(heatmaps: np.ndarray) -> pd.DataFrame:
    """Return a complete, non-overlapping four-quadrant saliency partition."""

    values = _heatmap_batch(heatmaps)
    _, height, width = values.shape
    row_split = (height + 1) // 2
    column_split = (width + 1) // 2
    masks = []
    for row_slice, column_slice in (
        (slice(0, row_split), slice(0, column_split)),
        (slice(0, row_split), slice(column_split, width)),
        (slice(row_split, height), slice(0, column_split)),
        (slice(row_split, height), slice(column_split, width)),
    ):
        mask = np.zeros((height, width), dtype=bool)
        mask[row_slice, column_slice] = True
        masks.append(mask)
    return pd.DataFrame(
        {
            column: saliency_concentration(values, mask)
            for column, mask in zip(QUADRANT_COLUMNS, masks)
        }
    )


def saliency_spatial_moments(heatmaps: np.ndarray) -> pd.DataFrame:
    """Return normalized centroid, spread, and horizontal asymmetry."""

    values = _heatmap_batch(heatmaps)
    sample_count, height, width = values.shape
    flat = values.reshape(sample_count, -1)
    mass = flat.sum(axis=1)
    valid = mass > 0.0
    probability = np.divide(
        values,
        mass[:, None, None],
        out=np.zeros_like(values),
        where=valid[:, None, None],
    )
    x_axis = (
        np.linspace(0.0, 1.0, width, dtype=np.float64)
        if width > 1
        else np.zeros(1, dtype=np.float64)
    )
    y_axis = (
        np.linspace(0.0, 1.0, height, dtype=np.float64)
        if height > 1
        else np.zeros(1, dtype=np.float64)
    )
    center_x = np.sum(probability * x_axis[None, None, :], axis=(1, 2))
    center_y = np.sum(probability * y_axis[None, :, None], axis=(1, 2))
    squared_distance = (x_axis[None, None, :] - center_x[:, None, None]) ** 2 + (
        y_axis[None, :, None] - center_y[:, None, None]
    ) ** 2
    spread = np.sqrt(np.sum(probability * squared_distance, axis=(1, 2)))
    asymmetry = 0.5 * np.sum(
        np.abs(probability - probability[:, :, ::-1]),
        axis=(1, 2),
    )
    for array in (center_x, center_y, spread, asymmetry):
        array[~valid] = np.nan
    return pd.DataFrame(
        {
            "saliency_center_x": center_x,
            "saliency_center_y": center_y,
            "saliency_spread": spread,
            "left_right_asymmetry": asymmetry,
        }
    )


def summarize_saliency_features(
    heatmaps: np.ndarray,
    *,
    region_masks: Mapping[str, np.ndarray] | None = None,
    semantic_region_names: Sequence[str] = SEMANTIC_REGION_NAMES,
) -> pd.DataFrame:
    """Materialize population-level spatial Grad-CAM features.

    Semantic columns remain NaN when a validated mask is unavailable. This
    keeps one stable schema without pretending that quadrants or a centered
    rectangle are anatomical eye/cheek/jaw masks.
    """

    values = _heatmap_batch(heatmaps)
    sample_count, height, width = values.shape
    frame = quadrant_saliency_concentration(values)
    moments = saliency_spatial_moments(values)
    for column in moments:
        frame[column] = moments[column].to_numpy()
    frame["saliency_entropy"] = saliency_entropy(values)
    frame["saliency_mass"] = values.sum(axis=(1, 2))
    frame["saliency_valid"] = frame["saliency_mass"].to_numpy() > 0.0

    masks = {} if region_masks is None else dict(region_masks)
    semantic_columns: list[str] = []
    available_count = np.zeros(sample_count, dtype=np.int64)
    for raw_name in semantic_region_names:
        name = str(raw_name).strip()
        if not name:
            raise ValueError("semantic region names must not be empty")
        column = f"{name}_attention"
        semantic_columns.append(column)
        if name not in masks:
            frame[column] = np.nan
            continue
        batch = _region_mask_batch(
            masks[name],
            sample_count=sample_count,
            height=height,
            width=width,
            name=name,
        )
        frame[column] = saliency_concentration(values, batch)
        available_count += np.any(batch, axis=(1, 2)).astype(np.int64)

    if FACE_MASK_NAME in masks:
        face_masks = _region_mask_batch(
            masks[FACE_MASK_NAME],
            sample_count=sample_count,
            height=height,
            width=width,
            name=FACE_MASK_NAME,
        )
        frame["face_attention"] = saliency_concentration(
            values,
            face_masks,
        )
        frame["outside_face_attention"] = saliency_concentration(
            values,
            ~face_masks,
        )
        frame["face_mask_available"] = np.any(face_masks, axis=(1, 2))
    else:
        frame["face_attention"] = np.nan
        frame["outside_face_attention"] = np.nan
        frame["face_mask_available"] = False
    frame["semantic_region_mask_count"] = available_count

    concentration_columns = [*QUADRANT_COLUMNS, *semantic_columns]
    concentrations = frame.loc[:, concentration_columns].to_numpy(
        dtype=np.float64,
    )
    finite = np.isfinite(concentrations)
    maximum = np.full(sample_count, np.nan, dtype=np.float64)
    rows_with_values = np.any(finite, axis=1)
    if np.any(rows_with_values):
        maximum[rows_with_values] = np.max(
            np.where(
                finite[rows_with_values],
                concentrations[rows_with_values],
                -np.inf,
            ),
            axis=1,
        )
    maximum[~frame["saliency_valid"].to_numpy()] = np.nan
    frame["maximum_region_concentration"] = maximum
    return frame
