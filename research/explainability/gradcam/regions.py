from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def build_landmark_region_masks(
    aligned_landmarks: np.ndarray,
    *,
    region_point_indices: Mapping[str, Sequence[int]],
    region_radii_pixels: Mapping[str, float],
    image_size: tuple[int, int] = (112, 112),
    face_polygon_indices: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Rasterize explicit aligned-landmark regions without guessing topology.

    The caller must provide landmark indices for the exact detector topology.
    A 5-point detector can define eye/nose/mouth regions, but cheek, jaw, and
    face masks must remain unavailable unless its specification supplies
    suitable points. Dense 68/106-point layouts can provide those indices
    explicitly. Returned masks have shape ``[N, H, W]``.
    """

    points = np.asarray(aligned_landmarks, dtype=np.float64)
    if points.ndim != 3 or points.shape[0] == 0 or points.shape[2] != 2:
        raise ValueError(
            "aligned_landmarks must have shape [sample_count, point_count, 2]"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("aligned_landmarks must contain only finite values")
    height, width = (int(image_size[0]), int(image_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("image_size values must be positive")
    if np.any(points[..., 0] < 0.0) or np.any(points[..., 0] > width - 1):
        raise ValueError("landmark x coordinates fall outside the aligned crop")
    if np.any(points[..., 1] < 0.0) or np.any(points[..., 1] > height - 1):
        raise ValueError("landmark y coordinates fall outside the aligned crop")

    point_count = points.shape[1]
    row_grid, column_grid = np.ogrid[:height, :width]
    masks: dict[str, np.ndarray] = {}
    for raw_name, raw_indices in region_point_indices.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("region names must not be empty")
        indices = tuple(int(value) for value in raw_indices)
        if not indices:
            raise ValueError(f"region {name!r} must reference at least one point")
        if min(indices) < 0 or max(indices) >= point_count:
            raise ValueError(f"region {name!r} contains an out-of-range landmark index")
        if name not in region_radii_pixels:
            raise ValueError(f"region {name!r} has no radius")
        radius = float(region_radii_pixels[name])
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"region radius for {name!r} must be positive")
        region_mask = np.zeros((len(points), height, width), dtype=bool)
        for sample_index in range(len(points)):
            for point_index in indices:
                center_x, center_y = points[sample_index, point_index]
                distance_squared = (column_grid - center_x) ** 2 + (
                    row_grid - center_y
                ) ** 2
                region_mask[sample_index] |= distance_squared <= radius**2
        masks[name] = region_mask

    if face_polygon_indices is not None:
        indices = tuple(int(value) for value in face_polygon_indices)
        if len(indices) < 3:
            raise ValueError("face_polygon_indices must contain at least 3 points")
        if min(indices) < 0 or max(indices) >= point_count:
            raise ValueError("face_polygon_indices contains an out-of-range index")
        try:
            import cv2
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "OpenCV is required to rasterize a face landmark polygon"
            ) from exc
        face_masks = np.zeros((len(points), height, width), dtype=np.uint8)
        for sample_index in range(len(points)):
            polygon = np.rint(points[sample_index, indices]).astype(np.int32)
            cv2.fillPoly(face_masks[sample_index], [polygon], color=1)
        masks["face"] = face_masks.astype(bool)
    return masks
