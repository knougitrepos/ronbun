from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from research.embeddings.face_analysis import FaceAnalysisSettings, get_face_analysis_app


@dataclass(frozen=True)
class ExtractedEmbedding:
    embedding: np.ndarray
    face_count: int
    selected_face_index: int
    bbox: tuple[float, float, float, float]
    detection_score: float


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("ArcFace returned a zero or non-finite embedding")
    return (values / norm).astype(np.float32)


def _face_area(face: Any) -> float:
    x1, y1, x2, y2 = np.asarray(face.bbox, dtype=float).tolist()
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class ArcFaceFeatureExtractor:
    """Deterministic ArcFace extraction for thesis experiments.

    If an image contains multiple faces, the largest detected face is selected;
    detection score breaks equal-area ties. The returned embedding is always a
    finite, L2-normalized 512-dimensional vector.
    """

    def __init__(self, settings: FaceAnalysisSettings | None = None):
        self.settings = settings or FaceAnalysisSettings()
        self.app = get_face_analysis_app(self.settings)

    def extract_with_metadata(self, image: np.ndarray) -> ExtractedEmbedding:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image must be a non-empty OpenCV array")
        faces = self.app.get(image)
        if not faces:
            raise ValueError("no face detected")

        selected_index = max(
            range(len(faces)),
            key=lambda index: (
                _face_area(faces[index]),
                float(getattr(faces[index], "det_score", 0.0)),
                -index,
            ),
        )
        face = faces[selected_index]
        embedding = _l2_normalize(face.embedding)
        if embedding.shape != (512,):
            raise ValueError(f"ArcFace embedding must have 512 dimensions, got {embedding.shape}")
        bbox = tuple(float(value) for value in np.asarray(face.bbox).reshape(-1).tolist())
        if len(bbox) != 4:
            raise ValueError("detected face bbox must have four coordinates")
        return ExtractedEmbedding(
            embedding=embedding,
            face_count=len(faces),
            selected_face_index=selected_index,
            bbox=bbox,
            detection_score=float(getattr(face, "det_score", float("nan"))),
        )

    def extract(self, image: np.ndarray) -> np.ndarray:
        return self.extract_with_metadata(image).embedding

    def extract_image_bytes(self, image_bytes: bytes) -> ExtractedEmbedding:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("image bytes could not be decoded")
        return self.extract_with_metadata(image)
