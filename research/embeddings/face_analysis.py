from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from insightface.app import FaceAnalysis


@dataclass(frozen=True)
class FaceAnalysisSettings:
    model_name: str = "buffalo_l"
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    ctx_id: int = 0
    detection_width: int = 640
    detection_height: int = 640


@lru_cache(maxsize=4)
def get_face_analysis_app(settings: FaceAnalysisSettings | None = None) -> FaceAnalysis:
    resolved = settings or FaceAnalysisSettings()
    app = FaceAnalysis(name=resolved.model_name, providers=list(resolved.providers))
    app.prepare(
        ctx_id=resolved.ctx_id,
        det_size=(resolved.detection_width, resolved.detection_height),
    )
    return app
