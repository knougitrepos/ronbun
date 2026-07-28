from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from insightface.app import FaceAnalysis


@dataclass(frozen=True)
class FaceAnalysisSettings:
    model_name: str = "buffalo_l"
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    allowed_modules: tuple[str, ...] = ("detection", "recognition")
    required_primary_provider: str | None = None
    ctx_id: int = 0
    detection_width: int = 640
    detection_height: int = 640


@lru_cache(maxsize=4)
def get_face_analysis_app(settings: FaceAnalysisSettings | None = None) -> FaceAnalysis:
    resolved = settings or FaceAnalysisSettings()
    if (
        not resolved.providers
        or len(set(resolved.providers)) != len(resolved.providers)
        or any(not str(value).strip() for value in resolved.providers)
    ):
        raise ValueError("providers must contain unique non-empty names")
    if (
        not resolved.allowed_modules
        or len(set(resolved.allowed_modules)) != len(resolved.allowed_modules)
        or any(not str(value).strip() for value in resolved.allowed_modules)
    ):
        raise ValueError("allowed_modules must contain unique non-empty names")
    app = FaceAnalysis(
        name=resolved.model_name,
        allowed_modules=list(resolved.allowed_modules),
        providers=list(resolved.providers),
    )
    app.prepare(
        ctx_id=resolved.ctx_id,
        det_size=(resolved.detection_width, resolved.detection_height),
    )
    active_providers: dict[str, tuple[str, ...]] = {}
    for module_name in resolved.allowed_modules:
        model = app.models.get(module_name)
        session = getattr(model, "session", None)
        get_providers = getattr(session, "get_providers", None)
        if model is None or not callable(get_providers):
            raise RuntimeError(
                f"InsightFace did not expose the required {module_name} session"
            )
        providers = tuple(str(value) for value in get_providers())
        if not providers:
            raise RuntimeError(
                f"InsightFace {module_name} session has no active provider"
            )
        active_providers[module_name] = providers
        required = resolved.required_primary_provider
        if required is not None and providers[0] != required:
            raise RuntimeError(
                "ONNX Runtime did not activate the required primary provider "
                f"for {module_name}: required={required}, active={providers}"
            )
    app._ronbun_session_providers = active_providers
    return app
