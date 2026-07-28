from __future__ import annotations

from types import SimpleNamespace

import pytest

import research.embeddings.face_analysis as module
from research.embeddings.face_analysis import FaceAnalysisSettings


class _Session:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return list(self._providers)


class _FaceAnalysis:
    providers_by_module = {
        "detection": ("CUDAExecutionProvider", "CPUExecutionProvider"),
        "recognition": ("CUDAExecutionProvider", "CPUExecutionProvider"),
    }
    init_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs
        allowed = kwargs["allowed_modules"]
        self.models = {
            name: SimpleNamespace(
                session=_Session(type(self).providers_by_module[name])
            )
            for name in allowed
        }

    def prepare(self, **kwargs):
        self.prepare_kwargs = kwargs


def test_face_analysis_loads_only_detection_and_recognition(monkeypatch):
    module.get_face_analysis_app.cache_clear()
    monkeypatch.setattr(module, "FaceAnalysis", _FaceAnalysis)
    settings = FaceAnalysisSettings(
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        required_primary_provider="CUDAExecutionProvider",
    )

    app = module.get_face_analysis_app(settings)

    assert _FaceAnalysis.init_kwargs["allowed_modules"] == [
        "detection",
        "recognition",
    ]
    assert app._ronbun_session_providers == {
        "detection": (
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ),
        "recognition": (
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ),
    }


def test_face_analysis_fails_closed_when_cuda_falls_back(monkeypatch):
    module.get_face_analysis_app.cache_clear()
    monkeypatch.setattr(module, "FaceAnalysis", _FaceAnalysis)
    monkeypatch.setattr(
        _FaceAnalysis,
        "providers_by_module",
        {
            "detection": ("CUDAExecutionProvider", "CPUExecutionProvider"),
            "recognition": ("CPUExecutionProvider",),
        },
    )

    with pytest.raises(RuntimeError, match="required primary provider"):
        module.get_face_analysis_app(
            FaceAnalysisSettings(
                providers=(
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ),
                required_primary_provider="CUDAExecutionProvider",
            )
        )
