from research.embeddings.arcface import ArcFaceFeatureExtractor, ExtractedEmbedding
from research.embeddings.base import (
    CheckpointProvenance,
    EmbeddingOutput,
    ModelSpec,
    PreprocessingSpec,
)
from research.embeddings.face_analysis import FaceAnalysisSettings, get_face_analysis_app
from research.embeddings.manifests import read_model_spec, write_model_spec
from research.embeddings.registry import (
    ModelFactoryUnavailableError,
    create_pytorch_adapter,
    create_pytorch_adapter_from_spec,
    load_pytorch_module_factory,
    register_pytorch_module_factory,
    unregister_pytorch_module_factory,
)

__all__ = [
    "ArcFaceFeatureExtractor",
    "CheckpointProvenance",
    "EmbeddingOutput",
    "ExtractedEmbedding",
    "FaceAnalysisSettings",
    "ModelFactoryUnavailableError",
    "ModelSpec",
    "PreprocessingSpec",
    "create_pytorch_adapter",
    "create_pytorch_adapter_from_spec",
    "get_face_analysis_app",
    "load_pytorch_module_factory",
    "read_model_spec",
    "register_pytorch_module_factory",
    "unregister_pytorch_module_factory",
    "write_model_spec",
]
