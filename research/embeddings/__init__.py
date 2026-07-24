from research.embeddings.arcface import ArcFaceFeatureExtractor, ExtractedEmbedding
from research.embeddings.base import (
    CheckpointProvenance,
    EmbeddingOutput,
    ModelSpec,
    PreprocessingSpec,
)
from research.embeddings.face_analysis import FaceAnalysisSettings, get_face_analysis_app
from research.embeddings.manifests import (
    ModelSpecSelectionError,
    read_model_spec,
    select_model_spec,
    write_model_spec,
)
from research.embeddings.registry import (
    ModelFactoryUnavailableError,
    create_pytorch_adapter,
    create_pytorch_adapter_from_spec,
    load_pytorch_module_factory,
    register_pytorch_module_factory,
    unregister_pytorch_module_factory,
)
from research.embeddings.smoke_inputs import (
    SmokeInputBatch,
    resolve_smoke_input_batch,
)

__all__ = [
    "ArcFaceFeatureExtractor",
    "CheckpointProvenance",
    "EmbeddingOutput",
    "ExtractedEmbedding",
    "FaceAnalysisSettings",
    "ModelFactoryUnavailableError",
    "ModelSpec",
    "ModelSpecSelectionError",
    "PreprocessingSpec",
    "SmokeInputBatch",
    "create_pytorch_adapter",
    "create_pytorch_adapter_from_spec",
    "get_face_analysis_app",
    "load_pytorch_module_factory",
    "read_model_spec",
    "register_pytorch_module_factory",
    "resolve_smoke_input_batch",
    "select_model_spec",
    "unregister_pytorch_module_factory",
    "write_model_spec",
]
