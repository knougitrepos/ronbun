from research.explainability.gradcam.artifacts import (
    read_population_heatmaps,
    read_population_saliency_features,
    read_prepared_population_artifact,
    write_population_saliency_artifact,
    write_prepared_population_artifact,
)
from research.explainability.gradcam.cases import (
    CASE_GROUPS,
    select_gradcam_cases,
    select_population_representative_cases,
    select_representative_cases,
)
from research.explainability.gradcam.extraction import (
    PopulationSaliencyResult,
    PreparedPopulationInputs,
    extract_population_gradcam,
    prepare_population_saliency_inputs,
)
from research.explainability.gradcam.features import (
    FACE_MASK_NAME,
    QUADRANT_COLUMNS,
    SEMANTIC_ATTENTION_COLUMNS,
    SEMANTIC_REGION_NAMES,
    quadrant_saliency_concentration,
    saliency_spatial_moments,
    summarize_saliency_features,
)
from research.explainability.gradcam.metrics import (
    central_region_concentration,
    occlude_by_saliency,
    occlusion_faithfulness,
    saliency_concentration,
    saliency_entropy,
)
from research.explainability.gradcam.optional import (
    TorchUnavailableError,
    is_torch_available,
)
from research.explainability.gradcam.pair import (
    PairCosineGradCAM,
    PairGradCAMResult,
    pair_cosine_target,
)
from research.explainability.gradcam.regions import build_landmark_region_masks
from research.explainability.gradcam.templates import (
    ELIGIBLE_REASON,
    LOO_TARGET_NAME,
    MISSING_IDENTITY_REASON,
    SINGLETON_IDENTITY_REASON,
    ZERO_RESIDUAL_REASON,
    LeaveOneOutTemplateBundle,
    build_leave_one_out_identity_templates,
)

__all__ = [
    "CASE_GROUPS",
    "ELIGIBLE_REASON",
    "FACE_MASK_NAME",
    "LOO_TARGET_NAME",
    "LeaveOneOutTemplateBundle",
    "MISSING_IDENTITY_REASON",
    "PairCosineGradCAM",
    "PairGradCAMResult",
    "PopulationSaliencyResult",
    "PreparedPopulationInputs",
    "QUADRANT_COLUMNS",
    "SEMANTIC_ATTENTION_COLUMNS",
    "SEMANTIC_REGION_NAMES",
    "SINGLETON_IDENTITY_REASON",
    "TorchUnavailableError",
    "ZERO_RESIDUAL_REASON",
    "build_landmark_region_masks",
    "build_leave_one_out_identity_templates",
    "central_region_concentration",
    "extract_population_gradcam",
    "is_torch_available",
    "occlude_by_saliency",
    "occlusion_faithfulness",
    "pair_cosine_target",
    "prepare_population_saliency_inputs",
    "quadrant_saliency_concentration",
    "read_population_heatmaps",
    "read_population_saliency_features",
    "read_prepared_population_artifact",
    "saliency_concentration",
    "saliency_entropy",
    "saliency_spatial_moments",
    "select_gradcam_cases",
    "select_population_representative_cases",
    "select_representative_cases",
    "summarize_saliency_features",
    "write_population_saliency_artifact",
    "write_prepared_population_artifact",
]
