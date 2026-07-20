"""Reusable experiment-stage orchestration used by dataset-specific notebooks."""

from research.experiments.scope import (
    EXPERIMENT_MODES,
    ExperimentScope,
    select_manifest_fraction,
    select_open_set_protocol_fraction,
)

from research.experiments.materialization import (
    materialize_compressed_embeddings,
    materialize_compressed_embeddings_with_frozen_stats,
    materialize_pca_sweep_embeddings,
)
from research.experiments.lfw_certification import (
    LFWCertificationInputs,
    assemble_lfw_certification_inputs,
    build_lfw_certification_inputs,
    write_vector_frame_csv,
)
from research.experiments.lfw_pgvector import (
    LFWTemplateScope,
    calibrate_lfw_pgvector_threshold,
    materialize_lfw_templates,
    protocol_frames,
    run_lfw_pgvector_search,
    summarize_lfw_pgvector_search,
)

__all__ = [
    "EXPERIMENT_MODES",
    "ExperimentScope",
    "materialize_compressed_embeddings",
    "materialize_compressed_embeddings_with_frozen_stats",
    "materialize_pca_sweep_embeddings",
    "LFWCertificationInputs",
    "assemble_lfw_certification_inputs",
    "build_lfw_certification_inputs",
    "LFWTemplateScope",
    "calibrate_lfw_pgvector_threshold",
    "materialize_lfw_templates",
    "protocol_frames",
    "run_lfw_pgvector_search",
    "select_manifest_fraction",
    "select_open_set_protocol_fraction",
    "summarize_lfw_pgvector_search",
    "write_vector_frame_csv",
]
