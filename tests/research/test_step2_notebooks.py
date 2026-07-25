from __future__ import annotations

from pathlib import Path
import re

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"

MODEL_VALIDATION_EXPECTED = {
    "00_checkpoint_registration.ipynb",
    "01_preprocessing_and_model_smoke.ipynb",
}
GRADCAM_EXPECTED_ORDER = (
    "03_source_and_model_freeze.ipynb",
    "04_origin_embedding_and_loo_templates.ipynb",
    "05_population_gradcam_extraction.ipynb",
    "06_saliency_feature_validation.ipynb",
    "07_compression_characterization.ipynb",
    "08_saliency_compression_join.ipynb",
    "09_representative_case_visualization.ipynb",
)


def _sources(directory: Path) -> dict[str, str]:
    return {
        path.name: "\n".join(
            cell.source for cell in nbformat.read(path, as_version=4).cells
        )
        for path in sorted(directory.glob("*.ipynb"))
    }


def test_step2_notebooks_are_valid_restartable_and_output_free() -> None:
    directories = {
        NOTEBOOK_ROOT / "model_validation": MODEL_VALIDATION_EXPECTED,
    }
    for directory, expected_names in directories.items():
        paths = sorted(directory.glob("*.ipynb"))
        assert {path.name for path in paths} == expected_names
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            assert notebook.metadata["ronbun"]["restart_policy"] == (
                "restart_kernel_and_run_all"
            )
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                compile(cell.source, f"{path.name}:cell-{index}", "exec")
                assert cell.execution_count is None
                assert cell.outputs == []

    gradcam_directory = NOTEBOOK_ROOT / "lfw"
    gradcam_paths = [
        path for path in sorted(gradcam_directory.glob("*.ipynb"))
        if path.name in GRADCAM_EXPECTED_ORDER
    ]
    assert tuple(path.name for path in gradcam_paths) == GRADCAM_EXPECTED_ORDER
    for path in gradcam_paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert notebook.metadata["ronbun"]["restart_policy"] == (
            "restart_kernel_and_run_all"
        )
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            compile(cell.source, f"{path.name}:cell-{index}", "exec")
            assert cell.execution_count is None
            assert cell.outputs == []


def test_step2_notebooks_fail_closed_by_default() -> None:
    model_sources = _sources(NOTEBOOK_ROOT / "model_validation")
    lfw_sources = _sources(NOTEBOOK_ROOT / "lfw")
    gradcam_sources = {
        name: source for name, source in lfw_sources.items()
        if name in GRADCAM_EXPECTED_ORDER
    }
    all_sources = [*model_sources.values(), *gradcam_sources.values()]

    for source in all_sources:
        assert 'MODEL_PROFILE = "arcface_ms1mv3_r100"' in source
        assert 'MODE = "dev"' in source
        assert "SEED = 42" in source
        assert "EXECUTE_STAGE = False" in source
        assert "WRITE_OUTPUTS = False" in source
        assert "if WRITE_OUTPUTS and not EXECUTE_STAGE:" in source

    for source in gradcam_sources.values():
        assert "DATA_FRACTION = 0.10" in source
    for source in model_sources.values():
        match = re.search(r"^DATA_FRACTION = ([0-9.]+)", source, re.MULTILINE)
        assert match is not None
        assert 0.0 < float(match.group(1)) <= 1.0

    assert "write_model_spec" in model_sources["00_checkpoint_registration.ipynb"]
    registration = model_sources["00_checkpoint_registration.ipynb"]
    assert (
        'SOURCE_COLOR_ORDER = CONFIG["aligned_crops"]["source_color_order"]'
        in registration
    )
    assert (
        "create_pytorch_adapter_from_spec"
        in model_sources["01_preprocessing_and_model_smoke.ipynb"]
    )
    assert (
        "select_model_spec"
        in model_sources["01_preprocessing_and_model_smoke.ipynb"]
    )
    smoke = model_sources["01_preprocessing_and_model_smoke.ipynb"]
    assert "resolve_smoke_input_batch" in smoke
    assert "SMOKE_INPUT_PATH = None" in smoke
    assert "SMOKE_CROPS_NPZ" not in smoke
    assert "quantitative_experiment_input" not in smoke


def test_gradcam_notebooks_enforce_population_first_stage_boundaries() -> None:
    lfw_sources = _sources(NOTEBOOK_ROOT / "lfw")
    sources = {
        name: source for name, source in lfw_sources.items()
        if name in GRADCAM_EXPECTED_ORDER
    }

    freeze = sources["03_source_and_model_freeze.ipynb"]
    assert "select_model_spec" in freeze
    assert "MODEL_UID = None" in freeze
    assert "MODEL_SPEC_PATH" not in freeze
    assert "SELECTED_MANIFEST_OUTPUT_PATH" in freeze
    assert "FREEZE_MANIFEST_OUTPUT_PATH" in freeze
    assert "PAIRED_METRICS_PATH" not in freeze
    assert "RETRIEVAL_METRICS_PATH" not in freeze

    origin = sources["04_origin_embedding_and_loo_templates.ipynb"]
    assert "read_model_spec" in origin
    assert 'freeze["model_uid"]' in origin
    assert "MODEL_SPEC_PATH" not in origin
    assert "prepare_population_saliency_inputs" in origin
    assert "write_prepared_population_artifact" in origin
    assert "require_all_eligible=False" in origin

    population = sources["05_population_gradcam_extraction.ipynb"]
    assert "read_model_spec" in population
    assert "prepared.model_uid" in population
    assert "MODEL_SPEC_PATH" not in population
    assert "extract_population_gradcam" in population
    assert "read_prepared_population_artifact" in population
    assert "write_population_saliency_artifact" in population
    assert "minimum_pass_repeat_cosine" in population

    validation = sources["06_saliency_feature_validation.ipynb"]
    assert "read_population_saliency_features" in validation
    assert "saliency_target_eligible" in validation
    assert "heatmap_available" in validation

    before_compression = (
        "03_source_and_model_freeze.ipynb",
        "04_origin_embedding_and_loo_templates.ipynb",
        "05_population_gradcam_extraction.ipynb",
        "06_saliency_feature_validation.ipynb",
    )
    for name in before_compression:
        source = sources[name]
        assert "fit_pca" not in source
        assert "fit_pq" not in source
        assert "PAIRED_METRICS_PATH" not in source
        assert "RETRIEVAL_METRICS_PATH" not in source


def test_gradcam_notebooks_compress_join_then_select_cases() -> None:
    lfw_sources = _sources(NOTEBOOK_ROOT / "lfw")
    sources = {
        name: source for name, source in lfw_sources.items()
        if name in GRADCAM_EXPECTED_ORDER
    }

    compression = sources["07_compression_characterization.ipynb"]
    assert "annotate_compression_lineage" in compression
    assert "origin_embedding_artifact_uid" in compression
    assert "origin_fallback_used" in compression
    assert '"pca_pq"' in compression

    join = sources["08_saliency_compression_join.ipynb"]
    assert "join_population_saliency_with_compression" in join
    assert "saliency_compression_associations" in join
    for key in ("extraction_uid", "dataset_id", "sample_id", "model_uid"):
        assert key in join

    visualization = sources["09_representative_case_visualization.ipynb"]
    assert "select_population_representative_cases" in visualization
    assert "read_population_heatmaps" in visualization
    assert '"regenerated_gradcam": False' in visualization
    assert "extract_population_gradcam" not in visualization
    assert "PairCosineGradCAM" not in visualization

    for name, source in sources.items():
        if name == "09_representative_case_visualization.ipynb":
            continue
        assert "select_population_representative_cases" not in source
        assert "select_gradcam_cases" not in source

    step1 = nbformat.read(
        NOTEBOOK_ROOT / "lfw" / "07_compression_characterization.ipynb",
        as_version=4,
    )
    step1_source = "\n".join(cell.source for cell in step1.cells)
    assert "PairCosineGradCAM" not in step1_source
