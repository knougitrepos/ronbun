from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"

MODEL_VALIDATION_EXPECTED = {
    "00_checkpoint_registration.ipynb",
    "01_preprocessing_and_model_smoke.ipynb",
}
GRADCAM_EXPECTED = {
    "00_source_and_model_freeze.ipynb",
    "01_case_selection.ipynb",
    "02_pair_gradcam_generation.ipynb",
    "03_saliency_feature_analysis.ipynb",
    "04_faithfulness_and_report.ipynb",
}


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
        NOTEBOOK_ROOT / "lfw" / "gradcam": GRADCAM_EXPECTED,
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


def test_step2_notebooks_fail_closed_and_keep_gradcam_separate() -> None:
    model_sources = _sources(NOTEBOOK_ROOT / "model_validation")
    gradcam_sources = _sources(NOTEBOOK_ROOT / "lfw" / "gradcam")
    all_sources = [*model_sources.values(), *gradcam_sources.values()]

    for source in all_sources:
        assert 'MODEL_NAME = "arcface"' in source
        assert 'MODE = "dev"' in source
        assert "DATA_FRACTION = 0.10" in source
        assert "SEED = 42" in source
        assert "EXECUTE_STAGE = False" in source
        assert "WRITE_OUTPUTS = False" in source

    assert "write_model_spec" in model_sources["00_checkpoint_registration.ipynb"]
    assert (
        "create_pytorch_adapter_from_spec"
        in model_sources["01_preprocessing_and_model_smoke.ipynb"]
    )
    assert "select_gradcam_cases" in gradcam_sources["01_case_selection.ipynb"]
    generation = gradcam_sources["02_pair_gradcam_generation.ipynb"]
    assert "PairCosineGradCAM" in generation
    assert 'target_space="origin_embedding"' in generation
    assert "hard PQ" not in generation
    faithfulness = gradcam_sources["04_faithfulness_and_report.ipynb"]
    assert "occlude_by_saliency" in faithfulness
    assert "occlusion_faithfulness" in faithfulness

    for source in gradcam_sources.values():
        assert "fit_pca" not in source
        assert "fit_pq" not in source

    step1 = nbformat.read(
        NOTEBOOK_ROOT / "lfw" / "06_step1_compression_characterization.ipynb",
        as_version=4,
    )
    step1_source = "\n".join(cell.source for cell in step1.cells)
    assert "PairCosineGradCAM" not in step1_source
