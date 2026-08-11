from __future__ import annotations

from pathlib import Path

import nbformat
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"
MODEL_ROOT = NOTEBOOK_ROOT / "common" / "model_preparation"
ALIGNED_CROP_NOTEBOOK = (
    NOTEBOOK_ROOT
    / "lfw"
    / "00_data_preparation"
    / "01_aligned_crop_materialization.ipynb"
)
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "experiments" / "step2_pytorch_gradcam.yaml"
)


def _sources(directory: Path) -> dict[str, str]:
    return {
        path.name: "\n".join(
            cell.source for cell in nbformat.read(path, as_version=4).cells
        )
        for path in sorted(directory.glob("*.ipynb"))
    }


def test_step2_config_uses_interpretable_aligned_crop_bundle() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    execution = config["execution"]
    assert execution == {
        "model_profile": "arcface_ms1mv3_r100",
        "mode": "real",
        "data_fraction": 1.0,
        "seed": 42,
        "execute_stage": True,
        "write_outputs": True,
        "overwrite": False,
        "allow_dirty": False,
        "device": "cuda",
    }
    aligned = config["aligned_crops"]
    assert aligned["faces_path"].endswith("aligned_faces.npy")
    assert aligned["index_path"].endswith("aligned_index.csv")
    assert aligned["failed_samples_path"].endswith("failed_samples.csv")
    assert aligned["bundle_manifest_path"].endswith("bundle_manifest.json")
    assert aligned["providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert aligned["required_primary_provider"] == "CUDAExecutionProvider"
    assert config["gradcam"]["population"]["saliency_sample_cap"] == {
        "lfw": None,
        "survface": None,
        "rfw_custom": 10000,
    }
    assert config["datasets"]["lfw"]["aligned_bundle_dir"] == (
        "data/interim/common/aligned_112"
    )


def test_aligned_crop_notebook_discovers_the_actual_repository_root() -> None:
    source = "\n".join(
        cell.source
        for cell in nbformat.read(ALIGNED_CROP_NOTEBOOK, as_version=4).cells
    )
    assert '(candidate / "research").is_dir()' in source
    assert '(candidate / "configs").is_dir()' in source
    assert "pyproject.toml" not in source
    assert 'CONFIG["aligned_crops"]["providers"]' in source
    assert 'CONFIG["aligned_crops"]["required_primary_provider"]' in source
    assert "providers=ALIGNMENT_PROVIDERS" in source


def test_model_validation_remains_fail_closed_on_inputs() -> None:
    sources = _sources(MODEL_ROOT)
    assert set(sources) == {
        "00_checkpoint_registration.ipynb",
        "01_preprocessing_and_model_smoke.ipynb",
    }
    for source in sources.values():
        assert 'EXECUTION = CONFIG["execution"]' in source
        assert 'MODEL_PROFILE = str(EXECUTION["model_profile"])' in source
        assert 'MODE = str(EXECUTION["mode"])' in source
        assert 'DATA_FRACTION = float(EXECUTION["data_fraction"])' in source
        assert 'EXECUTE_STAGE = bool(EXECUTION["execute_stage"])' in source
        assert 'WRITE_OUTPUTS = bool(EXECUTION["write_outputs"])' in source
        assert 'OVERWRITE = bool(EXECUTION["overwrite"])' in source
        assert 'CONFIG["models"]' in source
        assert "runs/step2/model_registry" not in source
        assert "runs/step3/model_registry" not in source
        assert "if WRITE_OUTPUTS and not EXECUTE_STAGE:" in source

    registration = sources["00_checkpoint_registration.ipynb"]
    assert "write_model_spec" in registration
    assert 'SOURCE_COLOR_ORDER = CONFIG["aligned_crops"]["source_color_order"]' in (
        registration
    )
    smoke = sources["01_preprocessing_and_model_smoke.ipynb"]
    assert "create_pytorch_adapter_from_spec" in smoke
    assert "select_model_spec" in smoke
    assert 'MODEL_UID = CONFIG["models"].get("model_uid")' in smoke
    assert "resolve_smoke_input_batch" in smoke
    assert "SMOKE_INPUT_PATH = None" in smoke
    assert "SMOKE_CROPS_NPZ" not in smoke


def test_gradcam_notebooks_are_thin_and_sequential_for_both_datasets() -> None:
    expected = {
        "00_source_and_model_freeze.ipynb": "freeze_step4_source_and_model",
        "00_population_gradcam_extraction.ipynb": (
            "extract_step4_population_gradcam"
        ),
        "01_saliency_feature_validation.ipynb": "validate_step4_saliency",
        "02_step2_compression_characterization.ipynb": (
            "characterize_step4_compression"
        ),
        "03_saliency_compression_join.ipynb": (
            "analyze_step4_saliency_compression"
        ),
        "04_representative_case_visualization.ipynb": (
            "finalize_step4_representative_cases"
        ),
    }
    origin_names = {
        "lfw": "01_origin_embedding_and_loo_templates.ipynb",
        "survface": "01_origin_embedding_and_top1_gallery_templates.ipynb",
    }
    for dataset_id in ("lfw", "survface"):
        root = NOTEBOOK_ROOT / dataset_id / "04_gradcam"
        prerequisite = _sources(root / "prerequisite")
        experiment = _sources(root / "experiment")
        assert tuple(experiment) == (
            "00_population_gradcam_extraction.ipynb",
            "01_saliency_feature_validation.ipynb",
            "02_step2_compression_characterization.ipynb",
            "03_saliency_compression_join.ipynb",
            "04_representative_case_visualization.ipynb",
        )
        assert tuple(prerequisite) == (
            "00_source_and_model_freeze.ipynb",
            origin_names[dataset_id],
        )
        for name, source in {**prerequisite, **experiment}.items():
            function_name = (
                "extract_step4_origin_embeddings"
                if name == origin_names[dataset_id]
                else expected[name]
            )
            assert function_name in source
            assert f'DATASET_ID = "{dataset_id}"' in source
            assert 'EXECUTION = CONFIG["execution"]' in source
            assert 'MODEL_PROFILE = str(EXECUTION["model_profile"])' in source
            assert 'MODE = str(EXECUTION["mode"])' in source
            assert 'DATA_FRACTION = float(EXECUTION["data_fraction"])' in source
            assert 'EXECUTE_STAGE = bool(EXECUTION["execute_stage"])' in source
            assert 'WRITE_OUTPUTS = bool(EXECUTION["write_outputs"])' in source
            assert 'OVERWRITE = bool(EXECUTION["overwrite"])' in source
            assert "execution_acknowledged=True" in source
            assert "read_parquet" not in source
            assert "to_parquet" not in source
            assert "run_step4_experiment" not in source


def test_step4_data_prerequisites_are_separate_notebooks() -> None:
    lfw_landmarks = _sources(
        NOTEBOOK_ROOT / "lfw" / "00_data_preparation"
    )["02_landmark_region_materialization.ipynb"]
    survface = _sources(
        NOTEBOOK_ROOT / "survface" / "00_data_preparation"
    )
    assert "materialize_step4_landmark_regions" in lfw_landmarks
    assert (
        "materialize_step4_aligned_crops"
        in survface["01_aligned_crop_materialization.ipynb"]
    )
    assert (
        "materialize_step4_landmark_regions"
        in survface["02_landmark_region_materialization.ipynb"]
    )
