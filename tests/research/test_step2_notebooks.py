from __future__ import annotations

from pathlib import Path

import nbformat
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"
MODEL_ROOT = NOTEBOOK_ROOT / "prerequisite" / "models"
GRADCAM_ROOT = NOTEBOOK_ROOT / "experiments" / "gradcam"
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
        "mode": "dev",
        "data_fraction": 1.0,
        "seed": 42,
        "execute_stage": True,
        "write_outputs": True,
        "overwrite": True,
    }
    aligned = config["aligned_crops"]
    assert aligned["faces_path"].endswith("aligned_faces.npy")
    assert aligned["index_path"].endswith("aligned_index.csv")
    assert aligned["failed_samples_path"].endswith("failed_samples.csv")
    assert aligned["bundle_manifest_path"].endswith("bundle_manifest.json")


def test_model_validation_remains_fail_closed_on_inputs() -> None:
    sources = _sources(MODEL_ROOT)
    assert set(sources) == {
        "00_checkpoint_registration.ipynb",
        "01_preprocessing_and_model_smoke.ipynb",
    }
    for source in sources.values():
        assert 'MODEL_PROFILE = "arcface_ms1mv3_r100"' in source
        assert "DATA_FRACTION = 1.0" in source
        assert "EXECUTE_STAGE = True" in source
        assert "WRITE_OUTPUTS = True" in source
        assert "OVERWRITE = True" in source
        assert "if WRITE_OUTPUTS and not EXECUTE_STAGE:" in source

    registration = sources["00_checkpoint_registration.ipynb"]
    assert "write_model_spec" in registration
    assert 'SOURCE_COLOR_ORDER = CONFIG["aligned_crops"]["source_color_order"]' in (
        registration
    )
    smoke = sources["01_preprocessing_and_model_smoke.ipynb"]
    assert "create_pytorch_adapter_from_spec" in smoke
    assert "select_model_spec" in smoke
    assert "resolve_smoke_input_batch" in smoke
    assert "SMOKE_INPUT_PATH = None" in smoke
    assert "SMOKE_CROPS_NPZ" not in smoke


def test_gradcam_prerequisites_precede_population_experiment() -> None:
    prerequisite = _sources(GRADCAM_ROOT / "prerequisite")
    experiment = _sources(GRADCAM_ROOT / "experiment")
    assert tuple(prerequisite) == (
        "00_source_and_model_freeze.ipynb",
        "01_origin_embedding_and_loo_templates.ipynb",
    )
    assert tuple(experiment) == (
        "00_population_gradcam_extraction.ipynb",
        "01_saliency_feature_validation.ipynb",
        "02_step2_compression_characterization.ipynb",
        "03_saliency_compression_join.ipynb",
        "04_representative_case_visualization.ipynb",
    )

    freeze = prerequisite["00_source_and_model_freeze.ipynb"]
    assert "select_model_spec" in freeze
    assert "RunStore.create_or_reuse_active" in freeze
    assert 'CONFIG["aligned_crops"]["index_path"]' in freeze
    assert 'CONFIG["aligned_crops"]["faces_path"]' in freeze
    assert "PAIRED_METRICS" not in freeze

    origin = prerequisite["01_origin_embedding_and_loo_templates.ipynb"]
    assert "prepare_population_saliency_inputs" in origin
    assert "write_prepared_population_artifact" in origin
    assert "require_all_eligible=False" in origin
    assert "overwrite=OVERWRITE" in origin

    population = experiment["00_population_gradcam_extraction.ipynb"]
    assert "extract_population_gradcam" in population
    assert "write_population_saliency_artifact" in population
    assert "minimum_pass_repeat_cosine" in population
    assert "overwrite=OVERWRITE" in population


def test_gradcam_compression_is_now_generated_sequentially_as_csv() -> None:
    sources = _sources(GRADCAM_ROOT / "experiment")
    compression = sources["02_step2_compression_characterization.ipynb"]
    assert "characterize_step2_compression" in compression
    assert "annotate_compression_lineage" in compression
    assert "origin_embedding_artifact_uid" in compression
    assert "origin_fallback_used" in compression
    assert '"pca_pq"' in compression
    assert "to_csv" in compression
    assert "read_parquet" not in compression
    assert "to_parquet" not in compression

    join = sources["03_saliency_compression_join.ipynb"]
    assert "join_population_saliency_with_compression" in join
    assert "saliency_compression_associations" in join
    assert "pd.read_csv" in join
    for key in ("extraction_uid", "dataset_id", "sample_id", "model_uid"):
        assert key in join

    visualization = sources["04_representative_case_visualization.ipynb"]
    assert "select_population_representative_cases" in visualization
    assert "read_population_heatmaps" in visualization
    assert '"regenerated_gradcam": False' in visualization
    assert "extract_population_gradcam" not in visualization
    assert "PairCosineGradCAM" not in visualization
    assert "resolve_active_run" in visualization
    assert "RUN.complete()" in visualization


def test_current_gradcam_tabular_artifacts_use_csv() -> None:
    all_sources = "\n".join(
        [
            *_sources(GRADCAM_ROOT / "prerequisite").values(),
            *_sources(GRADCAM_ROOT / "experiment").values(),
        ]
    )
    assert "to_parquet" not in all_sources
    assert "pd.read_parquet" not in all_sources
