from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "experiments" / "step1_embedding_compression.yaml"
)


def _code_source(dataset: str, notebook_name: str) -> str:
    notebook = nbformat.read(NOTEBOOK_ROOT / dataset / notebook_name, as_version=4)
    return "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )


def _call_argument_names(source: str, function_name: str) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name == function_name:
            calls.append(
                tuple(argument.id for argument in node.args if isinstance(argument, ast.Name))
            )
    return calls


def test_data_preparation_applies_one_scope_without_breaking_split_boundaries():
    for dataset in ("lfw", "survface"):
        source = _code_source(dataset, "data_preparation.ipynb")

        assert 'MODE = "dev"' in source or "MODE = 'dev'" in source
        assert "DATA_FRACTION = 0.10" in source
        assert "SEED = 42" in source
        assert "ExperimentScope(" in source
        assert source.count("select_manifest_fraction(") >= 2
        assert "select_open_set_protocol_fraction(" in source
        assert "validate_identity_disjoint_splits(" in source


def test_step1_notebooks_reject_stale_scope_and_manifest_identity_leakage():
    for dataset in ("lfw", "survface"):
        source = _code_source(dataset, "06_step1_compression_characterization.ipynb")

        assert "validate_prepared_scope(" in source
        assert "expected_scope = EXPERIMENT_SCOPE.as_dict()" in source
        assert "MODE/DATA_FRACTION/SEED" in source
        assert "validate_identity_disjoint_splits(" in source
        assert 'get("fit_split") != "development"' in source
        assert 'get("enabled") != [MODEL_NAME]' in source


def test_step1_notebook_paths_are_guarded_by_the_dataset_config():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    lfw_config = config["datasets"]["lfw"]
    survface_config = config["datasets"]["survface"]
    lfw_source = _code_source("lfw", "06_step1_compression_characterization.ipynb")
    survface_source = _code_source(
        "survface", "06_step1_compression_characterization.ipynb"
    )

    assert lfw_config["manifest_path"] == "data/interim/lfw/face_manifest.csv"
    assert 'dataset_config.get("manifest_path")' in lfw_source
    assert "configured_manifest" in lfw_source
    assert lfw_config["protocol_adapter"] == "lfw_identity_disjoint"
    assert "lfw_identity_disjoint" in lfw_source

    assert (
        survface_config["training_manifest_path"]
        == "data/interim/survface/training_manifest.csv"
    )
    assert survface_config["manifest_path"] == "data/interim/survface/official_manifest.csv"
    assert 'dataset_config.get("training_manifest_path")' in survface_source
    assert 'dataset_config.get("manifest_path")' in survface_source
    assert survface_config["protocol_adapter"] == "survface_official"
    assert "survface_official" in survface_source


def test_survface_official_test_is_evaluation_only_in_step1_code_path():
    preparation_source = _code_source("survface", "data_preparation.ipynb")
    study_source = _code_source(
        "survface", "06_step1_compression_characterization.ipynb"
    )

    assert "opaque_per_image_key_no_identity_labels" in preparation_source
    assert _call_argument_names(study_source, "fit") == [
        ("development_matrix",)
    ]
    assert _call_argument_names(study_source, "fit_pca_family") == [
        ("development_matrix",)
    ]
    assert _call_argument_names(study_source, "build_calibration_protocol") == [
        ("training_manifest",)
    ]
    assert _call_argument_names(study_source, "build_survface_official_protocol") == [
        ("official_manifest",)
    ]
    threshold_inputs = _call_argument_names(study_source, "choose_profile_threshold")
    assert threshold_inputs == [
        ("calibration_comparison",),
        ("calibration_comparison",),
    ]
    assert 'compared["threshold_source_split"] = "training_calibration"' in study_source
    assert 'compared["evaluation_split"] = "official_test"' in study_source
    assert '"official_test_role": "evaluation_only"' in study_source

