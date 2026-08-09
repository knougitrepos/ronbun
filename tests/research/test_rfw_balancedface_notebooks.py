from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"


def _load(dataset: str):
    path = (
        NOTEBOOK_ROOT
        / dataset
        / "00_data_preparation"
        / "00_data_preparation.ipynb"
    )
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return path, notebook


def test_rfw_and_balancedface_are_dataset_prerequisites():
    for dataset in ("balancedface", "rfw"):
        path, notebook = _load(dataset)
        assert path.is_file()
        assert notebook.metadata["ronbun"]["workflow_role"] == "prerequisite"
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            compile(cell.source, f"{path.name}:cell-{index}", "exec")


def test_dataset_prerequisites_use_full_execution_defaults_and_guards():
    _, rfw = _load("rfw")
    _, balanced = _load("balancedface")
    rfw_source = "\n".join(cell.source for cell in rfw.cells)
    balanced_source = "\n".join(cell.source for cell in balanced.cells)

    assert 'MODE = "real"' in rfw_source
    assert 'MODE = "dev"' in balanced_source
    for source in (rfw_source, balanced_source):
        assert "DATA_FRACTION = 1.0" in source
        assert "SEED = 42" in source
        assert "EXECUTE_STAGE = True" in source
        assert "WRITE_OUTPUTS = True" in source
        assert "OVERWRITE = True" in source
        assert "WRITE_OUTPUTS=True requires EXECUTE_STAGE=True" in source

    assert "select_rfw_protocol_scope" in rfw_source
    assert "strict_official=True" in rfw_source
    assert "PCA/PQ를 fit" in rfw_source
    assert "source_identities.txt" in rfw_source

    assert "RFW_SUCCESS_PATH" in balanced_source
    assert "build_balancedface_index_bundle" in balanced_source
    assert "VERIFY_RECORDIO_ARCHIVE = True" in balanced_source
    assert "alignment_and_group_coverage_audit_required" in balanced_source
    assert "RecordIO decoder" in balanced_source


def test_rfw_all_in_one_and_procedural_notebooks_have_complete_contracts():
    all_in_one_path = NOTEBOOK_ROOT / "rfw" / "00_rfw_all_in_one.ipynb"
    compression_path = (
        NOTEBOOK_ROOT
        / "rfw"
        / "02_compression"
        / "00_rfw_frozen_codec_verification.ipynb"
    )
    all_in_one = nbformat.read(all_in_one_path, as_version=4)
    compression = nbformat.read(compression_path, as_version=4)
    for path, notebook in (
        (all_in_one_path, all_in_one),
        (compression_path, compression),
    ):
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            compile(cell.source, f"{path.name}:cell-{index}", "exec")
            assert cell.execution_count is None
            assert cell.outputs == []

    all_source = "\n".join(cell.source for cell in all_in_one.cells)
    compression_source = "\n".join(cell.source for cell in compression.cells)
    for phrase in (
        'ARTIFACT_STORAGE_MODE = "results_only"',
        "ACKNOWLEDGE_LOCAL_EXECUTION",
        "RUN_PROTOCOL_STAGE",
        "RUN_EMBEDDING_STAGE",
        "RUN_EVALUATION_STAGE",
        "frozen_codec_specs_from_completed_run",
        "rfw_frozen_codec_evaluation_uid",
        "evaluate_rfw_frozen_codecs",
        "ALLOW_ORIGIN_ONLY",
        "24,000",
        "48,000",
        "open-set",
    ):
        assert phrase in all_source
    for phrase in (
        "CODEC_SOURCE_RUN_DIRS",
        "frozen_codec_specs_from_completed_run",
        "expected_model_uid=MODEL_UID",
        "ALLOW_ORIGIN_ONLY = True",
        "rfw_frozen_codec_evaluation_uid",
        "load_rfw_frozen_codec_evaluation",
        "origin-only baseline",
    ):
        assert phrase in compression_source
