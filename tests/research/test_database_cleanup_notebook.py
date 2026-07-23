from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "database" / "selective_cleanup.ipynb"
)


def _load():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    return notebook


def test_database_cleanup_notebook_is_valid_restartable_and_output_free():
    notebook = _load()

    assert notebook.metadata["ronbun"]["restart_policy"] == (
        "restart_kernel_and_run_all"
    )
    assert notebook.metadata["ronbun"]["destructive_operation"] == (
        "guarded_delete"
    )
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        compile(cell.source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec")
        assert cell.execution_count is None
        assert cell.outputs == []


def test_database_cleanup_notebook_defaults_to_no_connection_and_no_delete():
    notebook = _load()
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "CONNECT_TO_DATABASE = False" in source
    assert "EXECUTE_DELETE = False" in source
    assert 'CONFIRMATION_TOKEN = ""' in source
    assert "ALLOW_COMPLETED_RUN_DELETE = False" in source
    assert 'RUN_UID = ""' in source
    assert "INCLUDE_RESEARCH_RUN_RECORD = False" in source
    assert "WRITE_AUDIT_OUTPUT = True" in source


def test_database_cleanup_notebook_calls_guarded_module_without_raw_delete_sql():
    notebook = _load()
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "collect_table_totals" in source
    assert "collect_run_inventory" in source
    assert "build_cleanup_plan" in source
    assert "execute_cleanup_plan" in source
    assert "write_cleanup_audit" in source
    assert "DELETE FROM" not in source.upper()
    assert "images and arbitrary table names" not in source
    assert "`images`는 이 노트북에서 삭제하지 않습니다." in source
