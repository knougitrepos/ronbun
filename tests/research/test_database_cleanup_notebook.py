from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "common"
    / "maintenance"
    / "00_selective_cleanup.ipynb"
)


def _load():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    return notebook


def _source() -> str:
    notebook = _load()
    return "\n".join(cell.source for cell in notebook.cells)


def test_database_cleanup_notebook_is_valid_restartable_and_output_free():
    notebook = _load()

    assert notebook.metadata["ronbun"]["restart_policy"] == (
        "restart_kernel_and_run_all"
    )
    assert notebook.metadata["ronbun"]["destructive_operation"] == (
        "guarded_reset"
    )
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        compile(cell.source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec")
        assert cell.execution_count is None
        assert cell.outputs == []


def test_database_cleanup_notebook_has_safe_complete_reset_defaults():
    source = _source()

    assert 'RESET_MODE = "complete_run_reset"' in source
    assert "CONNECT_TO_DATABASE = False" in source
    assert "APPLY_ADDITIVE_SCHEMA = False" in source
    assert "ensure_database_schema(engine)" in source
    assert 'RUN_UID = ""' in source
    assert "ALLOW_COMPLETED_RUN_RESET = False" in source
    assert "ALLOW_PROMOTED_RESULTS_RESET = False" in source
    assert "ALLOW_UNVERIFIED_LINEAGE_RESET = False" in source
    assert "EXECUTE_RESET = False" in source
    assert 'CONFIRMATION_TOKEN = ""' in source
    assert "WRITE_AUDIT_OUTPUT = True" in source


def test_database_cleanup_notebook_uses_guarded_complete_reset_api_and_preview():
    source = _source()

    assert "from research.database.reset import (" in source
    assert "build_run_reset_plan" in source
    assert "execute_run_reset_plan" in source
    assert "run_reset_plan.database_plan.table_rows" in source
    assert "run_reset_plan.local_targets" in source
    assert "run_reset_plan.as_dict()" in source
    assert "run_reset_plan.total_database_rows" in source
    assert "run_reset_plan.total_files" in source
    assert "run_reset_plan.total_bytes" in source
    assert "run_reset_plan.preserved_resources" in source
    assert "run_reset_plan.warnings" in source
    assert "run_reset_plan.blockers" in source
    assert "run_reset_plan.plan_digest" in source
    assert "run_reset_plan.confirmation_token" in source


def test_database_cleanup_notebook_documents_full_run_artifact_closure():
    source = _source()

    required_phrases = (
        "임베딩 추출 및 임베딩 벡터 전처리 산출물",
        "PCA/PQ 학습 모델·변환 결과·codebook",
        "Grad-CAM, leave-one-out(LOO) template",
        "평가 표·그림·로그",
        "run result bundle과 pointer",
        "PostgreSQL embedding/template/평가 결과",
    )
    for phrase in required_phrases:
        assert phrase in source


def test_database_cleanup_notebook_documents_preserved_shared_resources():
    source = _source()

    required_phrases = (
        "data/raw/**",
        "data/interim/**",
        "공용 정렬 얼굴 crop",
        "사전학습 checkpoint",
        "공유 PostgreSQL `images` 테이블",
        "다른 `run_uid`",
        "results/paper/**",
        "quarantine",
        "감사 기록은 삭제된 데이터를 백업하지 않습니다.",
    )
    for phrase in required_phrases:
        assert phrase in source


def test_database_cleanup_notebook_retains_advanced_allowlisted_cleanup():
    source = _source()

    assert '"advanced_database_cleanup"' in source
    assert "SCOPE_EXACT_RUN_UID" in source
    assert "SCOPE_LEGACY_NULL_RUN_UID" in source
    assert "ADVANCED_TABLE_GROUPS_SELECTED" in source
    assert "ADVANCED_TABLE_NAMES_SELECTED" in source
    assert "build_cleanup_plan" in source
    assert "execute_cleanup_plan" in source
    assert "collect_table_totals" in source
    assert "collect_run_inventory" in source
    assert "write_cleanup_audit" in source
    assert "INCLUDE_RESEARCH_RUN_RECORD" not in source


def test_database_cleanup_notebook_contains_no_raw_destructive_operations():
    source = _source()
    lowered = source.lower()

    assert "delete from" not in lowered
    assert ".unlink(" not in lowered
    assert "rmtree(" not in lowered
    assert "os.remove(" not in lowered
    assert "shutil.rmtree" not in lowered
