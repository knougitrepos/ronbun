from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine, func, insert, select
from sqlalchemy.orm import Session

from research.database.cleanup import (
    CleanupConfirmationError,
    CleanupPlanChangedError,
    CleanupSelectionError,
    SCOPE_EXACT_RUN_UID,
    SCOPE_LEGACY_NULL_RUN_UID,
    build_cleanup_plan,
    execute_cleanup_plan,
    resolve_cleanup_tables,
    write_cleanup_audit,
)
from research.database.models import Base


def _engine():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    tables = Base.metadata.tables
    with engine.begin() as connection:
        connection.execute(
            insert(tables["images"]),
            [
                {"id": 1, "image_path": "a.jpg"},
                {"id": 2, "image_path": "b.jpg"},
                {"id": 3, "image_path": "c.jpg"},
            ],
        )
        connection.execute(
            insert(tables["embedding_512"]),
            [
                {
                    "id": 1,
                    "image_id": 1,
                    "run_uid": "run-a",
                    "vector_type": "origin_512",
                    "embedding": [0.0] * 512,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "run_uid": "run-b",
                    "vector_type": "origin_512",
                    "embedding": [0.0] * 512,
                },
            ],
        )
        connection.execute(
            insert(tables["embedding_256"]),
            {
                "id": 1,
                "image_id": 1,
                "run_uid": "run-a",
                "vector_type": "pca_256",
                "embedding": [0.0] * 256,
            },
        )
        connection.execute(
            insert(tables["research_runs"]),
            {
                "id": 1,
                "run_uid": "run-a",
                "run_name": "run a",
                "config_hash": "a" * 64,
                "config": {},
                "status": "created",
                "artifact_dir": "runs/run-a",
            },
        )
        connection.execute(
            insert(tables["research_splits"]),
            {
                "id": 1,
                "run_id": 1,
                "split_name": "development",
                "identity_id": "person-a",
                "role": "gallery",
            },
        )
    return engine


def _count(engine, table_name: str) -> int:
    with engine.connect() as connection:
        return int(
            connection.execute(
                select(func.count()).select_from(
                    Base.metadata.tables[table_name]
                )
            ).scalar_one()
        )


def test_table_selection_is_allowlisted_and_research_parent_is_guarded():
    assert resolve_cleanup_tables(table_names=["embedding_256"]) == (
        "embedding_256",
    )

    with pytest.raises(CleanupSelectionError, match="not in the cleanup allowlist"):
        resolve_cleanup_tables(table_names=["images"])
    with pytest.raises(CleanupSelectionError, match="cannot be selected directly"):
        resolve_cleanup_tables(table_names=["research_runs"])
    with pytest.raises(CleanupSelectionError, match="unknown cleanup table groups"):
        resolve_cleanup_tables(table_groups=["typo"])


def test_exact_run_cleanup_requires_current_token_and_preserves_other_rows(tmp_path):
    engine = _engine()
    with Session(engine) as session:
        plan = build_cleanup_plan(
            session,
            scope_kind=SCOPE_EXACT_RUN_UID,
            run_uid="run-a",
            table_names=["embedding_512"],
            project_root=tmp_path,
        )

    assert plan.total_rows == 1
    assert plan.executable
    with pytest.raises(CleanupConfirmationError, match="does not exactly match"):
        execute_cleanup_plan(
            engine,
            plan,
            confirmation_token="DELETE SOMETHING",
            project_root=tmp_path,
        )
    assert _count(engine, "embedding_512") == 2

    report = execute_cleanup_plan(
        engine,
        plan,
        confirmation_token=plan.confirmation_token or "",
        project_root=tmp_path,
    )

    assert report.total_deleted == 1
    assert _count(engine, "embedding_512") == 1
    assert _count(engine, "images") == 3
    with engine.connect() as connection:
        remaining_run_uids = connection.execute(
            select(Base.metadata.tables["embedding_512"].c.run_uid)
        ).scalars().all()
    assert remaining_run_uids == ["run-b"]

    audit_path = write_cleanup_audit(report, tmp_path / "audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["total_deleted"] == 1
    assert audit["run_uid"] == "run-a"


def test_transactional_repreview_rejects_changed_row_counts(tmp_path):
    engine = _engine()
    with Session(engine) as session:
        plan = build_cleanup_plan(
            session,
            scope_kind=SCOPE_EXACT_RUN_UID,
            run_uid="run-a",
            table_names=["embedding_512"],
            project_root=tmp_path,
        )
    with engine.begin() as connection:
        connection.execute(
            insert(Base.metadata.tables["embedding_512"]),
            {
                "id": 3,
                "image_id": 3,
                "run_uid": "run-a",
                "vector_type": "origin_512",
                "embedding": [0.0] * 512,
            },
        )

    with pytest.raises(CleanupPlanChangedError, match="changed after preview"):
        execute_cleanup_plan(
            engine,
            plan,
            confirmation_token=plan.confirmation_token or "",
            project_root=tmp_path,
        )

    assert _count(engine, "embedding_512") == 3


def test_completed_local_run_requires_an_explicit_override(tmp_path):
    engine = _engine()
    manifest_path = tmp_path / "runs" / "run-a" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"run_id": "run-a", "status": "completed"}),
        encoding="utf-8",
    )

    with Session(engine) as session:
        blocked = build_cleanup_plan(
            session,
            scope_kind=SCOPE_EXACT_RUN_UID,
            run_uid="run-a",
            table_names=["embedding_512"],
            project_root=tmp_path,
        )
        overridden = build_cleanup_plan(
            session,
            scope_kind=SCOPE_EXACT_RUN_UID,
            run_uid="run-a",
            table_names=["embedding_512"],
            allow_completed_run=True,
            project_root=tmp_path,
        )

    assert not blocked.executable
    assert any("completed run deletion is blocked" in item for item in blocked.blockers)
    assert overridden.executable
    assert any(
        "completed run protection was explicitly overridden" in item
        for item in overridden.warnings
    )


def test_legacy_null_scope_is_limited_to_nullable_image_embeddings(tmp_path):
    engine = _engine()
    with engine.begin() as connection:
        connection.execute(
            insert(Base.metadata.tables["embedding_512"]),
            {
                "id": 3,
                "image_id": 3,
                "run_uid": None,
                "vector_type": "origin_512",
                "embedding": [0.0] * 512,
            },
        )

    with Session(engine) as session:
        plan = build_cleanup_plan(
            session,
            scope_kind=SCOPE_LEGACY_NULL_RUN_UID,
            run_uid=None,
            table_names=["embedding_512"],
            project_root=tmp_path,
        )
        with pytest.raises(CleanupSelectionError, match="limited to image embedding"):
            build_cleanup_plan(
                session,
                scope_kind=SCOPE_LEGACY_NULL_RUN_UID,
                run_uid=None,
                table_names=["template_embedding_512"],
                project_root=tmp_path,
            )

    report = execute_cleanup_plan(
        engine,
        plan,
        confirmation_token=plan.confirmation_token or "",
        project_root=tmp_path,
    )
    assert report.total_deleted == 1
    assert _count(engine, "embedding_512") == 2


def test_research_run_record_selection_always_includes_fk_children(tmp_path):
    engine = _engine()
    with Session(engine) as session:
        plan = build_cleanup_plan(
            session,
            scope_kind=SCOPE_EXACT_RUN_UID,
            run_uid="run-a",
            include_research_run_record=True,
            project_root=tmp_path,
        )

    assert plan.selected_tables[-1] == "research_runs"
    assert "research_splits" in plan.selected_tables
    assert plan.total_rows == 2

    report = execute_cleanup_plan(
        engine,
        plan,
        confirmation_token=plan.confirmation_token or "",
        project_root=tmp_path,
    )
    assert report.total_deleted == 2
    assert _count(engine, "research_splits") == 0
    assert _count(engine, "research_runs") == 0
    assert _count(engine, "embedding_512") == 2


def test_empty_scope_can_be_an_explicit_verified_noop(tmp_path):
    engine = _engine()
    with Session(engine) as session:
        plan = build_cleanup_plan(
            session,
            scope_kind=SCOPE_EXACT_RUN_UID,
            run_uid="missing-run",
            table_names=["embedding_512"],
            allow_empty_scope=True,
            project_root=tmp_path,
        )

    assert plan.total_rows == 0
    assert plan.executable
    assert any("verified no-op" in item for item in plan.warnings)

    report = execute_cleanup_plan(
        engine,
        plan,
        confirmation_token=plan.confirmation_token or "",
        project_root=tmp_path,
    )
    assert report.total_deleted == 0
    assert _count(engine, "embedding_512") == 2


def test_before_commit_failure_rolls_back_database_rows(tmp_path):
    engine = _engine()
    with Session(engine) as session:
        plan = build_cleanup_plan(
            session,
            scope_kind=SCOPE_EXACT_RUN_UID,
            run_uid="run-a",
            table_names=["embedding_512"],
            project_root=tmp_path,
        )

    callbacks: list[str] = []

    def fail_before_commit() -> None:
        callbacks.append("called")
        raise RuntimeError("local quarantine failed")

    with pytest.raises(RuntimeError, match="local quarantine failed"):
        execute_cleanup_plan(
            engine,
            plan,
            confirmation_token=plan.confirmation_token or "",
            project_root=tmp_path,
            before_commit=fail_before_commit,
        )

    assert callbacks == ["called"]
    assert _count(engine, "embedding_512") == 2
