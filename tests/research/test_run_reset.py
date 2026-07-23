from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, insert, select
from sqlalchemy.orm import Session

import research.database.reset as reset_module
from research.database.cleanup import RUN_SCOPED_TABLES
from research.database.models import Base
from research.database.reset import (
    RunResetPlanChangedError,
    build_run_reset_plan,
    execute_run_reset_plan,
)


RUN_A = "run-a"
RUN_B = "run-b"


def _engine(*, populate: bool = True):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    if not populate:
        return engine

    tables = Base.metadata.tables
    with engine.begin() as connection:
        connection.execute(
            insert(tables["images"]),
            [
                {"id": 1, "image_path": "a.jpg"},
                {"id": 2, "image_path": "b.jpg"},
            ],
        )
        connection.execute(
            insert(tables["research_runs"]),
            [
                {
                    "id": 1,
                    "run_uid": RUN_A,
                    "run_name": "run a",
                    "config_hash": "a" * 64,
                    "config": {},
                    "status": "running",
                    "artifact_dir": "runs/run-a",
                },
                {
                    "id": 2,
                    "run_uid": RUN_B,
                    "run_name": "run b",
                    "config_hash": "b" * 64,
                    "config": {},
                    "status": "running",
                    "artifact_dir": "runs/run-b",
                },
            ],
        )
        for index, table_name in enumerate(
            (
                "embedding_512",
                "embedding_448",
                "embedding_384",
                "embedding_256",
                "embedding_128",
                "embedding_pq",
            ),
            start=1,
        ):
            for run_index, run_uid in enumerate((RUN_A, RUN_B)):
                row = {
                    "id": 10 * index + run_index,
                    "image_id": run_index + 1,
                    "run_uid": run_uid,
                    "vector_type": table_name,
                }
                if table_name == "embedding_pq":
                    row["codes"] = b"\x00"
                else:
                    dimension = int(table_name.split("_")[1])
                    row["embedding"] = [0.0] * dimension
                connection.execute(insert(tables[table_name]), row)

        for index, table_name in enumerate(
            (
                "template_embedding_512",
                "template_embedding_448",
                "template_embedding_384",
                "template_embedding_256",
                "template_embedding_128",
            ),
            start=1,
        ):
            dimension = int(table_name.rsplit("_", 1)[1])
            for run_index, run_uid in enumerate((RUN_A, RUN_B)):
                connection.execute(
                    insert(tables[table_name]),
                    {
                        "id": 100 * index + run_index,
                        "run_uid": run_uid,
                        "protocol_name": "test",
                        "vector_type": table_name,
                        "aggregation_method": "mean",
                        "enrollment_policy": "fixed",
                        "enrollment_target": 1,
                        "enrollment_count": 1,
                        "identity_id": f"person-{run_index}",
                        "model_uid": "model",
                        "source_image_ids": [run_index + 1],
                        "embedding": [0.0] * dimension,
                    },
                )

        for run_id in (1, 2):
            connection.execute(
                insert(tables["research_splits"]),
                {
                    "id": run_id,
                    "run_id": run_id,
                    "split_name": "test",
                    "identity_id": f"person-{run_id}",
                    "role": "gallery",
                },
            )
            connection.execute(
                insert(tables["research_templates"]),
                {
                    "id": run_id,
                    "run_id": run_id,
                    "identity_id": f"person-{run_id}",
                    "compression_profile": "origin_512",
                    "aggregation_method": "mean",
                    "enrollment_count": 1,
                },
            )
            connection.execute(
                insert(tables["research_search_results"]),
                {
                    "id": run_id,
                    "run_id": run_id,
                    "query_id": f"query-{run_id}",
                    "probe_type": "mated",
                    "compression_profile": "origin_512",
                    "y_true_accept": True,
                },
            )
            connection.execute(
                insert(tables["research_calibration_results"]),
                {
                    "id": run_id,
                    "run_id": run_id,
                    "model_name": "arcface",
                    "compression_profile": "origin_512",
                    "target_fpir": 0.01,
                },
            )
    return engine


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _make_local_state(
    root: Path,
    *,
    run_uid: str = RUN_A,
    status: str = "running",
    promoted: bool = False,
) -> dict[str, Path]:
    run_dir = root / "runs" / "lfw" / f"{run_uid}_workspace"
    _write_json(
        run_dir / "run_manifest.json",
        {"run_id": run_uid, "status": status},
    )
    (run_dir / "artifacts" / "embedding.npy").parent.mkdir(parents=True)
    (run_dir / "artifacts" / "embedding.npy").write_bytes(b"embedding")

    result_root = (
        root / "results" / ("paper" if promoted else "step2") / "lfw" / run_uid
    )
    _write_json(
        result_root / "result_manifest.json",
        {"run_id": run_uid, "files": {}},
    )
    (result_root / "metrics.json").write_text("{}", encoding="utf-8")

    pointer = root / "runs" / "lfw" / "active_run.json"
    _write_json(pointer, {"run_id": run_uid, "run_dir": "D:/stale/path"})

    shared_raw = root / "data" / "raw" / "lfw" / "source.jpg"
    shared_raw.parent.mkdir(parents=True)
    shared_raw.write_bytes(b"raw")
    shared_interim = root / "data" / "interim" / "lfw" / "face_manifest.csv"
    shared_interim.parent.mkdir(parents=True)
    shared_interim.write_text("image_path\nsource.jpg\n", encoding="utf-8")
    return {
        "run_dir": run_dir,
        "result_root": result_root,
        "pointer": pointer,
        "shared_raw": shared_raw,
        "shared_interim": shared_interim,
    }


def _count_scope(engine, table_name: str, run_uid: str) -> int:
    table = Base.metadata.tables[table_name]
    with engine.connect() as connection:
        if table_name == "images":
            statement = select(func.count()).select_from(table)
        elif "run_uid" in table.c:
            statement = (
                select(func.count())
                .select_from(table)
                .where(table.c.run_uid == run_uid)
            )
        elif table_name == "research_runs":
            statement = (
                select(func.count())
                .select_from(table)
                .where(table.c.run_uid == run_uid)
            )
        else:
            research_runs = Base.metadata.tables["research_runs"]
            statement = (
                select(func.count())
                .select_from(table.join(research_runs))
                .where(research_runs.c.run_uid == run_uid)
            )
        return int(connection.execute(statement).scalar_one())


def test_complete_run_reset_closes_db_and_owned_local_state(tmp_path):
    engine = _engine()
    local = _make_local_state(tmp_path)

    with Session(engine) as session:
        plan = build_run_reset_plan(
            session,
            run_uid=RUN_A,
            project_root=tmp_path,
        )

    assert plan.executable
    assert plan.total_database_rows == 16
    assert set(plan.database_plan.selected_tables) == set(RUN_SCOPED_TABLES)
    assert {item.kind for item in plan.local_targets} == {
        "run_workspace",
        "result_bundle",
        "active_run_pointer",
    }
    assert plan.confirmation_token.startswith(
        f"RESET {RUN_A} DB_ROWS=16 FILES="
    )

    report = execute_run_reset_plan(
        engine,
        plan,
        confirmation_token=plan.confirmation_token or "",
        project_root=tmp_path,
    )

    for table_name in RUN_SCOPED_TABLES:
        assert _count_scope(engine, table_name, RUN_A) == 0
        assert _count_scope(engine, table_name, RUN_B) == 1
    assert _count_scope(engine, "images", RUN_A) == 2
    assert not local["run_dir"].exists()
    assert not local["result_root"].exists()
    assert not local["pointer"].exists()
    assert local["shared_raw"].is_file()
    assert local["shared_interim"].is_file()
    assert report.audit_path is not None
    assert (tmp_path / report.audit_path).is_file()
    for item in report.quarantined_targets:
        assert (tmp_path / item.quarantine_relative_path).exists()


def test_local_only_reset_uses_verified_noop_db_phase(tmp_path):
    engine = _engine(populate=False)
    local = _make_local_state(tmp_path)

    with Session(engine) as session:
        plan = build_run_reset_plan(
            session,
            run_uid=RUN_A,
            project_root=tmp_path,
        )

    assert plan.total_database_rows == 0
    assert plan.executable
    report = execute_run_reset_plan(
        engine,
        plan,
        confirmation_token=plan.confirmation_token or "",
        project_root=tmp_path,
    )
    assert report.database_report.total_deleted == 0
    assert not local["run_dir"].exists()


def test_promoted_result_and_completed_run_require_independent_overrides(tmp_path):
    engine = _engine()
    _make_local_state(tmp_path, status="completed", promoted=True)

    with Session(engine) as session:
        blocked = build_run_reset_plan(
            session,
            run_uid=RUN_A,
            project_root=tmp_path,
        )
        completed_only = build_run_reset_plan(
            session,
            run_uid=RUN_A,
            project_root=tmp_path,
            allow_completed_run=True,
        )
        allowed = build_run_reset_plan(
            session,
            run_uid=RUN_A,
            project_root=tmp_path,
            allow_completed_run=True,
            allow_promoted_results=True,
        )

    assert not blocked.executable
    assert not completed_only.executable
    assert any("completed run deletion is blocked" in x for x in blocked.blockers)
    assert any("promoted result reset is blocked" in x for x in blocked.blockers)
    assert allowed.executable


def test_orphan_db_lineage_requires_explicit_override(tmp_path):
    engine = _engine(populate=False)
    tables = Base.metadata.tables
    with engine.begin() as connection:
        connection.execute(
            insert(tables["images"]),
            {"id": 1, "image_path": "orphan.jpg"},
        )
        connection.execute(
            insert(tables["embedding_512"]),
            {
                "id": 1,
                "image_id": 1,
                "run_uid": "orphan",
                "vector_type": "origin_512",
                "embedding": [0.0] * 512,
            },
        )

    with Session(engine) as session:
        blocked = build_run_reset_plan(
            session,
            run_uid="orphan",
            project_root=tmp_path,
        )
        allowed = build_run_reset_plan(
            session,
            run_uid="orphan",
            project_root=tmp_path,
            allow_unverified_lineage=True,
        )

    assert not blocked.executable
    assert any("no research_runs record" in item for item in blocked.blockers)
    assert allowed.executable


def test_changed_local_artifact_invalidates_preview_without_mutation(tmp_path):
    engine = _engine()
    local = _make_local_state(tmp_path)
    with Session(engine) as session:
        plan = build_run_reset_plan(
            session,
            run_uid=RUN_A,
            project_root=tmp_path,
        )

    (local["run_dir"] / "artifacts" / "later.bin").write_bytes(b"changed")
    with pytest.raises(RunResetPlanChangedError, match="changed after preview"):
        execute_run_reset_plan(
            engine,
            plan,
            confirmation_token=plan.confirmation_token or "",
            project_root=tmp_path,
        )

    assert local["run_dir"].is_dir()
    assert _count_scope(engine, "embedding_512", RUN_A) == 1


def test_quarantine_failure_restores_files_and_rolls_back_db(
    tmp_path,
    monkeypatch,
):
    engine = _engine()
    local = _make_local_state(tmp_path)
    with Session(engine) as session:
        plan = build_run_reset_plan(
            session,
            run_uid=RUN_A,
            project_root=tmp_path,
        )

    real_rename = Path.rename
    forward_moves = 0

    def fail_second_forward_move(source: Path, destination: Path):
        nonlocal forward_moves
        if "payload" in destination.parts:
            forward_moves += 1
            if forward_moves == 2:
                raise OSError("simulated quarantine failure")
        return real_rename(source, destination)

    monkeypatch.setattr(reset_module, "_rename_path", fail_second_forward_move)
    with pytest.raises(OSError, match="simulated quarantine failure"):
        execute_run_reset_plan(
            engine,
            plan,
            confirmation_token=plan.confirmation_token or "",
            project_root=tmp_path,
        )

    assert local["run_dir"].is_dir()
    assert local["result_root"].is_dir()
    assert local["pointer"].is_file()
    assert _count_scope(engine, "embedding_512", RUN_A) == 1


def test_symlink_inside_owned_run_is_blocked_when_supported(tmp_path):
    engine = _engine()
    local = _make_local_state(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = local["run_dir"] / "artifacts" / "outside-link"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")

    with Session(engine) as session:
        plan = build_run_reset_plan(
            session,
            run_uid=RUN_A,
            project_root=tmp_path,
        )

    assert not plan.executable
    assert any("symbolic links" in item for item in plan.blockers)
    assert outside.read_text(encoding="utf-8") == "outside"
