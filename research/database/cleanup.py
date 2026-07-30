from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

from sqlalchemy import delete, func, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from research.database.models import Base


SCOPE_EXACT_RUN_UID = "exact_run_uid"
SCOPE_LEGACY_NULL_RUN_UID = "legacy_null_run_uid"
SUPPORTED_SCOPE_KINDS = frozenset(
    {SCOPE_EXACT_RUN_UID, SCOPE_LEGACY_NULL_RUN_UID}
)

IMAGE_EMBEDDING_TABLES = (
    "embedding_512",
    "embedding_448",
    "embedding_384",
    "embedding_256",
    "embedding_128",
    "embedding_64",
    "embedding_32",
    "embedding_pq",
)
TEMPLATE_EMBEDDING_TABLES = (
    "template_embedding_512",
    "template_embedding_448",
    "template_embedding_384",
    "template_embedding_256",
    "template_embedding_128",
    "template_embedding_64",
    "template_embedding_32",
)
RESEARCH_RUN_CHILD_TABLES = (
    "research_calibration_results",
    "research_search_results",
    "research_templates",
    "research_splits",
)
RESEARCH_RUN_TABLE = "research_runs"
PRESERVED_SHARED_TABLES = ("images",)

TABLE_GROUPS = {
    "image_embeddings": IMAGE_EMBEDDING_TABLES,
    "template_embeddings": TEMPLATE_EMBEDDING_TABLES,
    "research_run_children": RESEARCH_RUN_CHILD_TABLES,
    "all_run_scoped_data": (
        *IMAGE_EMBEDDING_TABLES,
        *TEMPLATE_EMBEDDING_TABLES,
        *RESEARCH_RUN_CHILD_TABLES,
    ),
}

SELECTABLE_TABLES = frozenset(
    {
        *IMAGE_EMBEDDING_TABLES,
        *TEMPLATE_EMBEDDING_TABLES,
        *RESEARCH_RUN_CHILD_TABLES,
    }
)
DIRECT_RUN_UID_TABLES = frozenset(
    {*IMAGE_EMBEDDING_TABLES, *TEMPLATE_EMBEDDING_TABLES}
)
RUN_SCOPED_TABLES = (
    *IMAGE_EMBEDDING_TABLES,
    *TEMPLATE_EMBEDDING_TABLES,
    *RESEARCH_RUN_CHILD_TABLES,
    RESEARCH_RUN_TABLE,
)

# All FK children precede research_runs. The remaining tables are independent.
DELETE_ORDER = (
    *RESEARCH_RUN_CHILD_TABLES,
    *TEMPLATE_EMBEDDING_TABLES,
    *IMAGE_EMBEDDING_TABLES,
    RESEARCH_RUN_TABLE,
)
PLAN_VERSION = 2
COMPLETED_STATUSES = frozenset({"complete", "completed"})


class DatabaseCleanupError(RuntimeError):
    """Base exception for guarded database cleanup."""


class CleanupSelectionError(DatabaseCleanupError, ValueError):
    """The requested scope or table selection is unsafe or unsupported."""


class CleanupSchemaError(DatabaseCleanupError):
    """The live database schema does not match the cleanup allowlist."""


class CleanupConfirmationError(DatabaseCleanupError):
    """The explicit confirmation token is missing or stale."""


class CleanupPlanChangedError(DatabaseCleanupError):
    """Rows or protection metadata changed after the preview was created."""


@dataclass(frozen=True)
class TableTotal:
    table_name: str
    row_count: int | None
    exists: bool
    cleanup_policy: str

    def as_dict(self) -> dict[str, object]:
        return {
            "table_name": self.table_name,
            "row_count": self.row_count,
            "exists": self.exists,
            "cleanup_policy": self.cleanup_policy,
        }


@dataclass(frozen=True)
class RunInventoryRow:
    table_name: str
    run_uid: str | None
    row_count: int
    research_run_status: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "table_name": self.table_name,
            "run_uid": self.run_uid,
            "row_count": self.row_count,
            "research_run_status": self.research_run_status,
        }


@dataclass(frozen=True)
class LocalRunStatus:
    status: str
    manifest_path: str

    def as_dict(self) -> dict[str, str]:
        return {
            "status": self.status,
            "manifest_path": self.manifest_path,
        }


@dataclass(frozen=True)
class CleanupTablePreview:
    table_name: str
    row_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "table_name": self.table_name,
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class CleanupPlan:
    database: str
    database_user: str
    scope_kind: str
    run_uid: str | None
    selected_tables: tuple[str, ...]
    table_rows: tuple[CleanupTablePreview, ...]
    research_run_status: str | None
    local_run_statuses: tuple[LocalRunStatus, ...]
    allow_completed_run: bool
    allow_empty_scope: bool
    warnings: tuple[str, ...]
    blockers: tuple[str, ...]
    plan_digest: str
    confirmation_token: str | None

    @property
    def total_rows(self) -> int:
        return sum(item.row_count for item in self.table_rows)

    @property
    def executable(self) -> bool:
        return self.confirmation_token is not None and not self.blockers

    def as_dict(self) -> dict[str, object]:
        return {
            "plan_version": PLAN_VERSION,
            "database": self.database,
            "database_user": self.database_user,
            "scope_kind": self.scope_kind,
            "run_uid": self.run_uid,
            "selected_tables": list(self.selected_tables),
            "table_rows": [item.as_dict() for item in self.table_rows],
            "total_rows": self.total_rows,
            "research_run_status": self.research_run_status,
            "local_run_statuses": [
                item.as_dict() for item in self.local_run_statuses
            ],
            "allow_completed_run": self.allow_completed_run,
            "allow_empty_scope": self.allow_empty_scope,
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
            "plan_digest": self.plan_digest,
            "confirmation_token": self.confirmation_token,
            "executable": self.executable,
        }


@dataclass(frozen=True)
class CleanupExecutionReport:
    committed_at_utc: str
    database: str
    database_user: str
    scope_kind: str
    run_uid: str | None
    plan_digest: str
    deleted_by_table: tuple[CleanupTablePreview, ...]

    @property
    def total_deleted(self) -> int:
        return sum(item.row_count for item in self.deleted_by_table)

    def as_dict(self) -> dict[str, object]:
        return {
            "committed_at_utc": self.committed_at_utc,
            "database": self.database,
            "database_user": self.database_user,
            "scope_kind": self.scope_kind,
            "run_uid": self.run_uid,
            "plan_digest": self.plan_digest,
            "deleted_by_table": [
                item.as_dict() for item in self.deleted_by_table
            ],
            "total_deleted": self.total_deleted,
        }


def resolve_cleanup_tables(
    *,
    table_groups: Sequence[str] = (),
    table_names: Sequence[str] = (),
    include_research_run_record: bool = False,
) -> tuple[str, ...]:
    unknown_groups = sorted(set(table_groups).difference(TABLE_GROUPS))
    if unknown_groups:
        raise CleanupSelectionError(
            f"unknown cleanup table groups: {unknown_groups}; "
            f"allowed={sorted(TABLE_GROUPS)}"
        )

    requested_names = set(table_names)
    if RESEARCH_RUN_TABLE in requested_names:
        raise CleanupSelectionError(
            "research_runs cannot be selected directly; set "
            "include_research_run_record=True so every FK child is included first"
        )
    forbidden = sorted(requested_names.difference(SELECTABLE_TABLES))
    if forbidden:
        raise CleanupSelectionError(
            f"tables are not in the cleanup allowlist: {forbidden}; "
            "images and arbitrary table names are never accepted"
        )

    selected = set(requested_names)
    for group_name in table_groups:
        selected.update(TABLE_GROUPS[group_name])
    if include_research_run_record:
        selected.update(RESEARCH_RUN_CHILD_TABLES)
        selected.add(RESEARCH_RUN_TABLE)
    if not selected:
        raise CleanupSelectionError(
            "select at least one allowlisted table group or table"
        )
    return tuple(name for name in DELETE_ORDER if name in selected)


def collect_table_totals(session: Session) -> tuple[TableTotal, ...]:
    existing = set(inspect(session.connection()).get_table_names())
    rows: list[TableTotal] = []
    for table in Base.metadata.sorted_tables:
        if table.name == "images":
            policy = "preserved_shared_source"
        elif table.name == RESEARCH_RUN_TABLE:
            policy = "exact_run_uid_only_with_fk_children"
        elif table.name in RUN_SCOPED_TABLES:
            policy = "guarded_run_scope"
        else:
            policy = "not_cleanup_allowlisted"
        if table.name not in existing:
            rows.append(
                TableTotal(
                    table_name=table.name,
                    row_count=None,
                    exists=False,
                    cleanup_policy=policy,
                )
            )
            continue
        count = int(
            session.execute(
                select(func.count()).select_from(table)
            ).scalar_one()
        )
        rows.append(
            TableTotal(
                table_name=table.name,
                row_count=count,
                exists=True,
                cleanup_policy=policy,
            )
        )
    return tuple(rows)


def collect_run_inventory(session: Session) -> tuple[RunInventoryRow, ...]:
    _require_live_tables(session, RUN_SCOPED_TABLES)
    research_runs = _table(RESEARCH_RUN_TABLE)
    db_statuses = {
        str(run_uid): str(status)
        for run_uid, status in session.execute(
            select(research_runs.c.run_uid, research_runs.c.status)
        )
    }
    rows: list[RunInventoryRow] = []

    for table_name in (*IMAGE_EMBEDDING_TABLES, *TEMPLATE_EMBEDDING_TABLES):
        table = _table(table_name)
        statement = (
            select(table.c.run_uid, func.count())
            .select_from(table)
            .group_by(table.c.run_uid)
        )
        for run_uid, row_count in session.execute(statement):
            normalized_uid = None if run_uid is None else str(run_uid)
            rows.append(
                RunInventoryRow(
                    table_name=table_name,
                    run_uid=normalized_uid,
                    row_count=int(row_count),
                    research_run_status=(
                        db_statuses.get(normalized_uid)
                        if normalized_uid is not None
                        else None
                    ),
                )
            )

    for table_name in RESEARCH_RUN_CHILD_TABLES:
        table = _table(table_name)
        statement = (
            select(research_runs.c.run_uid, func.count())
            .select_from(
                table.join(
                    research_runs,
                    table.c.run_id == research_runs.c.id,
                )
            )
            .group_by(research_runs.c.run_uid)
        )
        for run_uid, row_count in session.execute(statement):
            normalized_uid = str(run_uid)
            rows.append(
                RunInventoryRow(
                    table_name=table_name,
                    run_uid=normalized_uid,
                    row_count=int(row_count),
                    research_run_status=db_statuses.get(normalized_uid),
                )
            )

    for run_uid, status in session.execute(
        select(research_runs.c.run_uid, research_runs.c.status)
    ):
        rows.append(
            RunInventoryRow(
                table_name=RESEARCH_RUN_TABLE,
                run_uid=str(run_uid),
                row_count=1,
                research_run_status=str(status),
            )
        )

    return tuple(
        sorted(
            rows,
            key=lambda item: (
                "" if item.run_uid is None else item.run_uid,
                DELETE_ORDER.index(item.table_name),
            ),
        )
    )


def build_cleanup_plan(
    session: Session,
    *,
    scope_kind: str,
    run_uid: str | None,
    table_groups: Sequence[str] = (),
    table_names: Sequence[str] = (),
    include_research_run_record: bool = False,
    allow_completed_run: bool = False,
    allow_empty_scope: bool = False,
    project_root: str | Path | None = None,
) -> CleanupPlan:
    selected_tables = resolve_cleanup_tables(
        table_groups=table_groups,
        table_names=table_names,
        include_research_run_record=include_research_run_record,
    )
    normalized_run_uid = _validate_scope(
        scope_kind=scope_kind,
        run_uid=run_uid,
        selected_tables=selected_tables,
        include_research_run_record=include_research_run_record,
    )
    required_tables = {*selected_tables}
    if scope_kind == SCOPE_EXACT_RUN_UID:
        required_tables.add(RESEARCH_RUN_TABLE)
    _require_live_tables(session, required_tables)

    database, database_user = _database_identity(session)
    table_rows = tuple(
        CleanupTablePreview(
            table_name=table_name,
            row_count=_count_scope_rows(
                session,
                table_name=table_name,
                scope_kind=scope_kind,
                run_uid=normalized_run_uid,
            ),
        )
        for table_name in selected_tables
    )

    research_run_status: str | None = None
    local_run_statuses: tuple[LocalRunStatus, ...] = ()
    warnings: list[str] = []
    blockers: list[str] = []
    if scope_kind == SCOPE_EXACT_RUN_UID:
        research_run_status = session.execute(
            select(_table(RESEARCH_RUN_TABLE).c.status).where(
                _table(RESEARCH_RUN_TABLE).c.run_uid == normalized_run_uid
            )
        ).scalar_one_or_none()
        if research_run_status is not None:
            research_run_status = str(research_run_status)
        else:
            warnings.append(
                "matching research_runs record was not found; only string run_uid "
                "lineage is available in the selected tables"
            )
        if project_root is not None:
            local_run_statuses = _find_local_run_statuses(
                Path(project_root),
                normalized_run_uid,
            )
            if not local_run_statuses:
                warnings.append(
                    "matching local run_manifest.json was not found"
                )

        protected_sources: list[str] = []
        if _is_completed(research_run_status):
            protected_sources.append(
                f"research_runs.status={research_run_status}"
            )
        for item in local_run_statuses:
            if _is_completed(item.status):
                protected_sources.append(
                    f"{item.manifest_path}: status={item.status}"
                )
            elif item.status == "unreadable":
                blockers.append(
                    f"target local run manifest is unreadable: {item.manifest_path}"
                )
        if protected_sources and not allow_completed_run:
            blockers.append(
                "completed run deletion is blocked; verify the immutable run policy "
                "and set allow_completed_run=True only for an intentional override "
                f"({'; '.join(protected_sources)})"
            )
        elif protected_sources:
            warnings.append(
                "completed run protection was explicitly overridden: "
                + "; ".join(protected_sources)
            )

    total_rows = sum(item.row_count for item in table_rows)
    if total_rows == 0 and not allow_empty_scope:
        blockers.append("the selected scope matches zero rows")
    elif total_rows == 0:
        warnings.append(
            "the selected DB scope matches zero rows; the DB phase will be a "
            "verified no-op"
        )

    digest_payload = {
        "plan_version": PLAN_VERSION,
        "database": database,
        "database_user": database_user,
        "scope_kind": scope_kind,
        "run_uid": normalized_run_uid,
        "selected_tables": list(selected_tables),
        "table_rows": [item.as_dict() for item in table_rows],
        "research_run_status": research_run_status,
        "local_run_statuses": [
            item.as_dict() for item in local_run_statuses
        ],
        "allow_completed_run": bool(allow_completed_run),
        "allow_empty_scope": bool(allow_empty_scope),
    }
    plan_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    confirmation_token = None
    if not blockers:
        if scope_kind == SCOPE_EXACT_RUN_UID:
            target_label = normalized_run_uid
        else:
            target_label = "NULL_RUN_UID"
        confirmation_token = (
            f"DELETE {total_rows} ROWS FOR {target_label} "
            f"{plan_digest[:12]}"
        )

    return CleanupPlan(
        database=database,
        database_user=database_user,
        scope_kind=scope_kind,
        run_uid=normalized_run_uid,
        selected_tables=selected_tables,
        table_rows=table_rows,
        research_run_status=research_run_status,
        local_run_statuses=local_run_statuses,
        allow_completed_run=bool(allow_completed_run),
        allow_empty_scope=bool(allow_empty_scope),
        warnings=tuple(warnings),
        blockers=tuple(blockers),
        plan_digest=plan_digest,
        confirmation_token=confirmation_token,
    )


def execute_cleanup_plan(
    engine: Engine,
    plan: CleanupPlan,
    *,
    confirmation_token: str,
    project_root: str | Path | None = None,
    before_commit: Callable[[], None] | None = None,
) -> CleanupExecutionReport:
    if plan.blockers or not plan.executable:
        raise CleanupConfirmationError(
            "cleanup plan is blocked and cannot be executed; rebuild the preview "
            "after resolving every blocker"
        )
    if confirmation_token != plan.confirmation_token:
        raise CleanupConfirmationError(
            "confirmation token does not exactly match the latest preview"
        )

    include_research_run_record = RESEARCH_RUN_TABLE in plan.selected_tables
    selectable_names = tuple(
        name
        for name in plan.selected_tables
        if name != RESEARCH_RUN_TABLE
    )
    deleted_rows: list[CleanupTablePreview] = []
    with Session(engine) as session:
        with session.begin():
            _lock_cleanup_tables(session, plan.selected_tables)
            refreshed = build_cleanup_plan(
                session,
                scope_kind=plan.scope_kind,
                run_uid=plan.run_uid,
                table_names=selectable_names,
                include_research_run_record=include_research_run_record,
                allow_completed_run=plan.allow_completed_run,
                allow_empty_scope=plan.allow_empty_scope,
                project_root=project_root,
            )
            if refreshed.plan_digest != plan.plan_digest:
                raise CleanupPlanChangedError(
                    "database rows, database identity, or run protection metadata "
                    "changed after preview; transaction rolled back, create a new plan"
                )
            if confirmation_token != refreshed.confirmation_token:
                raise CleanupPlanChangedError(
                    "confirmation token is stale after the transactional re-preview"
                )

            expected_by_table = {
                item.table_name: item.row_count for item in refreshed.table_rows
            }
            for table_name in refreshed.selected_tables:
                statement = delete(_table(table_name)).where(
                    _scope_predicate(
                        table_name=table_name,
                        scope_kind=refreshed.scope_kind,
                        run_uid=refreshed.run_uid,
                    )
                )
                result = session.execute(statement)
                expected = expected_by_table[table_name]
                if result.rowcount is not None and result.rowcount >= 0:
                    if int(result.rowcount) != expected:
                        raise CleanupPlanChangedError(
                            f"{table_name} deleted {result.rowcount} rows; "
                            f"preview expected {expected}; transaction rolled back"
                        )
                remaining = _count_scope_rows(
                    session,
                    table_name=table_name,
                    scope_kind=refreshed.scope_kind,
                    run_uid=refreshed.run_uid,
                )
                if remaining:
                    raise CleanupPlanChangedError(
                        f"{table_name} still has {remaining} matching rows; "
                        "transaction rolled back"
                    )
                deleted_rows.append(
                    CleanupTablePreview(
                        table_name=table_name,
                        row_count=expected,
                    )
                )
            if before_commit is not None:
                before_commit()

    return CleanupExecutionReport(
        committed_at_utc=datetime.now(timezone.utc).isoformat(),
        database=plan.database,
        database_user=plan.database_user,
        scope_kind=plan.scope_kind,
        run_uid=plan.run_uid,
        plan_digest=plan.plan_digest,
        deleted_by_table=tuple(deleted_rows),
    )


def write_cleanup_audit(
    report: CleanupExecutionReport,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = report.committed_at_utc.replace(":", "").replace("+", "_")
    path = destination / (
        f"{timestamp}_{report.plan_digest[:12]}_database_cleanup.json"
    )
    if path.exists():
        raise FileExistsError(f"cleanup audit already exists: {path}")
    path.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _table(table_name: str):
    return Base.metadata.tables[table_name]


def _require_live_tables(
    session: Session,
    table_names: Iterable[str],
) -> None:
    existing = set(inspect(session.connection()).get_table_names())
    missing = sorted(set(table_names).difference(existing))
    if missing:
        raise CleanupSchemaError(
            f"live database is missing cleanup tables: {missing}; "
            "do not delete against a partially migrated schema"
        )


def _validate_scope(
    *,
    scope_kind: str,
    run_uid: str | None,
    selected_tables: Sequence[str],
    include_research_run_record: bool,
) -> str | None:
    if scope_kind not in SUPPORTED_SCOPE_KINDS:
        raise CleanupSelectionError(
            f"unsupported scope_kind={scope_kind!r}; "
            f"allowed={sorted(SUPPORTED_SCOPE_KINDS)}"
        )
    if scope_kind == SCOPE_EXACT_RUN_UID:
        normalized = "" if run_uid is None else str(run_uid).strip()
        if not normalized:
            raise CleanupSelectionError(
                "run_uid is required for exact_run_uid cleanup"
            )
        if len(normalized) > 96:
            raise CleanupSelectionError(
                "run_uid exceeds the database limit of 96 characters"
            )
        return normalized

    if run_uid not in (None, ""):
        raise CleanupSelectionError(
            "run_uid must be empty when scope_kind='legacy_null_run_uid'"
        )
    unsupported = sorted(
        set(selected_tables).difference(IMAGE_EMBEDDING_TABLES)
    )
    if unsupported or include_research_run_record:
        raise CleanupSelectionError(
            "legacy_null_run_uid is limited to image embedding tables with a "
            f"nullable run_uid; unsupported={unsupported}"
        )
    return None


def _scope_predicate(
    *,
    table_name: str,
    scope_kind: str,
    run_uid: str | None,
):
    table = _table(table_name)
    if table_name in DIRECT_RUN_UID_TABLES:
        if scope_kind == SCOPE_LEGACY_NULL_RUN_UID:
            return table.c.run_uid.is_(None)
        return table.c.run_uid == run_uid
    if scope_kind != SCOPE_EXACT_RUN_UID:
        raise CleanupSelectionError(
            f"{table_name} does not support scope_kind={scope_kind}"
        )
    research_runs = _table(RESEARCH_RUN_TABLE)
    if table_name == RESEARCH_RUN_TABLE:
        return research_runs.c.run_uid == run_uid
    if table_name in RESEARCH_RUN_CHILD_TABLES:
        run_ids = select(research_runs.c.id).where(
            research_runs.c.run_uid == run_uid
        )
        return table.c.run_id.in_(run_ids)
    raise CleanupSelectionError(
        f"{table_name} has no guarded cleanup predicate"
    )


def _count_scope_rows(
    session: Session,
    *,
    table_name: str,
    scope_kind: str,
    run_uid: str | None,
) -> int:
    table = _table(table_name)
    return int(
        session.execute(
            select(func.count())
            .select_from(table)
            .where(
                _scope_predicate(
                    table_name=table_name,
                    scope_kind=scope_kind,
                    run_uid=run_uid,
                )
            )
        ).scalar_one()
    )


def _database_identity(session: Session) -> tuple[str, str]:
    bind = session.get_bind()
    if bind.dialect.name == "postgresql":
        database, database_user = session.execute(
            text("SELECT current_database(), current_user")
        ).one()
        return str(database), str(database_user)
    engine = bind.engine if hasattr(bind, "engine") else bind
    url = getattr(engine, "url", None)
    database = str(getattr(url, "database", None) or bind.dialect.name)
    database_user = str(getattr(url, "username", None) or "")
    return database, database_user


def _find_local_run_statuses(
    project_root: Path,
    run_uid: str,
) -> tuple[LocalRunStatus, ...]:
    root = project_root.resolve()
    runs_root = root / "runs"
    if not runs_root.exists():
        return ()
    found: list[LocalRunStatus] = []
    for manifest_path in runs_root.rglob("run_manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            if run_uid in str(manifest_path):
                found.append(
                    LocalRunStatus(
                        status="unreadable",
                        manifest_path=_relative_display_path(
                            manifest_path,
                            root,
                        ),
                    )
                )
            continue
        if str(payload.get("run_id", "")) != run_uid:
            continue
        found.append(
            LocalRunStatus(
                status=str(payload.get("status", "unknown")).strip().lower(),
                manifest_path=_relative_display_path(manifest_path, root),
            )
        )
    return tuple(
        sorted(found, key=lambda item: (item.manifest_path, item.status))
    )


def _relative_display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _is_completed(status: str | None) -> bool:
    return status is not None and status.strip().lower() in COMPLETED_STATUSES


def _lock_cleanup_tables(
    session: Session,
    selected_tables: Sequence[str],
) -> None:
    if session.get_bind().dialect.name != "postgresql":
        return
    lock_names = sorted({*selected_tables, RESEARCH_RUN_TABLE})
    for table_name in lock_names:
        if table_name not in Base.metadata.tables:
            raise CleanupSelectionError(
                f"cannot lock a non-allowlisted table: {table_name}"
            )
        session.execute(
            text(
                f'LOCK TABLE "{table_name}" IN SHARE ROW EXCLUSIVE MODE'
            )
        )
