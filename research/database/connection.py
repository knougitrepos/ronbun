from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session, sessionmaker

from research.database.models import Base
from research.database.migrations import migrate_legacy_research_schema
from research.database.settings import DatabaseSettings, load_database_settings


EXPECTED_TABLES = set(Base.metadata.tables.keys())


def create_database_engine(settings: DatabaseSettings | None = None) -> Engine:
    resolved = settings or load_database_settings()
    return create_engine(
        resolved.sqlalchemy_url,
        echo=resolved.echo_sql,
        hide_parameters=True,
        pool_pre_ping=resolved.pool_pre_ping,
        connect_args={"connect_timeout": resolved.connect_timeout_seconds},
    )


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    session = sessionmaker(bind=engine, autocommit=False, autoflush=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _with_connection(bind: Engine | Connection, fn):
    if isinstance(bind, Engine):
        with bind.connect() as conn:
            return fn(conn)
    return fn(bind)


def ensure_vector_extension(bind: Engine | Connection) -> None:
    if isinstance(bind, Engine):
        with bind.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    else:
        bind.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def ensure_vector_indexes(bind: Engine | Connection) -> None:
    statements = [
        """
        CREATE INDEX IF NOT EXISTS ix_embedding_512_embedding_hnsw_cosine
        ON embedding_512 USING hnsw (embedding vector_cosine_ops)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_template_embedding_512_hnsw_cosine
        ON template_embedding_512 USING hnsw (embedding vector_cosine_ops)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_template_embedding_512_scope
        ON template_embedding_512
        (run_uid, protocol_name, vector_type, aggregation_method, enrollment_policy, enrollment_target, model_uid)
        """,
    ]
    for dimension in (32, 64, 128, 256, 384, 448):
        statements.extend(
            [
                f"""
                CREATE INDEX IF NOT EXISTS ix_embedding_{dimension}_embedding_hnsw_cosine
                ON embedding_{dimension} USING hnsw (embedding vector_cosine_ops)
                """,
                f"""
                CREATE INDEX IF NOT EXISTS ix_template_embedding_{dimension}_hnsw_cosine
                ON template_embedding_{dimension} USING hnsw (embedding vector_cosine_ops)
                """,
                f"""
                CREATE INDEX IF NOT EXISTS ix_template_embedding_{dimension}_scope
                ON template_embedding_{dimension}
                (run_uid, protocol_name, vector_type, aggregation_method, enrollment_policy, enrollment_target, model_uid)
                """,
            ]
        )

    def _create(conn: Connection) -> None:
        for statement in statements:
            conn.execute(text(statement))

    if isinstance(bind, Engine):
        with bind.begin() as conn:
            _create(conn)
    else:
        _create(bind)


def ensure_database_schema(bind: Engine | Connection) -> None:
    """Create/migrate tables without eagerly building optional vector indexes."""

    ensure_vector_extension(bind)
    Base.metadata.create_all(bind=bind)
    migrate_legacy_research_schema(bind)


def init_database(bind: Engine | Connection) -> None:
    ensure_database_schema(bind)
    ensure_vector_indexes(bind)


def check_database_health(bind: Engine | Connection) -> dict[str, object]:
    def _check(conn: Connection) -> dict[str, object]:
        database, user, server_version = conn.execute(
            text("SELECT current_database(), current_user, version()")
        ).one()
        vector_version = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        inspector = inspect(conn)
        existing_tables = set(inspector.get_table_names())
        schema_issues: list[str] = []
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            live_columns = {
                column["name"]: column for column in inspector.get_columns(table.name)
            }
            for column in table.columns:
                live = live_columns.get(column.name)
                if live is None:
                    schema_issues.append(f"{table.name}.{column.name}: missing column")
                elif bool(live["nullable"]) != bool(column.nullable):
                    schema_issues.append(
                        f"{table.name}.{column.name}: nullable={live['nullable']} "
                        f"expected={column.nullable}"
                    )
            live_unique_items = list(inspector.get_unique_constraints(table.name))
            live_unique_items.extend(
                item
                for item in inspector.get_indexes(table.name)
                if item.get("unique")
            )
            for constraint in table.constraints:
                if constraint.__class__.__name__ != "UniqueConstraint":
                    continue
                columns = tuple(sorted(constraint.columns.keys()))
                matches = [
                    item
                    for item in live_unique_items
                    if item.get("column_names")
                    and tuple(sorted(item["column_names"])) == columns
                ]
                if not matches:
                    schema_issues.append(
                        f"{table.name}: missing unique constraint on {','.join(columns)}"
                    )
                    continue
                expected_nulls_not_distinct = bool(
                    constraint.dialect_options["postgresql"].get(
                        "nulls_not_distinct", False
                    )
                )
                if expected_nulls_not_distinct and not any(
                    bool(
                        item.get("dialect_options", {}).get(
                            "postgresql_nulls_not_distinct", False
                        )
                    )
                    for item in matches
                ):
                    schema_issues.append(
                        f"{table.name}: unique constraint on {','.join(columns)} "
                        "must use NULLS NOT DISTINCT"
                    )
        return {
            "database": database,
            "user": user,
            "server_version": server_version,
            "vector_extension_version": vector_version,
            "existing_tables": sorted(existing_tables),
            "missing_tables": sorted(EXPECTED_TABLES.difference(existing_tables)),
            "schema_issues": schema_issues,
        }

    return _with_connection(bind, _check)
