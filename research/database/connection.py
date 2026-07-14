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
    statements = (
        """
        CREATE INDEX IF NOT EXISTS ix_embedding_512_embedding_hnsw_cosine
        ON embedding_512 USING hnsw (embedding vector_cosine_ops)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_embedding_256_embedding_hnsw_cosine
        ON embedding_256 USING hnsw (embedding vector_cosine_ops)
        """,
    )

    def _create(conn: Connection) -> None:
        for statement in statements:
            conn.execute(text(statement))

    if isinstance(bind, Engine):
        with bind.begin() as conn:
            _create(conn)
    else:
        _create(bind)


def init_database(bind: Engine | Connection) -> None:
    ensure_vector_extension(bind)
    Base.metadata.create_all(bind=bind)
    migrate_legacy_research_schema(bind)
    ensure_vector_indexes(bind)


def check_database_health(bind: Engine | Connection) -> dict[str, object]:
    def _check(conn: Connection) -> dict[str, object]:
        database, user, server_version = conn.execute(
            text("SELECT current_database(), current_user, version()")
        ).one()
        vector_version = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        existing_tables = set(inspect(conn).get_table_names())
        return {
            "database": database,
            "user": user,
            "server_version": server_version,
            "vector_extension_version": vector_version,
            "existing_tables": sorted(existing_tables),
            "missing_tables": sorted(EXPECTED_TABLES.difference(existing_tables)),
        }

    return _with_connection(bind, _check)
