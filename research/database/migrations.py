from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine


_ADDITIVE_RESEARCH_SCHEMA_SQL = (
    "ALTER TABLE images ADD COLUMN IF NOT EXISTS content_sha256 VARCHAR(64)",
    "ALTER TABLE images ADD COLUMN IF NOT EXISTS file_size_bytes BIGINT",
    "ALTER TABLE embedding_512 ADD COLUMN IF NOT EXISTS run_uid VARCHAR(96)",
    "ALTER TABLE embedding_256 ADD COLUMN IF NOT EXISTS run_uid VARCHAR(96)",
    "ALTER TABLE embedding_pq ADD COLUMN IF NOT EXISTS run_uid VARCHAR(96)",
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_embedding_512_run_image_type "
        "ON embedding_512 (run_uid, image_id, vector_type) WHERE run_uid IS NOT NULL"
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_embedding_256_run_image_type "
        "ON embedding_256 (run_uid, image_id, vector_type) WHERE run_uid IS NOT NULL"
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_embedding_pq_run_image_type "
        "ON embedding_pq (run_uid, image_id, vector_type) WHERE run_uid IS NOT NULL"
    ),
    "CREATE INDEX IF NOT EXISTS ix_embedding_512_run_type ON embedding_512 (run_uid, vector_type)",
    "CREATE INDEX IF NOT EXISTS ix_embedding_256_run_type ON embedding_256 (run_uid, vector_type)",
    "CREATE INDEX IF NOT EXISTS ix_embedding_pq_run_type ON embedding_pq (run_uid, vector_type)",
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS run_uid VARCHAR(96)",
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS started_at TIMESTAMP",
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS finished_at TIMESTAMP",
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS metadata JSON",
    (
        "UPDATE research_runs SET run_uid = 'legacy-' || id::text "
        "WHERE run_uid IS NULL"
    ),
    "ALTER TABLE research_runs ALTER COLUMN run_uid SET NOT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_research_runs_run_uid ON research_runs (run_uid)",
    "ALTER TABLE research_runs DROP CONSTRAINT IF EXISTS research_runs_run_name_key",
    "ALTER TABLE research_search_results ADD COLUMN IF NOT EXISTS is_mated BOOLEAN",
    "ALTER TABLE research_search_results ADD COLUMN IF NOT EXISTS top1_correct BOOLEAN",
    "ALTER TABLE research_search_results ADD COLUMN IF NOT EXISTS accepted BOOLEAN",
)


def migrate_legacy_research_schema(bind: Engine | Connection) -> None:
    """Apply only additive/backward-compatible thesis2-to-thesis3 schema changes.

    Legacy tables and rows are intentionally retained. This function does not drop
    user data; it only adds thesis3 columns, backfills a stable legacy run UID, and
    removes the old run-name uniqueness constraint so repeated experiments are valid.
    """

    def _migrate(connection: Connection) -> None:
        for statement in _ADDITIVE_RESEARCH_SCHEMA_SQL:
            connection.execute(text(statement))

    if isinstance(bind, Engine):
        with bind.begin() as connection:
            _migrate(connection)
    else:
        _migrate(bind)
