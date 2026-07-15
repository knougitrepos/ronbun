from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine


_ADDITIVE_RESEARCH_SCHEMA_SQL = (
    "ALTER TABLE images ADD COLUMN IF NOT EXISTS content_sha256 VARCHAR(64)",
    "ALTER TABLE images ADD COLUMN IF NOT EXISTS file_size_bytes BIGINT",
    "ALTER TABLE embedding_512 ADD COLUMN IF NOT EXISTS run_uid VARCHAR(96)",
    "ALTER TABLE embedding_448 ADD COLUMN IF NOT EXISTS run_uid VARCHAR(96)",
    "ALTER TABLE embedding_448 ADD COLUMN IF NOT EXISTS vector_type VARCHAR(32)",
    "UPDATE embedding_448 SET vector_type = 'pca_448' WHERE vector_type IS NULL",
    "ALTER TABLE embedding_448 ALTER COLUMN vector_type SET NOT NULL",
    "ALTER TABLE embedding_384 ADD COLUMN IF NOT EXISTS run_uid VARCHAR(96)",
    "ALTER TABLE embedding_384 ADD COLUMN IF NOT EXISTS vector_type VARCHAR(32)",
    "UPDATE embedding_384 SET vector_type = 'pca_384' WHERE vector_type IS NULL",
    "ALTER TABLE embedding_384 ALTER COLUMN vector_type SET NOT NULL",
    "ALTER TABLE embedding_256 ADD COLUMN IF NOT EXISTS run_uid VARCHAR(96)",
    "ALTER TABLE embedding_128 ADD COLUMN IF NOT EXISTS run_uid VARCHAR(96)",
    "ALTER TABLE embedding_128 ADD COLUMN IF NOT EXISTS vector_type VARCHAR(32)",
    "UPDATE embedding_128 SET vector_type = 'pca_128' WHERE vector_type IS NULL",
    "ALTER TABLE embedding_128 ALTER COLUMN vector_type SET NOT NULL",
    "ALTER TABLE embedding_pq ADD COLUMN IF NOT EXISTS run_uid VARCHAR(96)",
    (
        "UPDATE embedding_512 SET vector_type = 'origin_512' "
        "WHERE vector_type = 'arcface' "
        "AND NOT EXISTS ("
        "SELECT 1 FROM embedding_512 canonical "
        "WHERE canonical.run_uid IS NOT DISTINCT FROM embedding_512.run_uid "
        "AND canonical.image_id = embedding_512.image_id "
        "AND canonical.vector_type = 'origin_512'"
        ")"
    ),
    (
        "UPDATE embedding_512 SET vector_type = 'origin_512' "
        "WHERE vector_type = 'origin' "
        "AND NOT EXISTS ("
        "SELECT 1 FROM embedding_512 canonical "
        "WHERE canonical.run_uid IS NOT DISTINCT FROM embedding_512.run_uid "
        "AND canonical.image_id = embedding_512.image_id "
        "AND canonical.vector_type = 'origin_512'"
        ")"
    ),
    (
        "UPDATE embedding_256 SET vector_type = 'pca_256' "
        "WHERE vector_type = 'pca' "
        "AND NOT EXISTS ("
        "SELECT 1 FROM embedding_256 canonical "
        "WHERE canonical.run_uid IS NOT DISTINCT FROM embedding_256.run_uid "
        "AND canonical.image_id = embedding_256.image_id "
        "AND canonical.vector_type = 'pca_256'"
        ")"
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_embedding_512_run_image_type "
        "ON embedding_512 (run_uid, image_id, vector_type) WHERE run_uid IS NOT NULL"
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_embedding_256_run_image_type "
        "ON embedding_256 (run_uid, image_id, vector_type) WHERE run_uid IS NOT NULL"
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_embedding_448_run_image_type_unique "
        "ON embedding_448 (run_uid, image_id, vector_type) WHERE run_uid IS NOT NULL"
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_embedding_384_run_image_type_unique "
        "ON embedding_384 (run_uid, image_id, vector_type) WHERE run_uid IS NOT NULL"
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_embedding_128_run_image_type_unique "
        "ON embedding_128 (run_uid, image_id, vector_type) WHERE run_uid IS NOT NULL"
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_embedding_pq_run_image_type "
        "ON embedding_pq (run_uid, image_id, vector_type) WHERE run_uid IS NOT NULL"
    ),
    "CREATE INDEX IF NOT EXISTS ix_embedding_512_run_type ON embedding_512 (run_uid, vector_type)",
    "CREATE INDEX IF NOT EXISTS ix_embedding_256_run_type ON embedding_256 (run_uid, vector_type)",
    "CREATE INDEX IF NOT EXISTS ix_embedding_448_run_type ON embedding_448 (run_uid, vector_type)",
    "CREATE INDEX IF NOT EXISTS ix_embedding_384_run_type ON embedding_384 (run_uid, vector_type)",
    "CREATE INDEX IF NOT EXISTS ix_embedding_128_run_type ON embedding_128 (run_uid, vector_type)",
    "CREATE INDEX IF NOT EXISTS ix_embedding_pq_run_type ON embedding_pq (run_uid, vector_type)",
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS run_uid VARCHAR(96)",
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS started_at TIMESTAMP",
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS finished_at TIMESTAMP",
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS metadata JSON",
    (
        "UPDATE research_runs SET artifact_dir = 'legacy://research-run/' || id::text "
        "WHERE artifact_dir IS NULL"
    ),
    (
        "UPDATE research_runs SET run_uid = 'legacy-' || id::text "
        "WHERE run_uid IS NULL"
    ),
    "ALTER TABLE research_runs ALTER COLUMN run_uid SET NOT NULL",
    "ALTER TABLE research_runs ALTER COLUMN artifact_dir SET NOT NULL",
    "ALTER TABLE embedding_pq ALTER COLUMN codes SET NOT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_research_runs_run_uid ON research_runs (run_uid)",
    "ALTER TABLE research_runs DROP CONSTRAINT IF EXISTS research_runs_run_name_key",
    "ALTER TABLE research_search_results ADD COLUMN IF NOT EXISTS is_mated BOOLEAN",
    "ALTER TABLE research_search_results ADD COLUMN IF NOT EXISTS top1_correct BOOLEAN",
    "ALTER TABLE research_search_results ADD COLUMN IF NOT EXISTS accepted BOOLEAN",
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_research_split "
        "ON research_splits (run_id, split_name, identity_id, role)"
    ),
    (
        "DO $$ BEGIN "
        "IF EXISTS ("
        "SELECT 1 FROM pg_constraint "
        "WHERE conrelid = 'research_templates'::regclass "
        "AND conname = 'uq_research_template' "
        "AND pg_get_constraintdef(oid) NOT LIKE '%enrollment_count%'"
        ") THEN "
        "ALTER TABLE research_templates DROP CONSTRAINT uq_research_template; "
        "END IF; END $$"
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_research_template "
        "ON research_templates "
        "(run_id, identity_id, compression_profile, aggregation_method, enrollment_count)"
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_research_search_result "
        "ON research_search_results (run_id, query_id, compression_profile)"
    ),
    (
        "DO $$ BEGIN "
        "IF EXISTS ("
        "SELECT 1 FROM pg_constraint constraint_row "
        "JOIN pg_index index_row ON index_row.indexrelid = constraint_row.conindid "
        "WHERE constraint_row.conrelid = 'research_calibration_results'::regclass "
        "AND constraint_row.conname = 'uq_research_calibration_result' "
        "AND NOT index_row.indnullsnotdistinct"
        ") THEN "
        "ALTER TABLE research_calibration_results "
        "DROP CONSTRAINT uq_research_calibration_result; "
        "END IF; END $$"
    ),
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_research_calibration_result "
        "ON research_calibration_results "
        "(run_id, model_name, compression_profile, target_fpir) NULLS NOT DISTINCT"
    ),
)


def _reject_null_pq_codes(connection: Connection) -> None:
    null_count = int(
        connection.execute(
            text("SELECT count(1) FROM embedding_pq WHERE codes IS NULL")
        ).scalar_one()
    )
    if null_count:
        raise RuntimeError(
            "cannot migrate embedding_pq.codes to NOT NULL: "
            f"{null_count} legacy rows have NULL codes; repair or remove those rows first"
        )


def migrate_legacy_research_schema(bind: Engine | Connection) -> None:
    """Apply backward-compatible thesis2-to-thesis3 schema changes.

    Legacy tables and rows are retained. Besides additive columns and backfills, the
    old template/calibration uniqueness constraints are replaced with the thesis3
    definitions needed for enrollment-count experiments and NULL-safe calibration.
    """

    def _migrate(connection: Connection) -> None:
        _reject_null_pq_codes(connection)
        for statement in _ADDITIVE_RESEARCH_SCHEMA_SQL:
            connection.execute(text(statement))

    if isinstance(bind, Engine):
        with bind.begin() as connection:
            _migrate(connection)
    else:
        _migrate(bind)
