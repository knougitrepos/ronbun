"""Optional pgvector persistence for a pinned, completed 1024D input artifact."""
import json
from pathlib import Path

from research.database.models import Embedding1024, TemplateEmbedding1024
from research.database.repository import VectorRepository
from research.experiments.facex_inputs import read_inputs
from research.runtime.hashing import sha256_file


def ensure_origin_1024_schema(connection):
    """Add the separate origin tables; preserve all existing 512D tables/data.

    Requires the project's images table and pgvector extension to exist.
    The existing init_database path also creates these tables and HNSW indexes.
    """
    Embedding1024.__table__.create(connection, checkfirst=True)
    TemplateEmbedding1024.__table__.create(connection, checkfirst=True)


def persist_origin_input(directory, engine=None):
    from research.database.connection import create_database_engine, session_scope
    path = Path(directory)
    manifest, population, vectors, _, _ = read_inputs(path)
    manifest_hash = sha256_file(path / "manifest.json")
    run_uid = "input-" + manifest_hash[:24]
    hashes = {row["image_id"]: row["sha256"] for row in (
        json.loads(line) for line in (path / "image_hashes.jsonl").read_text(encoding="utf-8").splitlines())}
    owned = engine is None
    engine = engine if engine is not None else create_database_engine()
    try:
        with engine.begin() as connection:
            ensure_origin_1024_schema(connection)
        with session_scope(engine) as session:
            repository = VectorRepository(session)
            for i, row in population.iterrows():
                image = repository.add_image(row.image_path, row.identity_id or None,
                                             content_sha256=hashes[row.image_id])
                repository.add_embedding_1024(image.id, "origin_1024",
                    dict(model_uid=manifest["model_uid"], source_image_id=row.image_id,
                         input_manifest_sha256=manifest_hash),
                    vectors[i], run_uid=run_uid)
        return dict(run_uid=run_uid, images=len(population), model_uid=manifest["model_uid"], table="embedding_1024")
    finally:
        if owned:
            engine.dispose()
