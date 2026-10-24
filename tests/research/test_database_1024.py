"""1024D pgvector round trip in an isolated schema, rolled back afterwards."""
import uuid

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from research.database.models import Embedding1024, Image, TemplateEmbedding1024
from research.database.repository import VectorRepository


def test_1024_payload_guard_precedes_database():
    repo = VectorRepository(None)
    with pytest.raises(ValueError, match="1024D"):
        repo.add_embedding_1024(1, "origin_1024", {}, np.ones(512))
    with pytest.raises(ValueError, match="finite"):
        repo.add_embedding_1024(1, "origin_1024", {}, np.full(1024, np.nan))


def test_real_postgres_1024_round_trip():
    from research.database.connection import create_database_engine
    try:
        engine = create_database_engine()
        connection = engine.connect()
    except Exception as exc:
        pytest.skip(f"PostgreSQL unavailable ({type(exc).__name__})")
    transaction = connection.begin()
    try:
        if not connection.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar():
            pytest.skip("pgvector extension unavailable")
        schema = "facex_smoke_" + uuid.uuid4().hex
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        connection.execute(text(f'SET LOCAL search_path TO "{schema}", public'))
        for table in (Image.__table__, Embedding1024.__table__, TemplateEmbedding1024.__table__):
            table.create(connection)
        with Session(bind=connection, join_transaction_mode="create_savepoint") as session:
            repo = VectorRepository(session)
            image = repo.add_image("synthetic-fixture://1024", "synthetic")
            vector = np.zeros(1024, dtype=np.float32)
            vector[0] = 1
            stored = repo.add_embedding_1024(image.id, "origin_1024", {"model_uid": "synthetic"}, vector, run_uid="synthetic")
            np.testing.assert_allclose(stored.embedding, vector)
            found = repo.find_similar_1024(vector, top_k=1, run_uid="synthetic")
            assert found[0]["image_id"] == image.id
            scope = dict(run_uid="synthetic", protocol_name="test", vector_type="origin_1024",
                         aggregation_method="mean_unit", enrollment_policy="fixed", enrollment_target=1, model_uid="synthetic")
            repo.upsert_template_1024(**scope, enrollment_count=1, identity_id="person", source_image_ids=["image"], embedding=vector)
            hits = repo.find_similar_templates_1024(vector, **scope, top_k=1, search_mode="exact")
            assert hits[0]["identity_id"] == "person"
    finally:
        transaction.rollback()
        connection.close()
        engine.dispose()
