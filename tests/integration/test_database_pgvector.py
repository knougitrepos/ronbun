import numpy as np
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from core.config import ConfigLoader
from core.database import (
    check_database_health,
    ensure_vector_extension,
    ensure_vector_indexes,
    init_database,
    VectorRepository,
)
from core.schemas import Base


def _engine():
    cfg = ConfigLoader().db
    url = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
        cfg["user"],
        cfg["password"],
        cfg["host"],
        cfg["port"],
        cfg["dbname"],
    )
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            conn.execute(text("select 1"))
    except Exception as exc:  # pragma: no cover - depends on local services
        pytest.skip(f"PostgreSQL test database unavailable: {exc}")
    return engine


@pytest.mark.db
def test_database_health_reports_pgvector_and_missing_tables_without_creating():
    health = check_database_health(bind=_engine())

    assert health["database"] == "postgres"
    assert health["user"] == "postgres"
    assert health["server_version"].startswith("PostgreSQL 18.")
    assert health["vector_extension_version"] == "0.8.4"
    assert isinstance(health["missing_tables"], list)
    assert isinstance(health["existing_tables"], list)


@pytest.mark.db
def test_init_database_and_pgvector_search_work_inside_rollback_transaction():
    engine = _engine()
    with engine.connect() as conn:
        outer = conn.begin()
        try:
            ensure_vector_extension(bind=conn)
            init_database(bind=conn)
            session = sessionmaker(bind=conn, autoflush=False)()
            repo = VectorRepository(session)

            image_a = repo.add_image("tmp/a.jpg", label="a")
            image_b = repo.add_image("tmp/b.jpg", label="b")
            repo.add_embedding_512(
                image_a.id,
                vector_type="origin",
                parameters={},
                embedding=np.r_[1.0, np.zeros(511)].tolist(),
            )
            repo.add_embedding_512(
                image_b.id,
                vector_type="origin",
                parameters={},
                embedding=np.r_[0.0, 1.0, np.zeros(510)].tolist(),
            )
            repo.add_embedding_256(
                image_a.id,
                vector_type="pca",
                parameters={"n_components": 256},
                embedding=np.r_[1.0, np.zeros(255)].tolist(),
            )
            repo.add_embedding_256(
                image_b.id,
                vector_type="pca",
                parameters={"n_components": 256},
                embedding=np.r_[0.0, 1.0, np.zeros(254)].tolist(),
            )

            results = repo.find_similar_512(np.r_[1.0, np.zeros(511)], top_k=2, vector_type="origin")
            pca_results = repo.find_similar_256(
                np.r_[1.0, np.zeros(255)],
                top_k=1,
                vector_type="pca",
                param_filter={"n_components": 256},
            )

            assert [row["label"] for row in results] == ["a", "b"]
            assert results[0]["distance"] < results[1]["distance"]
            assert results[0]["similarity"] > 0.99
            assert [row["label"] for row in pca_results] == ["a"]
        finally:
            outer.rollback()


@pytest.mark.db
def test_vector_indexes_are_created_for_pgvector_search_profiles():
    engine = _engine()
    with engine.connect() as conn:
        outer = conn.begin()
        try:
            init_database(bind=conn)
            ensure_vector_indexes(bind=conn)

            index_names = {
                row[0]
                for row in conn.execute(
                    text(
                        "SELECT indexname FROM pg_indexes "
                        "WHERE schemaname = 'public' "
                        "AND tablename IN ('embedding_512', 'embedding_256')"
                    )
                )
            }

            assert "ix_embedding_512_embedding_hnsw_cosine" in index_names
            assert "ix_embedding_256_embedding_hnsw_cosine" in index_names
        finally:
            outer.rollback()
