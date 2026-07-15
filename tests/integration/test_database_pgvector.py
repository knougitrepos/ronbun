import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from research.database import (
    VectorRepository,
    check_database_health,
    create_database_engine,
    ensure_vector_extension,
    ensure_vector_indexes,
    init_database,
    load_database_settings,
)
from research.compression import ORIGIN_512, PCA_128, PCA_256


def _engine():
    settings = load_database_settings()
    engine = create_database_engine(settings)
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
            health = check_database_health(bind=conn)
            assert health["missing_tables"] == []
            assert health["schema_issues"] == []
            session = sessionmaker(bind=conn, autoflush=False)()
            repo = VectorRepository(session)

            image_a = repo.add_image("tmp/a.jpg", label="a")
            image_b = repo.add_image("tmp/b.jpg", label="b")
            repo.add_embedding_512(
                image_a.id,
                vector_type=ORIGIN_512,
                parameters={},
                embedding=np.r_[1.0, np.zeros(511)].tolist(),
                run_uid="pgvector-integration",
            )
            repo.add_embedding_512(
                image_b.id,
                vector_type=ORIGIN_512,
                parameters={},
                embedding=np.r_[0.0, 1.0, np.zeros(510)].tolist(),
                run_uid="pgvector-integration",
            )
            repo.add_embedding_256(
                image_a.id,
                vector_type=PCA_256,
                parameters={"n_components": 256},
                embedding=np.r_[1.0, np.zeros(255)].tolist(),
                run_uid="pgvector-integration",
            )
            repo.upsert_pca_embedding(
                128,
                image_a.id,
                vector_type=PCA_128,
                parameters={"n_components": 128},
                embedding=np.r_[1.0, np.zeros(127)].tolist(),
                run_uid="pgvector-integration",
            )
            repo.add_embedding_256(
                image_b.id,
                vector_type=PCA_256,
                parameters={"n_components": 256},
                embedding=np.r_[0.0, 1.0, np.zeros(254)].tolist(),
                run_uid="pgvector-integration",
            )

            duplicate = repo.add_embedding_512(
                image_a.id,
                vector_type=ORIGIN_512,
                parameters={"retry": True},
                embedding=np.r_[1.0, np.zeros(511)].tolist(),
                run_uid="pgvector-integration",
            )
            results = repo.find_similar_512(
                np.r_[1.0, np.zeros(511)],
                top_k=2,
                vector_type=ORIGIN_512,
                run_uid="pgvector-integration",
            )
            pca_results = repo.find_similar_256(
                np.r_[1.0, np.zeros(255)],
                top_k=1,
                vector_type=PCA_256,
                param_filter={"n_components": 256},
                run_uid="pgvector-integration",
            )
            _, unchanged_action = repo.upsert_embedding_256(
                image_a.id,
                vector_type=PCA_256,
                parameters={"n_components": 256},
                embedding=np.r_[1.0, np.zeros(255)].tolist(),
                run_uid="pgvector-integration",
            )
            with pytest.raises(ValueError, match="different provenance"):
                repo.upsert_embedding_256(
                    image_a.id,
                    vector_type=PCA_256,
                    parameters={"n_components": 128},
                    embedding=np.r_[1.0, np.zeros(255)].tolist(),
                    run_uid="pgvector-integration",
                )
            for image, vector in (
                (image_a, np.r_[1.0, np.zeros(511)]),
                (image_b, np.r_[0.0, 1.0, np.zeros(510)]),
            ):
                repo.upsert_template_512(
                    run_uid="pgvector-integration",
                    protocol_name="lfw-test",
                    vector_type=ORIGIN_512,
                    aggregation_method="mean",
                    enrollment_policy="fixed",
                    enrollment_target=1,
                    enrollment_count=1,
                    identity_id=image.label,
                    model_uid="arcface-test-model",
                    source_image_ids=[image.id],
                    embedding=vector.tolist(),
                    parameters={"attempt": "fixed"},
                )
            for image, vector in (
                (image_a, np.r_[1.0, np.zeros(127)]),
                (image_b, np.r_[0.0, 1.0, np.zeros(126)]),
            ):
                repo.upsert_pca_template(
                    128,
                    run_uid="pgvector-integration",
                    protocol_name="lfw-test",
                    vector_type=PCA_128,
                    aggregation_method="mean",
                    enrollment_policy="fixed",
                    enrollment_target=1,
                    enrollment_count=1,
                    identity_id=image.label,
                    model_uid="pca-128-test-model",
                    source_image_ids=[image.id],
                    embedding=vector.tolist(),
                    parameters={"attempt": "fixed"},
                )
            _, unchanged_template_action = repo.upsert_template_512(
                run_uid="pgvector-integration",
                protocol_name="lfw-test",
                vector_type=ORIGIN_512,
                aggregation_method="mean",
                enrollment_policy="fixed",
                enrollment_target=1,
                enrollment_count=1,
                identity_id="a",
                model_uid="arcface-test-model",
                source_image_ids=[image_a.id],
                embedding=np.r_[1.0, np.zeros(511)].tolist(),
                parameters={"attempt": "fixed"},
            )
            with pytest.raises(ValueError, match="different provenance"):
                repo.upsert_template_512(
                    run_uid="pgvector-integration",
                    protocol_name="lfw-test",
                    vector_type=ORIGIN_512,
                    aggregation_method="mean",
                    enrollment_policy="fixed",
                    enrollment_target=1,
                    enrollment_count=1,
                    identity_id="a",
                    model_uid="arcface-test-model",
                    source_image_ids=[image_a.id],
                    embedding=np.r_[1.0, np.zeros(511)].tolist(),
                    parameters={"attempt": "changed"},
                )
            exact_templates = repo.find_similar_templates_512(
                np.r_[1.0, np.zeros(511)],
                run_uid="pgvector-integration",
                protocol_name="lfw-test",
                vector_type=ORIGIN_512,
                aggregation_method="mean",
                enrollment_policy="fixed",
                enrollment_target=1,
                model_uid="arcface-test-model",
                top_k=2,
                search_mode="exact",
            )
            hnsw_templates = repo.find_similar_templates_512(
                np.r_[1.0, np.zeros(511)],
                run_uid="pgvector-integration",
                protocol_name="lfw-test",
                vector_type=ORIGIN_512,
                aggregation_method="mean",
                enrollment_policy="fixed",
                enrollment_target=1,
                model_uid="arcface-test-model",
                top_k=2,
                search_mode="hnsw",
            )
            pca_128_templates = repo.find_similar_pca_templates(
                128,
                np.r_[1.0, np.zeros(127)],
                run_uid="pgvector-integration",
                protocol_name="lfw-test",
                vector_type=PCA_128,
                aggregation_method="mean",
                enrollment_policy="fixed",
                enrollment_target=1,
                model_uid="pca-128-test-model",
                top_k=2,
                search_mode="hnsw",
            )

            assert duplicate.image_id == image_a.id
            assert unchanged_action == "skipped"
            assert unchanged_template_action == "skipped"
            assert len(repo.get_embeddings_512(run_uid="pgvector-integration")) == 2
            assert [row["label"] for row in results] == ["a", "b"]
            assert results[0]["distance"] < results[1]["distance"]
            assert results[0]["similarity"] > 0.99
            assert [row["label"] for row in pca_results] == ["a"]
            assert [row["identity_id"] for row in exact_templates] == ["a", "b"]
            assert [row["identity_id"] for row in hnsw_templates] == ["a", "b"]
            assert [row["identity_id"] for row in pca_128_templates] == ["a", "b"]
            assert {row["search_mode"] for row in exact_templates} == {"exact"}
            assert {row["search_mode"] for row in hnsw_templates} == {"hnsw"}
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
                        "AND tablename IN ("
                        "'embedding_512', 'embedding_448', 'embedding_384', "
                        "'embedding_256', 'embedding_128', "
                        "'template_embedding_512', 'template_embedding_448', "
                        "'template_embedding_384', 'template_embedding_256', "
                        "'template_embedding_128'"
                        ")"
                    )
                )
            }

            assert "ix_embedding_512_embedding_hnsw_cosine" in index_names
            assert "ix_embedding_256_embedding_hnsw_cosine" in index_names
            assert "ix_embedding_128_embedding_hnsw_cosine" in index_names
            assert "ix_embedding_384_embedding_hnsw_cosine" in index_names
            assert "ix_embedding_448_embedding_hnsw_cosine" in index_names
            assert "ix_template_embedding_512_hnsw_cosine" in index_names
            assert "ix_template_embedding_256_hnsw_cosine" in index_names
            assert "ix_template_embedding_128_hnsw_cosine" in index_names
            assert "ix_template_embedding_384_hnsw_cosine" in index_names
            assert "ix_template_embedding_448_hnsw_cosine" in index_names
        finally:
            outer.rollback()
