from research.database.models import (
    Embedding128,
    Embedding256,
    Embedding384,
    Embedding448,
    Embedding512,
    EmbeddingPQ,
    Image,
    ResearchCalibrationResult,
    ResearchRun,
    ResearchSearchResult,
    ResearchSplit,
    ResearchTemplate,
    TemplateEmbedding128,
    TemplateEmbedding256,
    TemplateEmbedding384,
    TemplateEmbedding448,
    TemplateEmbedding512,
)
from research.database.migrations import _reject_null_pq_codes


def test_research_tables_are_registered_without_replacing_existing_tables():
    assert ResearchRun.__tablename__ == "research_runs"
    assert ResearchSplit.__tablename__ == "research_splits"
    assert ResearchTemplate.__tablename__ == "research_templates"
    assert ResearchSearchResult.__tablename__ == "research_search_results"
    assert ResearchCalibrationResult.__tablename__ == "research_calibration_results"
    assert "run_uid" in ResearchRun.__table__.columns
    assert "content_sha256" in Image.__table__.columns
    assert "run_uid" in Embedding512.__table__.columns
    assert "run_uid" in Embedding256.__table__.columns
    assert "run_uid" in EmbeddingPQ.__table__.columns
    assert "config_hash" in ResearchRun.__table__.columns
    assert "compression_profile" in ResearchTemplate.__table__.columns
    assert "top1_score" in ResearchSearchResult.__table__.columns
    assert "is_mated" in ResearchSearchResult.__table__.columns
    assert "top1_correct" in ResearchSearchResult.__table__.columns
    assert "target_fpir" in ResearchCalibrationResult.__table__.columns
    assert TemplateEmbedding512.__tablename__ == "template_embedding_512"
    assert TemplateEmbedding128.__tablename__ == "template_embedding_128"
    assert TemplateEmbedding256.__tablename__ == "template_embedding_256"
    assert TemplateEmbedding384.__tablename__ == "template_embedding_384"
    assert TemplateEmbedding448.__tablename__ == "template_embedding_448"
    assert TemplateEmbedding512.__table__.columns["embedding"].type.dim == 512
    assert TemplateEmbedding128.__table__.columns["embedding"].type.dim == 128
    assert TemplateEmbedding256.__table__.columns["embedding"].type.dim == 256
    assert TemplateEmbedding384.__table__.columns["embedding"].type.dim == 384
    assert TemplateEmbedding448.__table__.columns["embedding"].type.dim == 448
    assert Embedding128.__table__.columns["embedding"].type.dim == 128
    assert Embedding384.__table__.columns["embedding"].type.dim == 384
    assert Embedding448.__table__.columns["embedding"].type.dim == 448
    assert EmbeddingPQ.__table__.columns["codes"].nullable is False


def test_null_legacy_pq_codes_are_rejected_before_not_null_migration():
    class Result:
        @staticmethod
        def scalar_one():
            return 2

    class Connection:
        @staticmethod
        def execute(_statement):
            return Result()

    try:
        _reject_null_pq_codes(Connection())
    except RuntimeError as exc:
        assert "2 legacy rows have NULL codes" in str(exc)
    else:
        raise AssertionError("NULL PQ codes must block the migration")
