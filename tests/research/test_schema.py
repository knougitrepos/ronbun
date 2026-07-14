from research.database.models import (
    Embedding256,
    Embedding512,
    EmbeddingPQ,
    Image,
    ResearchCalibrationResult,
    ResearchRun,
    ResearchSearchResult,
    ResearchSplit,
    ResearchTemplate,
)


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
