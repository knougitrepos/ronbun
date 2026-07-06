from core.schemas import (
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
    assert "config_hash" in ResearchRun.__table__.columns
    assert "compression_profile" in ResearchTemplate.__table__.columns
    assert "top1_score" in ResearchSearchResult.__table__.columns
    assert "target_fpir" in ResearchCalibrationResult.__table__.columns
