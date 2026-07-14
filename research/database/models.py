from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    BigInteger,
    String,
    Text,
    TIMESTAMP,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector


Base = declarative_base()


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    image_path = Column(Text, unique=True, nullable=False)
    label = Column(Text)
    content_sha256 = Column(String(64))
    file_size_bytes = Column(BigInteger)
    created_at = Column(TIMESTAMP)


class Embedding512(Base):
    __tablename__ = "embedding_512"
    __table_args__ = (
        UniqueConstraint("run_uid", "image_id", "vector_type", name="uq_embedding_512_run_image_type"),
    )

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    run_uid = Column(String(96))
    vector_type = Column(String(32), nullable=False)
    parameters = Column(JSON)
    embedding = Column(Vector(512), nullable=False)
    created_at = Column(TIMESTAMP)
    log = Column(Text)


class Embedding256(Base):
    __tablename__ = "embedding_256"
    __table_args__ = (
        UniqueConstraint("run_uid", "image_id", "vector_type", name="uq_embedding_256_run_image_type"),
    )

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    run_uid = Column(String(96))
    vector_type = Column(String(32), nullable=False)
    parameters = Column(JSON)
    embedding = Column(Vector(256), nullable=False)
    created_at = Column(TIMESTAMP)
    log = Column(Text)


class EmbeddingPQ(Base):
    __tablename__ = "embedding_pq"
    __table_args__ = (
        UniqueConstraint("run_uid", "image_id", "vector_type", name="uq_embedding_pq_run_image_type"),
    )

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    run_uid = Column(String(96))
    vector_type = Column(String(32), nullable=False)
    parameters = Column(JSON)
    codes = Column(LargeBinary, nullable=False)
    created_at = Column(TIMESTAMP)
    log = Column(Text)


class ResearchRun(Base):
    __tablename__ = "research_runs"

    id = Column(Integer, primary_key=True)
    run_uid = Column(String(96), unique=True, nullable=False)
    run_name = Column(Text, nullable=False)
    config_hash = Column(String(64), nullable=False)
    config = Column(JSON, nullable=False)
    status = Column(String(32), nullable=False, default="created")
    artifact_dir = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP)
    started_at = Column(TIMESTAMP)
    finished_at = Column(TIMESTAMP)
    details = Column("metadata", JSON)


class ResearchSplit(Base):
    __tablename__ = "research_splits"
    __table_args__ = (
        UniqueConstraint("run_id", "split_name", "identity_id", "role", name="uq_research_split"),
    )

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("research_runs.id"), nullable=False)
    split_name = Column(String(32), nullable=False)
    identity_id = Column(Text, nullable=False)
    role = Column(String(64), nullable=False)
    details = Column("metadata", JSON)
    created_at = Column(TIMESTAMP)


class ResearchTemplate(Base):
    __tablename__ = "research_templates"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "identity_id",
            "compression_profile",
            "aggregation_method",
            name="uq_research_template",
        ),
    )

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("research_runs.id"), nullable=False)
    identity_id = Column(Text, nullable=False)
    compression_profile = Column(String(64), nullable=False)
    aggregation_method = Column(String(64), nullable=False)
    enrollment_count = Column(Integer, nullable=False)
    quality = Column(Float)
    variance = Column(Float)
    source_image_ids = Column(JSON)
    embedding_artifact = Column(Text)
    details = Column("metadata", JSON)
    created_at = Column(TIMESTAMP)


class ResearchSearchResult(Base):
    __tablename__ = "research_search_results"
    __table_args__ = (
        UniqueConstraint("run_id", "query_id", "compression_profile", name="uq_research_search_result"),
    )

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("research_runs.id"), nullable=False)
    query_id = Column(Text, nullable=False)
    query_identity_id = Column(Text)
    probe_type = Column(String(64), nullable=False)
    compression_profile = Column(String(64), nullable=False)
    top1_identity = Column(Text)
    top1_score = Column(Float)
    top2_score = Column(Float)
    score_margin = Column(Float)
    is_mated = Column(Boolean)
    top1_correct = Column(Boolean)
    accepted = Column(Boolean)
    y_true_accept = Column(Boolean, nullable=False)
    ranked_identities = Column(JSON)
    details = Column("metadata", JSON)
    created_at = Column(TIMESTAMP)


class ResearchCalibrationResult(Base):
    __tablename__ = "research_calibration_results"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "model_name",
            "compression_profile",
            "target_fpir",
            name="uq_research_calibration_result",
        ),
    )

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("research_runs.id"), nullable=False)
    model_name = Column(String(128), nullable=False)
    compression_profile = Column(String(64))
    target_fpir = Column(Float, nullable=False)
    threshold = Column(Float)
    metrics = Column(JSON)
    feature_columns = Column(JSON)
    details = Column("metadata", JSON)
    created_at = Column(TIMESTAMP)
