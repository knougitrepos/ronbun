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


class Embedding448(Base):
    __tablename__ = "embedding_448"
    __table_args__ = (
        UniqueConstraint("run_uid", "image_id", "vector_type", name="uq_embedding_448_run_image_type"),
    )

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    run_uid = Column(String(96))
    vector_type = Column(String(32), nullable=False)
    parameters = Column(JSON)
    embedding = Column(Vector(448), nullable=False)
    created_at = Column(TIMESTAMP)
    log = Column(Text)


class Embedding384(Base):
    __tablename__ = "embedding_384"
    __table_args__ = (
        UniqueConstraint("run_uid", "image_id", "vector_type", name="uq_embedding_384_run_image_type"),
    )

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    run_uid = Column(String(96))
    vector_type = Column(String(32), nullable=False)
    parameters = Column(JSON)
    embedding = Column(Vector(384), nullable=False)
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


class Embedding128(Base):
    __tablename__ = "embedding_128"
    __table_args__ = (
        UniqueConstraint("run_uid", "image_id", "vector_type", name="uq_embedding_128_run_image_type"),
    )

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    run_uid = Column(String(96))
    vector_type = Column(String(32), nullable=False)
    parameters = Column(JSON)
    embedding = Column(Vector(128), nullable=False)
    created_at = Column(TIMESTAMP)
    log = Column(Text)


class Embedding64(Base):
    __tablename__ = "embedding_64"
    __table_args__ = (
        UniqueConstraint("run_uid", "image_id", "vector_type", name="uq_embedding_64_run_image_type"),
    )

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    run_uid = Column(String(96))
    vector_type = Column(String(32), nullable=False)
    parameters = Column(JSON)
    embedding = Column(Vector(64), nullable=False)
    created_at = Column(TIMESTAMP)
    log = Column(Text)


class Embedding32(Base):
    __tablename__ = "embedding_32"
    __table_args__ = (
        UniqueConstraint("run_uid", "image_id", "vector_type", name="uq_embedding_32_run_image_type"),
    )

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    run_uid = Column(String(96))
    vector_type = Column(String(32), nullable=False)
    parameters = Column(JSON)
    embedding = Column(Vector(32), nullable=False)
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


class TemplateEmbedding512(Base):
    __tablename__ = "template_embedding_512"
    __table_args__ = (
        UniqueConstraint(
            "run_uid",
            "protocol_name",
            "vector_type",
            "aggregation_method",
            "enrollment_policy",
            "enrollment_target",
            "identity_id",
            "model_uid",
            name="uq_template_embedding_512_scope",
        ),
    )

    id = Column(Integer, primary_key=True)
    run_uid = Column(String(96), nullable=False)
    protocol_name = Column(String(64), nullable=False)
    vector_type = Column(String(32), nullable=False)
    aggregation_method = Column(String(64), nullable=False)
    enrollment_policy = Column(String(32), nullable=False)
    enrollment_target = Column(Integer, nullable=False)
    enrollment_count = Column(Integer, nullable=False)
    identity_id = Column(Text, nullable=False)
    model_uid = Column(String(128), nullable=False)
    source_image_ids = Column(JSON, nullable=False)
    quality = Column(Float)
    variance = Column(Float)
    angular_error = Column(Float)
    reconstruction_error_norm = Column(Float)
    parameters = Column(JSON)
    embedding = Column(Vector(512), nullable=False)
    created_at = Column(TIMESTAMP)


class TemplateEmbedding448(Base):
    __tablename__ = "template_embedding_448"
    __table_args__ = (
        UniqueConstraint(
            "run_uid", "protocol_name", "vector_type", "aggregation_method",
            "enrollment_policy", "enrollment_target", "identity_id", "model_uid",
            name="uq_template_embedding_448_scope",
        ),
    )

    id = Column(Integer, primary_key=True)
    run_uid = Column(String(96), nullable=False)
    protocol_name = Column(String(64), nullable=False)
    vector_type = Column(String(32), nullable=False)
    aggregation_method = Column(String(64), nullable=False)
    enrollment_policy = Column(String(32), nullable=False)
    enrollment_target = Column(Integer, nullable=False)
    enrollment_count = Column(Integer, nullable=False)
    identity_id = Column(Text, nullable=False)
    model_uid = Column(String(128), nullable=False)
    source_image_ids = Column(JSON, nullable=False)
    quality = Column(Float)
    variance = Column(Float)
    angular_error = Column(Float)
    reconstruction_error_norm = Column(Float)
    parameters = Column(JSON)
    embedding = Column(Vector(448), nullable=False)
    created_at = Column(TIMESTAMP)


class TemplateEmbedding384(Base):
    __tablename__ = "template_embedding_384"
    __table_args__ = (
        UniqueConstraint(
            "run_uid", "protocol_name", "vector_type", "aggregation_method",
            "enrollment_policy", "enrollment_target", "identity_id", "model_uid",
            name="uq_template_embedding_384_scope",
        ),
    )

    id = Column(Integer, primary_key=True)
    run_uid = Column(String(96), nullable=False)
    protocol_name = Column(String(64), nullable=False)
    vector_type = Column(String(32), nullable=False)
    aggregation_method = Column(String(64), nullable=False)
    enrollment_policy = Column(String(32), nullable=False)
    enrollment_target = Column(Integer, nullable=False)
    enrollment_count = Column(Integer, nullable=False)
    identity_id = Column(Text, nullable=False)
    model_uid = Column(String(128), nullable=False)
    source_image_ids = Column(JSON, nullable=False)
    quality = Column(Float)
    variance = Column(Float)
    angular_error = Column(Float)
    reconstruction_error_norm = Column(Float)
    parameters = Column(JSON)
    embedding = Column(Vector(384), nullable=False)
    created_at = Column(TIMESTAMP)


class TemplateEmbedding256(Base):
    __tablename__ = "template_embedding_256"
    __table_args__ = (
        UniqueConstraint(
            "run_uid",
            "protocol_name",
            "vector_type",
            "aggregation_method",
            "enrollment_policy",
            "enrollment_target",
            "identity_id",
            "model_uid",
            name="uq_template_embedding_256_scope",
        ),
    )

    id = Column(Integer, primary_key=True)
    run_uid = Column(String(96), nullable=False)
    protocol_name = Column(String(64), nullable=False)
    vector_type = Column(String(32), nullable=False)
    aggregation_method = Column(String(64), nullable=False)
    enrollment_policy = Column(String(32), nullable=False)
    enrollment_target = Column(Integer, nullable=False)
    enrollment_count = Column(Integer, nullable=False)
    identity_id = Column(Text, nullable=False)
    model_uid = Column(String(128), nullable=False)
    source_image_ids = Column(JSON, nullable=False)
    quality = Column(Float)
    variance = Column(Float)
    angular_error = Column(Float)
    reconstruction_error_norm = Column(Float)
    parameters = Column(JSON)
    embedding = Column(Vector(256), nullable=False)
    created_at = Column(TIMESTAMP)


class TemplateEmbedding128(Base):
    __tablename__ = "template_embedding_128"
    __table_args__ = (
        UniqueConstraint(
            "run_uid", "protocol_name", "vector_type", "aggregation_method",
            "enrollment_policy", "enrollment_target", "identity_id", "model_uid",
            name="uq_template_embedding_128_scope",
        ),
    )

    id = Column(Integer, primary_key=True)
    run_uid = Column(String(96), nullable=False)
    protocol_name = Column(String(64), nullable=False)
    vector_type = Column(String(32), nullable=False)
    aggregation_method = Column(String(64), nullable=False)
    enrollment_policy = Column(String(32), nullable=False)
    enrollment_target = Column(Integer, nullable=False)
    enrollment_count = Column(Integer, nullable=False)
    identity_id = Column(Text, nullable=False)
    model_uid = Column(String(128), nullable=False)
    source_image_ids = Column(JSON, nullable=False)
    quality = Column(Float)
    variance = Column(Float)
    angular_error = Column(Float)
    reconstruction_error_norm = Column(Float)
    parameters = Column(JSON)
    embedding = Column(Vector(128), nullable=False)
    created_at = Column(TIMESTAMP)


class TemplateEmbedding64(Base):
    __tablename__ = "template_embedding_64"
    __table_args__ = (
        UniqueConstraint(
            "run_uid", "protocol_name", "vector_type", "aggregation_method",
            "enrollment_policy", "enrollment_target", "identity_id", "model_uid",
            name="uq_template_embedding_64_scope",
        ),
    )

    id = Column(Integer, primary_key=True)
    run_uid = Column(String(96), nullable=False)
    protocol_name = Column(String(64), nullable=False)
    vector_type = Column(String(32), nullable=False)
    aggregation_method = Column(String(64), nullable=False)
    enrollment_policy = Column(String(32), nullable=False)
    enrollment_target = Column(Integer, nullable=False)
    enrollment_count = Column(Integer, nullable=False)
    identity_id = Column(Text, nullable=False)
    model_uid = Column(String(128), nullable=False)
    source_image_ids = Column(JSON, nullable=False)
    quality = Column(Float)
    variance = Column(Float)
    angular_error = Column(Float)
    reconstruction_error_norm = Column(Float)
    parameters = Column(JSON)
    embedding = Column(Vector(64), nullable=False)
    created_at = Column(TIMESTAMP)


class TemplateEmbedding32(Base):
    __tablename__ = "template_embedding_32"
    __table_args__ = (
        UniqueConstraint(
            "run_uid", "protocol_name", "vector_type", "aggregation_method",
            "enrollment_policy", "enrollment_target", "identity_id", "model_uid",
            name="uq_template_embedding_32_scope",
        ),
    )

    id = Column(Integer, primary_key=True)
    run_uid = Column(String(96), nullable=False)
    protocol_name = Column(String(64), nullable=False)
    vector_type = Column(String(32), nullable=False)
    aggregation_method = Column(String(64), nullable=False)
    enrollment_policy = Column(String(32), nullable=False)
    enrollment_target = Column(Integer, nullable=False)
    enrollment_count = Column(Integer, nullable=False)
    identity_id = Column(Text, nullable=False)
    model_uid = Column(String(128), nullable=False)
    source_image_ids = Column(JSON, nullable=False)
    quality = Column(Float)
    variance = Column(Float)
    angular_error = Column(Float)
    reconstruction_error_norm = Column(Float)
    parameters = Column(JSON)
    embedding = Column(Vector(32), nullable=False)
    created_at = Column(TIMESTAMP)


PCA_EMBEDDING_MODELS = {
    32: Embedding32,
    64: Embedding64,
    128: Embedding128,
    256: Embedding256,
    384: Embedding384,
    448: Embedding448,
}

PCA_TEMPLATE_MODELS = {
    32: TemplateEmbedding32,
    64: TemplateEmbedding64,
    128: TemplateEmbedding128,
    256: TemplateEmbedding256,
    384: TemplateEmbedding384,
    448: TemplateEmbedding448,
}


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
            "enrollment_count",
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
            postgresql_nulls_not_distinct=True,
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
