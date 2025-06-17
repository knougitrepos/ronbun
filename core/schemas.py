from sqlalchemy import Column, Integer, String, Text, JSON, TIMESTAMP, ForeignKey, LargeBinary, Float, Boolean
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    image_path = Column(Text, unique=True, nullable=False)
    label = Column(Text)
    created_at = Column(TIMESTAMP)
    # 각 임베딩 테이블과의 관계는 필요시 추가

class Embedding256(Base):
    __tablename__ = 'embedding_256'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    vector_type = Column(String(32), nullable=False)
    parameters = Column(JSON)
    embedding = Column(Vector(256), nullable=False)
    created_at = Column(TIMESTAMP)
    log = Column(Text)  # 변환 로그/분산/오류 등 기록

class Embedding512(Base):
    __tablename__ = 'embedding_512'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    vector_type = Column(String(32), nullable=False)  # 'origin'이면 ArcFace 원본 벡터
    parameters = Column(JSON)  # 원본은 {}, 변환은 파라미터 dict
    embedding = Column(Vector(512), nullable=False)
    created_at = Column(TIMESTAMP)
    log = Column(Text)

class Embedding128(Base):
    __tablename__ = 'embedding_128'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    vector_type = Column(String(32), nullable=False)
    parameters = Column(JSON)
    embedding = Column(Vector(128), nullable=False)
    created_at = Column(TIMESTAMP)
    log = Column(Text)

class EmbeddingPQ(Base):
    __tablename__ = 'embedding_pq'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    vector_type = Column(String(32), nullable=False)
    parameters = Column(JSON)
    codes = Column(LargeBinary)  # 또는 JSON, depending on storage preference
    created_at = Column(TIMESTAMP)
    log = Column(Text)

class RecallTestResults(Base):
    __tablename__ = 'recall_test_results'
    id = Column(Integer, primary_key=True)
    test_name = Column(Text, nullable=False)
    vector_type_compared = Column(Text, nullable=False)
    distance_metric = Column(Text, nullable=False)
    k_value = Column(Integer, nullable=False)
    recall_value = Column(Float, nullable=False)
    method = Column(Text)
    parameters = Column(JSON)
    created_at = Column(TIMESTAMP)
    cross_validation_fold = Column(Integer)

class ImageEmbeddings(Base):
    __tablename__ = 'image_embeddings'
    id = Column(Integer, primary_key=True)
    image_path = Column(Text, unique=True, nullable=False)
    label = Column(Text)
    used_feature_extract_model = Column(Text)
    used_distance_model = Column(Text)
    is_extract_face = Column(Boolean, default=False)
    vec_origin = Column(Vector(512), nullable=False)
    vec_wavelet = Column(Vector(256))
    params_wavelet = Column(JSON)
    vec_dct = Column(Vector(256))
    params_dct = Column(JSON)
    vec_pca = Column(Vector(256))
    params_pca = Column(JSON)
    vec_quantized = Column(Vector(256))
    params_quantized = Column(JSON)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

class ImageEmbeddingsSummary(Base):
    __tablename__ = 'image_embeddings_summary'
    id = Column(Integer, primary_key=True)
    label = Column(Text, nullable=False)
    count = Column(Integer, nullable=False)
    updated_at = Column(TIMESTAMP)

class RecallTestResultsSummary(Base):
    __tablename__ = 'recall_test_results_summary'
    id = Column(Integer, primary_key=True)
    label = Column(Text, nullable=False)
    count = Column(Integer, nullable=False)
    updated_at = Column(TIMESTAMP)
    explain = Column(Text)

# 필요시 Embedding128, Embedding1024 등 추가 가능 