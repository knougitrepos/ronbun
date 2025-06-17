import os
from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, TIMESTAMP, Float, text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pgvector.sqlalchemy import Vector
from core.schemas import *  # 모든 테이블 강제 import
from core.database import engine, Base, SessionLocal

# 데이터베이스 ORM 모델, 세션, 초기화/리셋 등 DB 유틸리티 관리

# DB 연결 설정
# DB_HOST = Config.DB_HOST
# DB_PORT = Config.DB_PORT
# DB_USER = Config.DB_USER
# DB_PASSWORD = Config.DB_PASS
# DB_NAME = Config.DB_NAME
# DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# engine = create_engine(DB_URL, echo=False)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# ORM 테이블 정의
# class RecallTestResults(Base):
#     __tablename__ = 'recall_test_results'
#     id = Column(Integer, primary_key=True)
#     test_name = Column(Text, nullable=False)
#     vector_type_compared = Column(Text, nullable=False)
#     distance_metric = Column(Text, nullable=False)
#     k_value = Column(Integer, nullable=False)
#     recall_value = Column(Float, nullable=False)
#     method = Column(Text)
#     parameters = Column(JSON)
#     created_at = Column(TIMESTAMP)
#     cross_validation_fold = Column(Integer)

# class ImageEmbeddings(Base):
#     __tablename__ = 'image_embeddings'
#     id = Column(Integer, primary_key=True)
#     image_path = Column(Text, unique=True, nullable=False)
#     label = Column(Text)
#     used_feature_extract_model = Column(Text)
#     used_distance_model = Column(Text)
#     is_extract_face = Column(Boolean, default=False)
#     vec_origin = Column(Vector(512), nullable=False)
#     vec_wavelet = Column(Vector(256))
#     params_wavelet = Column(JSON)
#     vec_dct = Column(Vector(256))
#     params_dct = Column(JSON)
#     vec_pca = Column(Vector(256))
#     params_pca = Column(JSON)
#     vec_quantized = Column(Vector(256))
#     params_quantized = Column(JSON)
#     created_at = Column(TIMESTAMP)
#     updated_at = Column(TIMESTAMP)

# class ImageEmbeddingsSummary(Base):
#     __tablename__ = 'image_embeddings_summary'
#     id = Column(Integer, primary_key=True)
#     label = Column(Text, nullable=False)
#     count = Column(Integer, nullable=False)
#     updated_at = Column(TIMESTAMP)

# class RecallTestResultsSummary(Base):
#     __tablename__ = 'recall_test_results_summary'
#     id = Column(Integer, primary_key=True)
#     label = Column(Text, nullable=False)
#     count = Column(Integer, nullable=False)
#     updated_at = Column(TIMESTAMP)
#     explain = Column(Text)

# 테이블 생성 함수
def init_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS recall_test_results CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS image_embeddings CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS image_embeddings_summary CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS recall_test_results_summary CASCADE;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        Base.metadata.create_all(bind=engine)
        print("테이블 및 EXTENSION 생성 완료")
    except Exception as e:
        print(f"init_db 에러: {e}")

# DB 전체 초기화 함수 (테이블 드롭 후 재생성)
def reset_db():
    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        print("core/schemas.py has been reset")
    except Exception as e:
        print(f"reset_db 에러: {e}")

# 세션 생성 유틸 (FastAPI Dependency 등에서 사용)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 예시: 데이터 추가 함수 (ORM 방식)
def add_recall_test_result(db: Session, **kwargs):
    obj = RecallTestResults(**kwargs)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

# 예시: 데이터 전체 조회 함수
def get_all_recall_test_results(db: Session):
    return db.query(RecallTestResults).all()

# (필요시 다른 CRUD 함수도 추가 가능)

def add_image_embedding(
    db: Session,
    image_path: str,
    label: str = None,
    used_feature_extract_model: str = None,
    used_distance_model: str = None,
    is_extract_face: bool = False,
    vec_origin=None,
    vec_wavelet=None, params_wavelet=None,
    vec_dct=None, params_dct=None,
    vec_pca=None, params_pca=None,
    vec_quantized=None, params_quantized=None,
    created_at=None, updated_at=None
):
    """
    여러 벡터와 메타 정보를 한 번에 저장하는 함수
    """
    embedding = ImageEmbeddings(
        image_path=image_path,
        label=label,
        used_feature_extract_model=used_feature_extract_model,
        used_distance_model=used_distance_model,
        is_extract_face=is_extract_face,
        vec_origin=vec_origin,
        vec_wavelet=vec_wavelet,
        params_wavelet=params_wavelet,
        vec_dct=vec_dct,
        params_dct=params_dct,
        vec_pca=vec_pca,
        params_pca=params_pca,
        vec_quantized=vec_quantized,
        params_quantized=params_quantized,
        created_at=created_at,
        updated_at=updated_at
    )
    db.add(embedding)
    db.commit()
    db.refresh(embedding)
    return embedding
