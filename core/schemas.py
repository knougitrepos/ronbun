from sqlalchemy import Column, Integer, String, Text, JSON, TIMESTAMP, ForeignKey
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

class Embedding512(Base):
    __tablename__ = 'embedding_512'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    vector_type = Column(String(32), nullable=False)  # 'origin'이면 ArcFace 원본 벡터
    parameters = Column(JSON)  # 원본은 {}, 변환은 파라미터 dict
    embedding = Column(Vector(512), nullable=False)
    created_at = Column(TIMESTAMP)

class Embedding128(Base):
    __tablename__ = 'embedding_128'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    vector_type = Column(String(32), nullable=False)
    parameters = Column(JSON)
    embedding = Column(Vector(128), nullable=False)
    created_at = Column(TIMESTAMP)

# 필요시 Embedding128, Embedding1024 등 추가 가능 