from sqlalchemy import create_engine, String, cast
from sqlalchemy.orm import sessionmaker, Session
from core.schemas import Base, Image, Embedding256, Embedding512, Embedding128
from core.config import ConfigLoader
from sqlalchemy import and_

db_cfg = ConfigLoader().db
print("[DB CONFIG]", db_cfg)
DB_URL = f"postgresql+psycopg2://{db_cfg['user']}:{db_cfg['password']}@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}"
print("[DB URL]", DB_URL)
engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class VectorRepository:
    def __init__(self, db_session: Session):
        self.db = db_session

    def add_image(self, image_path, label=None, created_at=None):
        image = Image(image_path=image_path, label=label, created_at=created_at)
        self.db.add(image)
        self.db.commit()
        self.db.refresh(image)
        return image

    def get_image_by_path(self, image_path):
        return self.db.query(Image).filter_by(image_path=image_path).first()

    def add_embedding_256(self, image_id, vector_type, parameters, embedding, created_at=None, log=None):
        emb = Embedding256(
            image_id=image_id,
            vector_type=vector_type,
            parameters=parameters,
            embedding=embedding,
            created_at=created_at,
            log=log
        )
        self.db.add(emb)
        self.db.commit()
        self.db.refresh(emb)
        return emb

    def add_embedding_512(self, image_id, vector_type, parameters, embedding, created_at=None, log=None):
        """
        512차원 임베딩 저장. ArcFace origin(원본) 벡터는 vector_type='origin', parameters={}로 저장하는 것이 표준.
        """
        emb = Embedding512(
            image_id=image_id,
            vector_type=vector_type,
            parameters=parameters,
            embedding=embedding,
            created_at=created_at,
            log=log
        )
        self.db.add(emb)
        self.db.commit()
        self.db.refresh(emb)
        return emb

    def add_embedding_128(self, image_id, vector_type, parameters, embedding, created_at=None, log=None):
        emb = Embedding128(
            image_id=image_id,
            vector_type=vector_type,
            parameters=parameters,
            embedding=embedding,
            created_at=created_at,
            log=log
        )
        self.db.add(emb)
        self.db.commit()
        self.db.refresh(emb)
        return emb

    def get_embeddings_256(self, image_id=None, vector_type=None, param_filter=None):
        query = self.db.query(Embedding256)
        if image_id is not None:
            query = query.filter_by(image_id=image_id)
        if vector_type is not None:
            query = query.filter_by(vector_type=vector_type)
        if param_filter is not None:
            for k, v in param_filter.items():
                query = query.filter(cast(Embedding256.parameters[k], String) == str(v))
        return query.all()

    def get_embeddings_512(self, image_id=None, vector_type=None, param_filter=None):
        """
        512차원 임베딩 조회. ArcFace origin(원본) 벡터는 vector_type='origin', parameters={}로 조회.
        """
        query = self.db.query(Embedding512)
        if image_id is not None:
            query = query.filter_by(image_id=image_id)
        if vector_type is not None:
            query = query.filter_by(vector_type=vector_type)
        if param_filter is not None:
            for k, v in param_filter.items():
                query = query.filter(cast(Embedding512.parameters[k], String) == str(v))
        return query.all()

    def get_embeddings_128(self, image_id=None, vector_type=None, param_filter=None):
        query = self.db.query(Embedding128)
        if image_id is not None:
            query = query.filter_by(image_id=image_id)
        if vector_type is not None:
            query = query.filter_by(vector_type=vector_type)
        if param_filter is not None:
            for k, v in param_filter.items():
                query = query.filter(cast(Embedding128.parameters[k], String) == str(v))
        return query.all()

    def find_similar_256(self, query_vec, top_k=5):
        # 256차원 벡터용 HNSW 인덱스 검색 쿼리 예시
        pass

    def find_similar_512(self, query_vec, top_k=5):
        # 512차원 벡터용 HNSW 인덱스 검색 쿼리 예시
        pass 