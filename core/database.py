from sqlalchemy import create_engine, String, cast, inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import sessionmaker, Session
from core.schemas import Base, Image, Embedding256, Embedding512, Embedding128
from core.config import ConfigLoader
from datetime import datetime
import os

db_cfg = ConfigLoader().db
DB_URL = f"postgresql+psycopg2://{db_cfg['user']}:{db_cfg['password']}@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}"
engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

EXPECTED_TABLES = set(Base.metadata.tables.keys())


def _with_connection(bind: Engine | Connection, fn):
    if isinstance(bind, Engine):
        with bind.connect() as conn:
            return fn(conn)
    return fn(bind)


def ensure_vector_extension(bind: Engine | Connection = engine) -> None:
    def _create(conn: Connection):
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        if not conn.in_transaction():
            conn.commit()

    if isinstance(bind, Engine):
        with bind.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    else:
        _create(bind)


def init_database(bind: Engine | Connection = engine) -> None:
    ensure_vector_extension(bind)
    Base.metadata.create_all(bind=bind)
    ensure_vector_indexes(bind)


def ensure_vector_indexes(bind: Engine | Connection = engine) -> None:
    statements = [
        """
        CREATE INDEX IF NOT EXISTS ix_embedding_512_embedding_hnsw_cosine
        ON embedding_512 USING hnsw (embedding vector_cosine_ops)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_embedding_256_embedding_hnsw_cosine
        ON embedding_256 USING hnsw (embedding vector_cosine_ops)
        """,
    ]

    def _create(conn: Connection):
        for statement in statements:
            conn.execute(text(statement))

    if isinstance(bind, Engine):
        with bind.begin() as conn:
            _create(conn)
    else:
        _create(bind)


def check_database_health(bind: Engine | Connection = engine) -> dict:
    def _check(conn: Connection):
        database, user, server_version = conn.execute(
            text("SELECT current_database(), current_user, version()")
        ).one()
        vector_version = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        existing_tables = set(inspect(conn).get_table_names())
        return {
            "database": database,
            "user": user,
            "server_version": server_version,
            "vector_extension_version": vector_version,
            "existing_tables": sorted(existing_tables),
            "missing_tables": sorted(EXPECTED_TABLES.difference(existing_tables)),
        }

    return _with_connection(bind, _check)


class VectorRepository:
    def __init__(self, db_session: Session, auto_commit: bool = True):
        self.db = db_session
        self.auto_commit = auto_commit

    def _persist(self, obj):
        self.db.add(obj)
        if self.auto_commit:
            self.db.commit()
            self.db.refresh(obj)
        else:
            self.db.flush()
            self.db.refresh(obj)
        return obj

    def add_image(self, image_path, label=None, created_at=None):
        # label이 없으면 폴더명에서 자동 추출
        if label is None:
            label = os.path.basename(os.path.dirname(image_path))
        # created_at이 없으면 현재 시각으로 자동 입력
        if created_at is None:
            created_at = datetime.now()
        image = Image(image_path=image_path, label=label, created_at=created_at)
        return self._persist(image)

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
        return self._persist(emb)

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
        return self._persist(emb)

    def add_embedding_128(self, image_id, vector_type, parameters, embedding, created_at=None, log=None):
        emb = Embedding128(
            image_id=image_id,
            vector_type=vector_type,
            parameters=parameters,
            embedding=embedding,
            created_at=created_at,
            log=log
        )
        return self._persist(emb)

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

    def find_similar_256(self, query_vec, top_k=5, vector_type=None, param_filter=None):
        return self._find_similar(
            Embedding256,
            query_vec,
            expected_dim=256,
            top_k=top_k,
            vector_type=vector_type,
            param_filter=param_filter,
        )

    def find_similar_512(self, query_vec, top_k=5, vector_type=None, param_filter=None):
        return self._find_similar(
            Embedding512,
            query_vec,
            expected_dim=512,
            top_k=top_k,
            vector_type=vector_type,
            param_filter=param_filter,
        )

    def _find_similar(self, model, query_vec, expected_dim, top_k, vector_type=None, param_filter=None):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        values = query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)
        if len(values) != expected_dim:
            raise ValueError(f"query vector must have {expected_dim} dimensions")

        distance = model.embedding.cosine_distance(values)
        query = (
            self.db.query(model, Image, distance.label("distance"))
            .join(Image, model.image_id == Image.id)
        )
        if vector_type is not None:
            query = query.filter(model.vector_type == vector_type)
        if param_filter is not None:
            for key, value in param_filter.items():
                query = query.filter(cast(model.parameters[key], String) == str(value))

        rows = query.order_by(distance).limit(top_k).all()
        results = []
        for embedding, image, row_distance in rows:
            distance_value = float(row_distance)
            results.append(
                {
                    "embedding_id": embedding.id,
                    "image_id": image.id,
                    "image_path": image.image_path,
                    "label": image.label,
                    "vector_type": embedding.vector_type,
                    "parameters": embedding.parameters,
                    "distance": distance_value,
                    "similarity": 1.0 - distance_value,
                }
            )
        return results
