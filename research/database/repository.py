from __future__ import annotations

from datetime import datetime, timezone
import os

from sqlalchemy.orm import Session

from research.database.models import Embedding256, Embedding512, EmbeddingPQ, Image


class VectorRepository:
    def __init__(self, db_session: Session, *, auto_commit: bool = False):
        self.db = db_session
        self.auto_commit = auto_commit

    def _persist(self, obj):
        self.db.add(obj)
        if self.auto_commit:
            self.db.commit()
        else:
            self.db.flush()
        self.db.refresh(obj)
        return obj

    def commit(self) -> None:
        self.db.commit()

    def rollback(self) -> None:
        self.db.rollback()

    def add_image(
        self,
        image_path: str,
        label: str | None = None,
        created_at=None,
        *,
        content_sha256: str | None = None,
        file_size_bytes: int | None = None,
    ):
        existing = self.get_image_by_path(image_path)
        if existing is not None:
            if content_sha256 and existing.content_sha256 not in {None, content_sha256}:
                raise ValueError(f"image content changed at an existing path: {image_path}")
            if content_sha256 and existing.content_sha256 is None:
                existing.content_sha256 = content_sha256
                existing.file_size_bytes = file_size_bytes
                self.db.flush()
            return existing
        resolved_label = label or os.path.basename(os.path.dirname(image_path))
        image = Image(
            image_path=image_path,
            label=resolved_label,
            content_sha256=content_sha256,
            file_size_bytes=file_size_bytes,
            created_at=created_at or datetime.now(timezone.utc),
        )
        return self._persist(image)

    def get_image_by_path(self, image_path: str):
        return self.db.query(Image).filter_by(image_path=image_path).first()

    def add_embedding_256(
        self,
        image_id,
        vector_type,
        parameters,
        embedding,
        created_at=None,
        log=None,
        *,
        run_uid: str | None = None,
    ):
        existing = self._embedding_for_run(Embedding256, image_id, vector_type, run_uid)
        if existing is not None:
            return existing
        return self._persist(
            Embedding256(
                image_id=image_id,
                run_uid=run_uid,
                vector_type=vector_type,
                parameters=parameters,
                embedding=embedding,
                created_at=created_at or datetime.now(timezone.utc),
                log=log,
            )
        )

    def add_embedding_512(
        self,
        image_id,
        vector_type,
        parameters,
        embedding,
        created_at=None,
        log=None,
        *,
        run_uid: str | None = None,
    ):
        existing = self._embedding_for_run(Embedding512, image_id, vector_type, run_uid)
        if existing is not None:
            return existing
        return self._persist(
            Embedding512(
                image_id=image_id,
                run_uid=run_uid,
                vector_type=vector_type,
                parameters=parameters,
                embedding=embedding,
                created_at=created_at or datetime.now(timezone.utc),
                log=log,
            )
        )

    def add_embedding_pq(
        self,
        image_id,
        vector_type,
        parameters,
        codes: bytes,
        created_at=None,
        log=None,
        *,
        run_uid: str | None = None,
    ):
        existing = self._embedding_for_run(EmbeddingPQ, image_id, vector_type, run_uid)
        if existing is not None:
            return existing
        return self._persist(
            EmbeddingPQ(
                image_id=image_id,
                run_uid=run_uid,
                vector_type=vector_type,
                parameters=parameters,
                codes=codes,
                created_at=created_at or datetime.now(timezone.utc),
                log=log,
            )
        )

    def _embedding_for_run(self, model, image_id, vector_type, run_uid):
        if run_uid is None:
            return None
        return (
            self.db.query(model)
            .filter_by(image_id=image_id, vector_type=vector_type, run_uid=run_uid)
            .first()
        )

    def get_embeddings_256(
        self, image_id=None, vector_type=None, param_filter=None, *, run_uid=None
    ):
        return self._get_embeddings(Embedding256, image_id, vector_type, param_filter, run_uid)

    def get_embeddings_512(
        self, image_id=None, vector_type=None, param_filter=None, *, run_uid=None
    ):
        return self._get_embeddings(Embedding512, image_id, vector_type, param_filter, run_uid)

    def _get_embeddings(self, model, image_id, vector_type, param_filter, run_uid):
        query = self.db.query(model)
        if image_id is not None:
            query = query.filter_by(image_id=image_id)
        if vector_type is not None:
            query = query.filter_by(vector_type=vector_type)
        if run_uid is not None:
            query = query.filter_by(run_uid=run_uid)
        if param_filter is not None:
            for key, value in param_filter.items():
                query = query.filter(model.parameters[key].as_string() == str(value))
        return query.all()

    def find_similar_256(
        self, query_vec, top_k=5, vector_type=None, param_filter=None, *, run_uid=None
    ):
        return self._find_similar(
            Embedding256,
            query_vec,
            expected_dim=256,
            top_k=top_k,
            vector_type=vector_type,
            param_filter=param_filter,
            run_uid=run_uid,
        )

    def find_similar_512(
        self, query_vec, top_k=5, vector_type=None, param_filter=None, *, run_uid=None
    ):
        return self._find_similar(
            Embedding512,
            query_vec,
            expected_dim=512,
            top_k=top_k,
            vector_type=vector_type,
            param_filter=param_filter,
            run_uid=run_uid,
        )

    def _find_similar(
        self, model, query_vec, expected_dim, top_k, vector_type, param_filter, run_uid
    ):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        values = query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)
        if len(values) != expected_dim:
            raise ValueError(f"query vector must have {expected_dim} dimensions")

        distance = model.embedding.cosine_distance(values)
        query = self.db.query(model, Image, distance.label("distance")).join(
            Image, model.image_id == Image.id
        )
        if vector_type is not None:
            query = query.filter(model.vector_type == vector_type)
        if run_uid is not None:
            query = query.filter(model.run_uid == run_uid)
        if param_filter is not None:
            for key, value in param_filter.items():
                query = query.filter(model.parameters[key].as_string() == str(value))

        rows = query.order_by(distance).limit(top_k).all()
        return [
            {
                "embedding_id": embedding.id,
                "image_id": image.id,
                "image_path": image.image_path,
                "label": image.label,
                "vector_type": embedding.vector_type,
                "parameters": embedding.parameters,
                "distance": float(row_distance),
                "similarity": 1.0 - float(row_distance),
            }
            for embedding, image, row_distance in rows
        ]
