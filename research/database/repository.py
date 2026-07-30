from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import math
import os
from time import perf_counter

from sqlalchemy import text
from sqlalchemy.orm import Session

from research.database.models import (
    PCA_EMBEDDING_MODELS,
    PCA_TEMPLATE_MODELS,
    Embedding256,
    Embedding512,
    EmbeddingPQ,
    Image,
    TemplateEmbedding256,
    TemplateEmbedding512,
)


@contextmanager
def _temporary_local_settings(session: Session, settings: dict[str, str]):
    """Apply transaction-local planner settings and restore their prior values."""

    previous: dict[str, str | None] = {}
    original_error: BaseException | None = None
    try:
        for name, value in settings.items():
            previous[name] = session.execute(
                text("SELECT current_setting(:name, true)"), {"name": name}
            ).scalar_one_or_none()
            session.execute(
                text("SELECT set_config(:name, :value, true)"),
                {"name": name, "value": value},
            )
        yield
    except BaseException as exc:
        original_error = exc
        raise
    finally:
        try:
            for name, value in reversed(previous.items()):
                if value is not None:
                    session.execute(
                        text("SELECT set_config(:name, :value, true)"),
                        {"name": name, "value": value},
                    )
        except Exception:
            if original_error is None:
                raise


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

    def upsert_embedding_256(
        self,
        image_id,
        vector_type,
        parameters,
        embedding,
        created_at=None,
        log=None,
        *,
        run_uid: str,
        replace_if_changed: bool = False,
    ):
        return self._upsert_embedding(
            Embedding256,
            image_id=image_id,
            vector_type=vector_type,
            parameters=parameters,
            payload_name="embedding",
            payload=embedding,
            created_at=created_at,
            log=log,
            run_uid=run_uid,
            replace_if_changed=replace_if_changed,
        )

    def upsert_embedding_pq(
        self,
        image_id,
        vector_type,
        parameters,
        codes: bytes,
        created_at=None,
        log=None,
        *,
        run_uid: str,
        replace_if_changed: bool = False,
    ):
        return self._upsert_embedding(
            EmbeddingPQ,
            image_id=image_id,
            vector_type=vector_type,
            parameters=parameters,
            payload_name="codes",
            payload=codes,
            created_at=created_at,
            log=log,
            run_uid=run_uid,
            replace_if_changed=replace_if_changed,
        )

    def _upsert_embedding(
        self,
        model,
        *,
        image_id,
        vector_type,
        parameters,
        payload_name,
        payload,
        created_at,
        log,
        run_uid,
        replace_if_changed,
    ):
        if not run_uid:
            raise ValueError("run_uid is required for reproducible embedding upsert")
        existing = self._embedding_for_run(model, image_id, vector_type, run_uid)
        if existing is None:
            values = {
                "image_id": image_id,
                "run_uid": run_uid,
                "vector_type": vector_type,
                "parameters": parameters,
                payload_name: payload,
                "created_at": created_at or datetime.now(timezone.utc),
                "log": log,
            }
            return self._persist(model(**values)), "inserted"
        if existing.parameters == parameters:
            return existing, "skipped"
        if not replace_if_changed:
            raise ValueError(
                "an embedding for this run/image/profile already exists with different "
                "provenance; start a new run or explicitly allow replacement"
            )
        existing.parameters = parameters
        setattr(existing, payload_name, payload)
        existing.log = log
        self.db.flush()
        self.db.refresh(existing)
        return existing, "updated"

    def _embedding_for_run(self, model, image_id, vector_type, run_uid):
        if run_uid is None:
            return None
        return (
            self.db.query(model)
            .filter_by(image_id=image_id, vector_type=vector_type, run_uid=run_uid)
            .first()
        )

    def upsert_template_512(self, **values):
        return self._upsert_template(TemplateEmbedding512, **values)

    def upsert_template_256(self, **values):
        return self._upsert_template(TemplateEmbedding256, **values)

    def upsert_pca_template(self, dimension: int, **values):
        try:
            model = PCA_TEMPLATE_MODELS[int(dimension)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported PCA template dimension: {dimension}"
            ) from exc
        return self._upsert_template(model, **values)

    def _upsert_template(
        self,
        model,
        *,
        run_uid: str,
        protocol_name: str,
        vector_type: str,
        aggregation_method: str,
        enrollment_policy: str,
        enrollment_target: int,
        enrollment_count: int,
        identity_id: str,
        model_uid: str,
        source_image_ids,
        embedding,
        quality=None,
        variance=None,
        angular_error=None,
        reconstruction_error_norm=None,
        parameters=None,
        created_at=None,
        replace_if_changed: bool = False,
    ):
        required_text = {
            "run_uid": run_uid,
            "protocol_name": protocol_name,
            "vector_type": vector_type,
            "aggregation_method": aggregation_method,
            "identity_id": identity_id,
            "model_uid": model_uid,
        }
        missing = [
            name
            for name, value in required_text.items()
            if value is None or not str(value).strip()
        ]
        if missing:
            raise ValueError(f"template scope values must not be empty: {missing}")
        if enrollment_policy not in {"fixed", "official_all"}:
            raise ValueError("enrollment_policy must be fixed or official_all")
        if enrollment_policy == "fixed" and enrollment_target < 1:
            raise ValueError("fixed enrollment_target must be positive")
        if enrollment_policy == "official_all" and enrollment_target != 0:
            raise ValueError("official_all enrollment_target must be 0")
        if enrollment_count < 1:
            raise ValueError("enrollment_count must be positive")
        source_ids = list(source_image_ids)
        if len(source_ids) != enrollment_count or len(set(source_ids)) != enrollment_count:
            raise ValueError(
                "enrollment_count must equal the number of unique source_image_ids"
            )
        key = {
            "run_uid": run_uid,
            "protocol_name": protocol_name,
            "vector_type": vector_type,
            "aggregation_method": aggregation_method,
            "enrollment_policy": enrollment_policy,
            "enrollment_target": enrollment_target,
            "identity_id": str(identity_id),
            "model_uid": model_uid,
        }
        existing = self.db.query(model).filter_by(**key).first()
        values = {
            "source_image_ids": source_ids,
            "enrollment_count": enrollment_count,
            "embedding": embedding,
            "quality": quality,
            "variance": variance,
            "angular_error": angular_error,
            "reconstruction_error_norm": reconstruction_error_norm,
            "parameters": parameters,
        }
        if existing is None:
            return self._persist(
                model(**key, **values, created_at=created_at or datetime.now(timezone.utc))
            ), "inserted"
        comparable = {
            name: getattr(existing, name)
            for name in (
                "source_image_ids",
                "enrollment_count",
                "quality",
                "variance",
                "angular_error",
                "reconstruction_error_norm",
                "parameters",
            )
        }
        if comparable == {name: value for name, value in values.items() if name != "embedding"}:
            return existing, "skipped"
        if not replace_if_changed:
            raise ValueError(
                "a template for this run/protocol/profile already exists with different "
                "provenance; start a new run or explicitly allow replacement"
            )
        for name, value in values.items():
            setattr(existing, name, value)
        self.db.flush()
        self.db.refresh(existing)
        return existing, "updated"

    def find_similar_templates_512(self, query_vec, **kwargs):
        return self._find_similar_templates(
            TemplateEmbedding512, query_vec, expected_dim=512, **kwargs
        )

    def find_similar_templates_256(self, query_vec, **kwargs):
        return self._find_similar_templates(
            TemplateEmbedding256, query_vec, expected_dim=256, **kwargs
        )

    def find_similar_pca_templates(self, dimension: int, query_vec, **kwargs):
        try:
            expected_dim = int(dimension)
            model = PCA_TEMPLATE_MODELS[expected_dim]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported PCA template dimension: {dimension}"
            ) from exc
        return self._find_similar_templates(
            model, query_vec, expected_dim=expected_dim, **kwargs
        )

    def upsert_pca_embedding(
        self,
        dimension: int,
        image_id,
        vector_type,
        parameters,
        embedding,
        created_at=None,
        log=None,
        *,
        run_uid: str,
        replace_if_changed: bool = False,
    ):
        try:
            model = PCA_EMBEDDING_MODELS[int(dimension)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported PCA embedding dimension: {dimension}"
            ) from exc
        return self._upsert_embedding(
            model,
            image_id=image_id,
            vector_type=vector_type,
            parameters=parameters,
            payload_name="embedding",
            payload=embedding,
            created_at=created_at,
            log=log,
            run_uid=run_uid,
            replace_if_changed=replace_if_changed,
        )

    def _find_similar_templates(
        self,
        model,
        query_vec,
        *,
        expected_dim: int,
        run_uid: str,
        protocol_name: str,
        vector_type: str,
        aggregation_method: str,
        enrollment_policy: str,
        enrollment_target: int,
        model_uid: str,
        top_k: int = 5,
        search_mode: str = "hnsw",
        ef_search: int | None = None,
    ):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if search_mode not in {"exact", "hnsw"}:
            raise ValueError("search_mode must be 'exact' or 'hnsw'")
        if ef_search is not None and (
            isinstance(ef_search, bool)
            or not isinstance(ef_search, int)
            or ef_search < 1
        ):
            raise ValueError("ef_search must be a positive integer")
        values = query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)
        if len(values) != expected_dim:
            raise ValueError(f"query vector must have {expected_dim} dimensions")
        try:
            values = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError("query vector must contain numeric values") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("query vector must contain only finite values")

        planner_settings = (
            {
                "enable_indexscan": "off",
                "enable_indexonlyscan": "off",
                "enable_bitmapscan": "off",
            }
            if search_mode == "exact"
            else {
                "enable_seqscan": "off",
                "hnsw.iterative_scan": "strict_order",
                **({"hnsw.ef_search": str(ef_search)} if ef_search is not None else {}),
            }
        )

        distance = model.embedding.cosine_distance(values)
        query = self.db.query(model, distance.label("distance")).filter_by(
            run_uid=run_uid,
            protocol_name=protocol_name,
            vector_type=vector_type,
            aggregation_method=aggregation_method,
            enrollment_policy=enrollment_policy,
            enrollment_target=enrollment_target,
            model_uid=model_uid,
        )
        with _temporary_local_settings(self.db, planner_settings):
            started = perf_counter()
            rows = query.order_by(distance).limit(top_k).all()
            elapsed_ms = (perf_counter() - started) * 1000.0

        return [
            {
                "template_id": template.id,
                "identity_id": template.identity_id,
                "source_image_ids": template.source_image_ids,
                "enrollment_policy": template.enrollment_policy,
                "enrollment_target": template.enrollment_target,
                "enrollment_count": template.enrollment_count,
                "quality": template.quality,
                "variance": template.variance,
                "angular_error": template.angular_error,
                "reconstruction_error_norm": template.reconstruction_error_norm,
                "parameters": template.parameters,
                "distance": float(row_distance),
                "similarity": 1.0 - float(row_distance),
                "search_mode": search_mode,
                "query_elapsed_ms": elapsed_ms,
                "ef_search": ef_search if search_mode == "hnsw" else None,
            }
            for template, row_distance in rows
        ]

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
