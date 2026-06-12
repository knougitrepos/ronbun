from core.config import ConfigLoader
import numpy as np
from features.embedding_service import ArcFaceFeatureExtractor, NoneFaceDetectFeatureExtractor
from core.database import SessionLocal, VectorRepository
import os

class VectorPipeline:
    def __init__(self, config=None, extractor=None):
        self.config = config or ConfigLoader()
        self.experiment_cfg = self.config.experiment
        self.extractor = extractor  # None이면 extract_features에서 자동 지정
        # 필요한 설정값 로드

    def preprocess(self, image):
        # (필요시 전처리 추가)
        return image

    def extract_features(self, image):
        # extractor가 지정되어 있으면 사용, 아니면 얼굴 검출 없는 extractor 사용
        if self.extractor is not None:
            return self.extractor.extract(image)
        else:
            return NoneFaceDetectFeatureExtractor().extract(image)

    def transform_vector(self, vec, method, **kwargs):
        # Thesis scope: ArcFace origin -> PCA -> PQ.
        if method == 'pca':
            from core.pipeline.transformers.pca import PCATransformer
            n_components = kwargs.get('n_components', 0.95)
            transformer = PCATransformer(n_components=n_components)
            transformer.save_or_load_codebook(vec)
            result = transformer.transform(vec)
            log = transformer.last_log
        elif method == 'pq':
            from core.pipeline.transformers.pq import PQTransformer
            d = kwargs.get('d', vec.shape[1] if len(vec.shape) > 1 else len(vec))
            M = kwargs.get('M', 16)
            nbits = kwargs.get('nbits', 8)
            transformer = PQTransformer(d=d, M=M, nbits=nbits)
            result, log = transformer.fit_transform(vec)
        else:
            raise ValueError(f'Unknown transform method: {method}')
        print(log)
        return result, log

    def load_to_db(self, vec, meta, dim=None, vector_type='origin', parameters=None, log=None):
        # DB 저장
        image_path = meta.get('image_path')
        if not image_path:
            print(f"[에러] image_path가 meta에 없습니다: meta={meta}")
            return
        label = os.path.basename(os.path.dirname(image_path))
        session = SessionLocal()
        repo = VectorRepository(session)
        image = repo.get_image_by_path(image_path)
        if image is None:
            image = repo.add_image(image_path, label=label)
        if dim is None:
            dim = vec.shape[0]
        # 실제 벡터 차원과 저장하려는 dim이 다르면 저장하지 않음
        if vec.shape[0] != dim:
            print(f"[WARN] Vector dim mismatch: expected {dim}, got {vec.shape[0]}. Not saving to DB.")
            session.close()
            return
        # 절단, 패딩 없이 지원 차원만 저장
        if dim == 512:
            repo.add_embedding_512(
                image_id=image.id,
                vector_type=vector_type,
                parameters=parameters or {},
                embedding=vec,
                log=log
            )
        elif dim == 256:
            repo.add_embedding_256(
                image_id=image.id,
                vector_type=vector_type,
                parameters=parameters or {},
                embedding=vec,
                log=log
            )
        elif dim == 128:
            repo.add_embedding_128(
                image_id=image.id,
                vector_type=vector_type,
                parameters=parameters or {},
                embedding=vec,
                log=log
            )
        else:
            print(f"[WARN] Unsupported embedding dimension: {dim} (Not saving, no truncation/padding)")
        session.close()

    def run(self, image, transform_method=None, transform_params=None, meta=None):
        pre_img = self.preprocess(image)
        feat = self.extract_features(pre_img)
        if transform_method:
            vec, log = self.transform_vector(feat, transform_method, **(transform_params or {}))
        else:
            vec = feat
        self.load_to_db(vec, meta or {}, 512, log=log)
        return vec
