from core.config import ConfigLoader
from core.pipeline.transformers.wavelet import WaveletTransformer
from core.pipeline.transformers.dct import DCTTransformer
import numpy as np
from features.embedding_service import ArcFaceFeatureExtractor
from core.database import SessionLocal, VectorRepository
import os

class VectorPipeline:
    def __init__(self, config=None):
        self.config = config or ConfigLoader()
        self.experiment_cfg = self.config.experiment
        self.extractor = ArcFaceFeatureExtractor()
        # 필요한 설정값 로드

    def preprocess(self, image):
        # (필요시 전처리 추가)
        return image

    def extract_features(self, image):
        # ArcFace 임베딩 추출
        return self.extractor.extract(image)

    def transform_vector(self, vec, method, **kwargs):
        # method: 'wavelet', 'dct', 'pca', 'pq' ...
        if method == 'wavelet':
            transformer = WaveletTransformer(
                wavelet_name=kwargs.get('wavelet_name', 'haar'),
                level=kwargs.get('level', 1),
                mode=kwargs.get('mode', 'low')
            )
        elif method == 'dct':
            transformer = DCTTransformer(
                keep_dim=kwargs.get('keep_dim', None),
                mode=kwargs.get('mode', 'low')
            )
        elif method == 'pca':
            from core.pipeline.transformers.pca import PCATransformer
            n_components = kwargs.get('n_components', 0.95)
            transformer = PCATransformer(n_components=n_components)
            # fit_transform 필요: vec shape (N, D)
            return transformer.fit_transform(vec)
        elif method == 'pq':
            from core.pipeline.transformers.pq import PQTransformer
            d = kwargs.get('d', vec.shape[1] if len(vec.shape) > 1 else len(vec))
            M = kwargs.get('M', 16)
            nbits = kwargs.get('nbits', 8)
            transformer = PQTransformer(d=d, M=M, nbits=nbits)
            # fit_transform 필요: vec shape (N, D)
            return transformer.fit_transform(vec)
        else:
            raise ValueError(f'Unknown transform method: {method}')
        return transformer.transform(vec)

    def load_to_db(self, vec, meta):
        # DB 저장
        image_path = meta.get('image_path')
        if not image_path:
            print(f"[에러] image_path가 meta에 없습니다: meta={meta}")
            return  # 또는 raise Exception("image_path is required")
        # label 추출: 폴더명
        label = os.path.basename(os.path.dirname(image_path))
        session = SessionLocal()
        repo = VectorRepository(session)
        image = repo.get_image_by_path(image_path)
        if image is None:
            image = repo.add_image(image_path, label=label)
        repo.add_embedding_512(
            image_id=image.id,
            vector_type='origin',
            parameters={},
            embedding=vec
        )
        session.close()

    def run(self, image, transform_method=None, transform_params=None, meta=None):
        pre_img = self.preprocess(image)
        feat = self.extract_features(pre_img)
        if transform_method:
            vec = self.transform_vector(feat, transform_method, **(transform_params or {}))
        else:
            vec = feat
        self.load_to_db(vec, meta or {})
        return vec 