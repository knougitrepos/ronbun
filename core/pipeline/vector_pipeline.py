from core.config import ConfigLoader
from core.pipeline.transformers.wavelet import WaveletTransformer
from core.pipeline.transformers.dct import DCTTransformer

class VectorPipeline:
    def __init__(self, config=None):
        self.config = config or ConfigLoader()
        self.experiment_cfg = self.config.experiment
        # 필요한 설정값 로드

    def preprocess(self, image):
        # 전처리 단계 (예: 얼굴 검출, 정렬 등)
        pass

    def extract_features(self, image):
        # 특징 추출 단계 (ArcFace, EfficientNet 등)
        pass

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
        # DB 저장 단계
        pass

    def run(self, image, transform_method=None, transform_params=None):
        # 전체 파이프라인 실행 예시
        pre_img = self.preprocess(image)
        feat = self.extract_features(pre_img)
        if transform_method:
            vec = self.transform_vector(feat, transform_method, **(transform_params or {}))
        else:
            vec = feat
        self.load_to_db(vec, meta={})
        return vec 