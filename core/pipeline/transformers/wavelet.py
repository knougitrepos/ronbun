from core.pipeline.transformers.base import BaseTransformer
import pywt
import numpy as np

class WaveletTransformer(BaseTransformer):
    def __init__(self, wavelet_name='haar', level=1, mode='low'):
        self.wavelet_name = wavelet_name
        self.level = level
        self.mode = mode  # 'low' or 'high'

    def transform(self, vec):
        coeffs = pywt.wavedec(vec, self.wavelet_name, level=self.level)
        if self.mode == 'low':
            return coeffs[0]  # 저주파 성분
        elif self.mode == 'high':
            # 고주파: 모든 세부 성분을 합쳐 반환
            return np.concatenate(coeffs[1:])
        else:
            raise ValueError("mode는 'low' 또는 'high'만 허용") 