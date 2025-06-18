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
        log = f"[Wavelet] wavelet_name: {self.wavelet_name}, level: {self.level}, mode: {self.mode}, input_dim: {len(vec)}, coeffs_len: {[len(c) for c in coeffs]}"
        if self.mode == 'low':
            return coeffs[0], log
        elif self.mode == 'high':
            return np.concatenate(coeffs[1:]), log
        else:
            raise ValueError("mode는 'low' 또는 'high'만 허용") 