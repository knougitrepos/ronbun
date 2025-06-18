from core.pipeline.transformers.base import BaseTransformer
from scipy.fft import dct
import numpy as np

class DCTTransformer(BaseTransformer):
    def __init__(self, keep_dim=None, mode='low'):
        self.keep_dim = keep_dim  # None이면 전체 사용
        self.mode = mode  # 'low' or 'high'

    def transform(self, vec):
        dct_vec = dct(vec, norm='ortho')
        log = f"[DCT] keep_dim: {self.keep_dim}, mode: {self.mode}, input_dim: {len(vec)}"
        if self.keep_dim is not None:
            if self.mode == 'low':
                dct_vec[self.keep_dim:] = 0
            elif self.mode == 'high':
                dct_vec[:self.keep_dim] = 0
            else:
                raise ValueError("mode는 'low' 또는 'high'만 허용")
        return dct_vec, log 