from core.pipeline.transformers.base import BaseTransformer
from sklearn.decomposition import PCA
import numpy as np

class PCATransformer(BaseTransformer):
    def __init__(self, n_components=256):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.fitted = False

    def fit(self, X):
        self.pca.fit(X)
        self.fitted = True

    def transform(self, vec):
        if not self.fitted:
            raise RuntimeError("PCA 모델이 fit되지 않았습니다.")
        return self.pca.transform(np.array([vec]))[0] 