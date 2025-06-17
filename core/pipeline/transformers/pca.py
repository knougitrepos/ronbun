from core.pipeline.transformers.base import BaseTransformer
from sklearn.decomposition import PCA
import numpy as np
import joblib
import os

class PCATransformer(BaseTransformer):
    def __init__(self, n_components=0.95):
        self.n_components = n_components
        self.pca = None
        self.last_log = None

    def fit(self, X):
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
        log = self._log_variance()
        self.last_log = log
        print(log)
        return self, log

    def transform(self, X):
        if self.pca is None:
            raise ValueError("PCA 모델이 fit되지 않았습니다.")
        return self.pca.transform(X)

    def fit_transform(self, X):
        self.pca = PCA(n_components=self.n_components)
        X_new = self.pca.fit_transform(X)
        log = self._log_variance()
        self.last_log = log
        print(log)
        return X_new, log

    def _log_variance(self):
        if self.pca is None:
            return "[PCA] 모델이 fit되지 않았습니다."
        evr = self.pca.explained_variance_ratio_
        cum_evr = np.cumsum(evr)
        n_dim = self.pca.n_components_ if hasattr(self.pca, 'n_components_') else self.n_components
        log = f"[PCA] n_components: {n_dim}, explained_variance_ratio: {evr}, cumulative: {cum_evr[-1]:.4f}"
        if cum_evr[-1] < 0.95:
            log += f"\n[경고] 누적 분산 설명력 0.95 미만! ({cum_evr[-1]:.4f})"
        return log

    def save_codebook(self, codebook_dir="codebook"):
        if self.pca is None:
            raise ValueError("PCA 모델이 fit되지 않았습니다.")
        n_dim = self.pca.n_components_ if hasattr(self.pca, 'n_components_') else self.n_components
        os.makedirs(codebook_dir, exist_ok=True)
        path = os.path.join(codebook_dir, f"pca_{n_dim}.joblib")
        joblib.dump(self.pca, path)
        return path

    @staticmethod
    def load_codebook(n_dim, codebook_dir="codebook"):
        path = os.path.join(codebook_dir, f"pca_{n_dim}.joblib")
        pca = joblib.load(path)
        transformer = PCATransformer(n_components=n_dim)
        transformer.pca = pca
        return transformer 