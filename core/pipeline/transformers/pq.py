import faiss
import numpy as np
import os

class PQTransformer:
    def __init__(self, d, M=16, nbits=8):
        self.d = d
        self.M = M
        self.nbits = nbits
        self.index = faiss.IndexPQ(d, M, nbits)
        self.trained = False
        self.last_log = None

    def fit(self, X):
        X = np.ascontiguousarray(X.astype(np.float32))
        self.index.train(X)
        self.trained = True
        log = self._log_quantization_error(X)
        self.last_log = log
        print(log)
        return self, log

    def transform(self, X):
        if not self.trained:
            raise ValueError("PQ 인덱스가 train되지 않았습니다.")
        X = np.ascontiguousarray(X.astype(np.float32))
        return self.index.sa_encode(X)

    def fit_transform(self, X):
        self.fit(X)
        codes = self.transform(X)
        log = self.last_log
        return codes, log

    def _log_quantization_error(self, X):
        # PQ로 양자화 후 재구성 오차(MSE) 계산
        codes = self.index.sa_encode(X)
        X_rec = self.index.sa_decode(codes)
        mse = np.mean((X - X_rec) ** 2)
        log = f"[PQ] d: {self.d}, M: {self.M}, nbits: {self.nbits}, quantization MSE: {mse:.6f}"
        if mse > 0.1:
            log += f"\n[경고] PQ 양자화 오차가 큼! (MSE={mse:.6f})"
        return log

    def save_codebook(self, codebook_dir="codebook"):
        if not self.trained:
            raise ValueError("PQ 인덱스가 train되지 않았습니다.")
        os.makedirs(codebook_dir, exist_ok=True)
        path = os.path.join(codebook_dir, f"pq_{self.d}.faiss")
        faiss.write_index(self.index, path)
        return path

    @staticmethod
    def load_codebook(d, M=16, nbits=8, codebook_dir="codebook"):
        path = os.path.join(codebook_dir, f"pq_{d}.faiss")
        index = faiss.read_index(path)
        transformer = PQTransformer(d=d, M=M, nbits=nbits)
        transformer.index = index
        transformer.trained = True
        return transformer

    def save_or_load_codebook(self, X, codebook_dir="codebook"):
        path = os.path.join(codebook_dir, f"pq_{self.d}.faiss")
        if os.path.exists(path):
            print(f"[PQ] 기존 코드북을 로드합니다: {path}")
            self.index = faiss.read_index(path)
            self.trained = True
            return self
        else:
            print(f"[PQ] 코드북이 없어 새로 생성합니다: {path}")
            self.fit(X)
            self.save_codebook(codebook_dir)
            return self 