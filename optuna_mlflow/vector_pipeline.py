import numpy as np
from sklearn.decomposition import PCA

def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def zscore_normalize(vec):
    mean = np.mean(vec)
    std = np.std(vec)
    if std == 0:
        return vec
    return (vec - mean) / std

def apply_pca(vec, n_components=256):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(vec)

def apply_pq(vec, n_bits=8):
    # PQ 양자화 샘플(실제 구현 필요)
    return vec.astype(np.int32) % (2 ** n_bits) 