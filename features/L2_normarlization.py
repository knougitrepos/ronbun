import numpy as np

def l2_normalize(vec):
    """L2 정규화 함수"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm
