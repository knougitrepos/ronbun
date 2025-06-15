import numpy as np

# L2 정규화 관련 함수 관리
def l2_normalize(vec):
    """L2 정규화 함수"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm
