import numpy as np

def zscore_normalize(vec):
    """Z-score 정규화 함수"""
    mean = np.mean(vec)
    std = np.std(vec)
    if std == 0:
        return vec
    return (vec - mean) / std
