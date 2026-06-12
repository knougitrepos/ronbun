import numpy as np


def zscore_normalize(vec):
    mean = np.mean(vec)
    std = np.std(vec)
    if std == 0:
        return vec
    return (vec - mean) / std


def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def apply_pca(vec, pca_model):
    return pca_model.transform([vec])[0]


def apply_quantize(vec, n_bits=8):
    min_v, max_v = np.min(vec), np.max(vec)
    scale = (2**n_bits - 1) / (max_v - min_v) if max_v != min_v else 1
    return np.round((vec - min_v) * scale).astype(np.int32)


def dequantize(quantized, min_v, max_v, n_bits=8):
    scale = (2**n_bits - 1) / (max_v - min_v) if max_v != min_v else 1
    return quantized / scale + min_v


def transform_vector(vec, method, **kwargs):
    if method == "zscore":
        return zscore_normalize(vec)
    if method == "l2":
        return l2_normalize(vec)
    if method == "pca":
        return apply_pca(vec, kwargs["pca_model"])
    if method == "quantize":
        return apply_quantize(vec, kwargs.get("n_bits", 8))
    raise ValueError(f"Unknown transform method: {method}")
