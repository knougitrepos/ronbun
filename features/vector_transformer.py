import numpy as np
from sklearn.decomposition import PCA
from scipy.fft import dct, idct
import pywt

# Z-score 정규화
def zscore_normalize(vec):
    mean = np.mean(vec)
    std = np.std(vec)
    if std == 0:
        return vec
    return (vec - mean) / std

# L2 정규화
def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# PCA 변환 (fit된 pca_model 필요)
def apply_pca(vec, pca_model):
    return pca_model.transform([vec])[0]

# DCT 변환
def apply_dct(vec):
    return dct(vec, norm='ortho')

def apply_idct(vec):
    return idct(vec, norm='ortho')

# Wavelet 변환 (1D)
def apply_wavelet(vec, wavelet_name='haar', level=1):
    coeffs = pywt.wavedec(vec, wavelet_name, level=level)
    return coeffs

def inverse_wavelet(coeffs, wavelet_name='haar'):
    return pywt.waverec(coeffs, wavelet_name)

# Quantize (단순 예시)
def apply_quantize(vec, n_bits=8):
    min_v, max_v = np.min(vec), np.max(vec)
    scale = (2 ** n_bits - 1) / (max_v - min_v) if max_v != min_v else 1
    quantized = np.round((vec - min_v) * scale).astype(np.int32)
    return quantized

def dequantize(quantized, min_v, max_v, n_bits=8):
    scale = (2 ** n_bits - 1) / (max_v - min_v) if max_v != min_v else 1
    return quantized / scale + min_v

# 변환 함수 선택 및 적용 (동적 제어)
def transform_vector(vec, method, **kwargs):
    if method == 'zscore':
        return zscore_normalize(vec)
    elif method == 'l2':
        return l2_normalize(vec)
    elif method == 'pca':
        return apply_pca(vec, kwargs['pca_model'])
    elif method == 'dct':
        return apply_dct(vec)
    elif method == 'wavelet':
        return apply_wavelet(vec, kwargs.get('wavelet_name', 'haar'), kwargs.get('level', 1))
    elif method == 'quantize':
        return apply_quantize(vec, kwargs.get('n_bits', 8))
    else:
        raise ValueError(f'Unknown transform method: {method}') 