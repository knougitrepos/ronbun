import numpy as np
from scipy.fft import dct, idct
import pywt
from PIL import Image
import io

def load_image_gray_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    return np.array(img)

def dct2(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

def idct2(coeffs):
    return idct(idct(coeffs.T, norm='ortho').T, norm='ortho')

def keep_low_freq_dct(coeffs, keep_size):
    out = np.zeros_like(coeffs)
    out[:keep_size, :keep_size] = coeffs[:keep_size, :keep_size]
    return out

def keep_high_freq_dct(coeffs, keep_size):
    out = coeffs.copy()
    out[:keep_size, :keep_size] = 0
    return out

def wavelet_decompose(image, wavelet_name='haar'):
    coeffs = pywt.dwt2(image, wavelet_name)
    LL, (LH, HL, HH) = coeffs
    return LL, LH, HL, HH

def wavelet_reconstruct(LL, LH, HL, HH, wavelet_name='haar'):
    return pywt.idwt2((LL, (LH, HL, HH)), wavelet_name)

def keep_low_freq_wavelet(image, wavelet_name='haar'):
    LL, LH, HL, HH = wavelet_decompose(image, wavelet_name)
    LH[:] = 0
    HL[:] = 0
    HH[:] = 0
    return wavelet_reconstruct(LL, LH, HL, HH, wavelet_name)

def keep_high_freq_wavelet(image, wavelet_name='haar'):
    LL, LH, HL, HH = wavelet_decompose(image, wavelet_name)
    LL[:] = 0
    return wavelet_reconstruct(LL, LH, HL, HH, wavelet_name)

def strong_low_freq_wavelet(image, wavelet_name='haar', level=4):
    coeffs = pywt.wavedec2(image, wavelet_name, level=level)
    coeffs_H = list(coeffs)
    for i in range(1, len(coeffs_H)):
        coeffs_H[i] = tuple([np.zeros_like(v) for v in coeffs_H[i]])
    low_only = pywt.waverec2(coeffs_H, wavelet_name)
    return low_only 