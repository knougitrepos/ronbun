from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import os
from features.origin_feature_extractor import extract_arcface_feature
from DB.db_utils import SessionLocal, ImageEmbeddings
from features.extract_example import load_image_gray_from_bytes, dct2, idct2, keep_low_freq_dct, keep_high_freq_dct, keep_low_freq_wavelet, keep_high_freq_wavelet
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import numpy as np
import pywt
import uuid
import matplotlib
from features.face_analysis_loader import get_face_analysis_app

templates = Jinja2Templates(directory="templates")

router = APIRouter()

@router.get("/extract", response_class=HTMLResponse, description="특징추출 페이지. 다양한 특징추출 모델 및 변환 옵션 제공.")
async def extract(request: Request):
    """특징추출 페이지: 다양한 특징추출 모델 및 변환 옵션 제공"""
    return templates.TemplateResponse("extract.html", {"request": request})

@router.post("/extract_features/origin")
async def extract_origin_features(request: Request):
    # 1. downloaded_datasets 내 모든 이미지 경로 탐색
    dataset_root = "downloaded_datasets"
    image_paths = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(root, file))

    def arcface_extract(image_path):
        return extract_arcface_feature(image_path)

    db = SessionLocal()
    for img_path in image_paths:
        is_extract_face = False
        try:
            vec_origin = arcface_extract(img_path)
            is_extract_face = True
        except Exception as e:
            print(f"특징 추출 에러: {img_path}, {e}")
            vec_origin = None
        label = os.path.basename(os.path.dirname(img_path))
        used_feature_extract_model = "ArcFace"
        used_distance_model = "cosine"
        try:
            obj = db.query(ImageEmbeddings).filter_by(image_path=img_path).first()
            if obj:
                obj.label = label
                obj.used_feature_extract_model = used_feature_extract_model
                obj.used_distance_model = used_distance_model
                obj.vec_origin = vec_origin
                obj.is_extract_face = is_extract_face
                from datetime import datetime
                obj.updated_at = datetime.now()
            else:
                from datetime import datetime
                obj = ImageEmbeddings(
                    image_path=img_path,
                    label=label,
                    used_feature_extract_model=used_feature_extract_model,
                    used_distance_model=used_distance_model,
                    vec_origin=vec_origin,
                    is_extract_face=is_extract_face,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                db.add(obj)
            db.commit()
        except Exception as e:
            print(f"DB 저장 에러: {img_path}, {e}")
            db.rollback()
    db.close()
    return RedirectResponse(url="/extract", status_code=303)

@router.get("/extract_example", response_class=HTMLResponse)
async def extract_example_get(request: Request):
    return templates.TemplateResponse("extract_example.html", {"request": request, "result_imgs": None})

@router.post("/extract_example", response_class=HTMLResponse)
async def extract_example_post(request: Request, image: UploadFile = File(...)):
    matplotlib.rc('font', family='Malgun Gothic')
    matplotlib.rcParams['axes.unicode_minus'] = False
    img_bytes = await image.read()
    image_np = load_image_gray_from_bytes(img_bytes)
    wavelet_families = [
        'haar', 'db2', 'db4', 'sym2', 'sym4', 'coif1', 'bior1.3', 'rbio1.3'
    ]
    max_levels = []
    for wname in wavelet_families:
        try:
            max_level = pywt.dwt_max_level(min(image_np.shape), pywt.Wavelet(wname).dec_len)
        except Exception:
            max_level = 1
        max_levels.append(max_level)
    max_level_all = max(max_levels)
    def strong_low_freq_wavelet(image, wavelet_name='haar', level=4):
        coeffs = pywt.wavedec2(image, wavelet_name, level=level)
        coeffs_H = list(coeffs)
        for i in range(1, len(coeffs_H)):
            coeffs_H[i] = tuple([np.zeros_like(v) for v in coeffs_H[i]])
        low_only = pywt.waverec2(coeffs_H, wavelet_name)
        return low_only
    def strong_high_freq_wavelet(image, wavelet_name='haar', level=4):
        coeffs = pywt.wavedec2(image, wavelet_name, level=level)
        coeffs_H = list(coeffs)
        coeffs_H[0] = np.zeros_like(coeffs_H[0])
        high_only = pywt.waverec2(coeffs_H, wavelet_name)
        return high_only
    # 저주파 전체 시각화
    fig1, axes1 = plt.subplots(len(wavelet_families), max_level_all, figsize=(5 * max_level_all, 5 * len(wavelet_families)))
    for fam_idx, wname in enumerate(wavelet_families):
        max_level = max_levels[fam_idx]
        for level in range(1, max_level+1):
            try:
                low_freq_img = strong_low_freq_wavelet(image_np, wavelet_name=wname, level=level)
                ax = axes1[fam_idx, level-1] if len(wavelet_families) > 1 else axes1[level-1]
                ax.set_title(f"{wname}\nlvl={level}")
                ax.imshow(np.clip(low_freq_img,0,255), cmap='gray')
                ax.axis('off')
            except Exception as e:
                ax = axes1[fam_idx, level-1] if len(wavelet_families) > 1 else axes1[level-1]
                ax.set_title(f"{wname}\nlvl={level}\nError")
                ax.axis('off')
        for level in range(max_level+1, max_level_all+1):
            ax = axes1[fam_idx, level-1] if len(wavelet_families) > 1 else axes1[level-1]
            ax.axis('off')
    plt.tight_layout()
    save_dir = os.path.join("static", "extract_example")
    os.makedirs(save_dir, exist_ok=True)
    file_id1 = str(uuid.uuid4())
    save_path1 = os.path.join(save_dir, f"result_low_{file_id1}.png")
    plt.savefig(save_path1)
    plt.close(fig1)
    img_url1 = f"/static/extract_example/result_low_{file_id1}.png"
    # 고주파 전체 시각화
    fig2, axes2 = plt.subplots(len(wavelet_families), max_level_all, figsize=(5 * max_level_all, 5 * len(wavelet_families)))
    for fam_idx, wname in enumerate(wavelet_families):
        max_level = max_levels[fam_idx]
        for level in range(1, max_level+1):
            try:
                high_freq_img = strong_high_freq_wavelet(image_np, wavelet_name=wname, level=level)
                ax = axes2[fam_idx, level-1] if len(wavelet_families) > 1 else axes2[level-1]
                ax.set_title(f"{wname}\nlvl={level}")
                ax.imshow(np.clip(high_freq_img,0,255), cmap='gray')
                ax.axis('off')
            except Exception as e:
                ax = axes2[fam_idx, level-1] if len(wavelet_families) > 1 else axes2[level-1]
                ax.set_title(f"{wname}\nlvl={level}\nError")
                ax.axis('off')
        for level in range(max_level+1, max_level_all+1):
            ax = axes2[fam_idx, level-1] if len(wavelet_families) > 1 else axes2[level-1]
            ax.axis('off')
    plt.tight_layout()
    file_id2 = str(uuid.uuid4())
    save_path2 = os.path.join(save_dir, f"result_high_{file_id2}.png")
    plt.savefig(save_path2)
    plt.close(fig2)
    img_url2 = f"/static/extract_example/result_high_{file_id2}.png"
    # DCT 저주파/고주파 전체 시각화
    dct_keep_sizes = [8, 16, 32, 64, 128, 256]
    fig3, axes3 = plt.subplots(1, len(dct_keep_sizes), figsize=(5 * len(dct_keep_sizes), 5))
    dct_coeffs = dct2(image_np)
    for i, keep_size in enumerate(dct_keep_sizes):
        try:
            low_img = idct2(keep_low_freq_dct(dct_coeffs, keep_size=keep_size))
            ax = axes3[i] if len(dct_keep_sizes) > 1 else axes3
            ax.set_title(f"DCT 저주파 {keep_size}x{keep_size}")
            ax.imshow(np.clip(low_img,0,255), cmap='gray')
            ax.axis('off')
        except Exception as e:
            ax = axes3[i] if len(dct_keep_sizes) > 1 else axes3
            ax.set_title(f"DCT 저주파 {keep_size}x{keep_size}\nError")
            ax.axis('off')
    plt.tight_layout()
    file_id3 = str(uuid.uuid4())
    save_path3 = os.path.join(save_dir, f"result_dct_low_{file_id3}.png")
    plt.savefig(save_path3)
    plt.close(fig3)
    img_url3 = f"/static/extract_example/result_dct_low_{file_id3}.png"
    # DCT 고주파 전체 시각화
    fig4, axes4 = plt.subplots(1, len(dct_keep_sizes), figsize=(5 * len(dct_keep_sizes), 5))
    for i, keep_size in enumerate(dct_keep_sizes):
        try:
            high_img = idct2(keep_high_freq_dct(dct_coeffs, keep_size=keep_size))
            ax = axes4[i] if len(dct_keep_sizes) > 1 else axes4
            ax.set_title(f"DCT 고주파 제외 {keep_size}x{keep_size}")
            ax.imshow(np.clip(high_img,0,255), cmap='gray')
            ax.axis('off')
        except Exception as e:
            ax = axes4[i] if len(dct_keep_sizes) > 1 else axes4
            ax.set_title(f"DCT 고주파 제외 {keep_size}x{keep_size}\nError")
            ax.axis('off')
    plt.tight_layout()
    file_id4 = str(uuid.uuid4())
    save_path4 = os.path.join(save_dir, f"result_dct_high_{file_id4}.png")
    plt.savefig(save_path4)
    plt.close(fig4)
    img_url4 = f"/static/extract_example/result_dct_high_{file_id4}.png"
    result_imgs = [
        ("Wavelet 모든 level 저주파", img_url1),
        ("Wavelet 모든 level 고주파", img_url2),
        ("DCT 모든 level 저주파", img_url3),
        ("DCT 모든 level 고주파", img_url4)
    ]
    return templates.TemplateResponse("extract_example.html", {"request": request, "result_imgs": result_imgs})

# 예시: 특징 추출 시
# app = get_face_analysis_app()
