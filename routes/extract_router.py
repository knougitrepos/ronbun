# 특징 추출, 변환, 예시 등 주요 기능 라우트 관리
from fastapi import APIRouter, Request, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import os
from DB.db_utils import SessionLocal, ImageEmbeddings, add_image_embedding
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
from features.embedding_service import ArcFaceFeatureExtractor, NoneFaceDetectFeatureExtractor
from features.vector_transformer import transform_vector
import datetime
import cv2
from core.pipeline.vector_pipeline import VectorPipeline
from core.schemas import Image

templates = Jinja2Templates(directory="templates")

router = APIRouter()

@router.get("/extract", response_class=HTMLResponse, description="특징추출 페이지. 다양한 특징추출 모델 및 변환 옵션 제공.")
async def extract(request: Request):
    """특징추출 페이지: 다양한 특징추출 모델 및 변환 옵션 제공"""
    return templates.TemplateResponse("extract.html", {"request": request})

# 대량 특징추출 서비스 함수
def extract_and_store_features(image_paths):
    vp = VectorPipeline(extractor=ArcFaceFeatureExtractor())
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            feat = vp.extract_features(img)
            meta = {'image_path': img_path}
            vp.load_to_db(feat, meta, dim=512, vector_type='origin', parameters={})
        except Exception as e:
            print(f"특징 추출 에러: {img_path}, {e}")

@router.post("/extract_features/origin")
async def extract_origin_features(request: Request, background_tasks: BackgroundTasks):
    dataset_root = "downloaded_datasets"
    image_paths = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(root, file))
    background_tasks.add_task(extract_and_store_features, image_paths)
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

@router.post("/extract_features/wavelet")
async def extract_wavelet_features(request: Request):
    from core.database import SessionLocal
    from core.schemas import Embedding128, Embedding256, Embedding512, Image
    session = SessionLocal()
    pca_vecs = []
    for Emb in [Embedding128, Embedding256, Embedding512]:
        pca_vecs.extend(session.query(Emb).filter(Emb.vector_type == 'pca').all())
    session.close()
    if not pca_vecs:
        return RedirectResponse(url="/extract?done=wavelet&count=0", status_code=303)
    wavelet_names = ['haar', 'db2', 'sym2']
    levels = [1, 2, 3, 4]
    modes = ['low', 'high']
    vp = VectorPipeline()
    count = 0
    for pca_vec in pca_vecs:
        vec = pca_vec.embedding
        dim = vec.shape[0]
        # image_id로부터 image_path 조회
        image_obj = session.query(Image).filter(Image.id == pca_vec.image_id).first()
        image_path = image_obj.image_path if image_obj else None
        meta = {'image_path': image_path}
        for wname in wavelet_names:
            for level in levels:
                for mode in modes:
                    vec_trans, log = vp.transform_vector(vec, method='wavelet', wavelet_name=wname, level=level, mode=mode)
                    params = {'wavelet_name': wname, 'level': level, 'mode': mode}
                    vp.load_to_db(vec_trans, meta, dim=dim, vector_type='wavelet', parameters=params, log=log)
                    count += 1
    return RedirectResponse(url=f"/extract?done=wavelet&count={count}", status_code=303)

@router.post("/extract_features/dct")
async def extract_dct_features(request: Request):
    from core.database import SessionLocal
    from core.schemas import Embedding128, Embedding256, Embedding512, Image
    session = SessionLocal()
    pca_vecs = []
    for Emb in [Embedding128, Embedding256, Embedding512]:
        pca_vecs.extend(session.query(Emb).filter(Emb.vector_type == 'pca').all())
    session.close()
    if not pca_vecs:
        return RedirectResponse(url="/extract?done=dct&count=0", status_code=303)
    keep_dims = [16, 32, 64, 128, 256]
    modes = ['low', 'high']
    vp = VectorPipeline()
    count = 0
    for pca_vec in pca_vecs:
        vec = pca_vec.embedding
        dim = vec.shape[0]
        # image_id로부터 image_path 조회
        image_obj = session.query(Image).filter(Image.id == pca_vec.image_id).first()
        image_path = image_obj.image_path if image_obj else None
        meta = {'image_path': image_path}
        for keep_dim in keep_dims:
            for mode in modes:
                vec_trans, log = vp.transform_vector(vec, method='dct', keep_dim=keep_dim, mode=mode)
                params = {'keep_dim': keep_dim, 'mode': mode}
                vp.load_to_db(vec_trans, meta, dim=dim, vector_type='dct', parameters=params, log=log)
                count += 1
    return RedirectResponse(url=f"/extract?done=dct&count={count}", status_code=303)

@router.post("/extract_features/pca")
async def extract_pca_features(request: Request):
    from core.database import SessionLocal
    from core.schemas import Embedding512
    import numpy as np
    session = SessionLocal()
    # 원본 특징벡터만 사용
    origin_vecs = session.query(Embedding512).filter(Embedding512.vector_type == 'origin').all()
    if not origin_vecs:
        session.close()
        return RedirectResponse(url="/extract?done=pca&count=0", status_code=303)
    n_components_list = [128, 256, 512]
    feats = [vec.embedding for vec in origin_vecs]
    feats_np = np.stack(feats)
    count = 0
    from core.pipeline.vector_pipeline import VectorPipeline
    vp = VectorPipeline()
    for n_components in n_components_list:
        vecs_trans, log = vp.transform_vector(feats_np, method='pca', n_components=n_components)
        print(log)
        for i, _ in enumerate(origin_vecs):
            image_obj = session.query(Image).filter(Image.id == origin_vecs[i].image_id).first()
            image_path = image_obj.image_path if image_obj else None
            meta = {'image_path': image_path}
            params = {'n_components': n_components}
            vp.load_to_db(vecs_trans[i], meta, dim=vecs_trans[i].shape[0], vector_type='pca', parameters=params, log=log)
            count += 1
    session.close()
    return RedirectResponse(url=f"/extract?done=pca&count={count}", status_code=303)

@router.post("/extract_features/pq")
async def extract_pq_features(request: Request):
    from core.database import SessionLocal
    from core.schemas import Embedding128, Embedding256, Embedding512
    session = SessionLocal()
    pca_vecs = []
    for Emb in [Embedding128, Embedding256, Embedding512]:
        pca_vecs.extend(session.query(Emb).filter(Emb.vector_type == 'pca').all())
    session.close()
    if not pca_vecs:
        return RedirectResponse(url="/extract?done=pq&count=0", status_code=303)
    M_list = [8, 16, 32]
    nbits_list = [4, 8]
    vp = VectorPipeline()
    count = 0
    for pca_vec in pca_vecs:
        vec = pca_vec.embedding
        dim = vec.shape[0]
        meta = {'image_path': None}
        for M in M_list:
            for nbits in nbits_list:
                vec_trans = vp.transform_vector(vec, method='pq', M=M, nbits=nbits)
                params = {'M': M, 'nbits': nbits}
                vp.load_to_db(vec_trans, meta, dim=dim, vector_type='pq', parameters=params)
                count += 1
    return RedirectResponse(url=f"/extract?done=pq&count={count}", status_code=303)

@router.get("/extract_features/check_pca")
async def check_pca_exists():
    from core.database import SessionLocal
    session = SessionLocal()
    from core.schemas import Embedding256
    has_pca = session.query(Embedding256).filter(Embedding256.vector_type == 'pca').first() is not None
    session.close()
    return JSONResponse(content={"has_pca": has_pca})

# 예시: 특징 추출 시
# app = get_face_analysis_app()
