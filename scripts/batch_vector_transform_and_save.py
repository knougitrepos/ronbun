import os
import cv2
import numpy as np
from core.config import ConfigLoader
from core.pipeline.vector_pipeline import VectorPipeline
from DB.db_utils import SessionLocal
from core.database import VectorRepository
from core.celery_app import celery_app


# 데이터셋 경로(예시)
DATASET_ROOT = "downloaded_datasets"

# 변환 파라미터 조합 예시
WAVELET_LEVELS = [1, 2, 3]
WAVELET_MODES = ['low', 'high']
DCT_KEEP_DIMS = [32, 64, 128, 256]
DCT_MODES = ['low', 'high']



config = ConfigLoader()
pipeline = VectorPipeline(config)

# DB 세션 및 저장소 준비
session = SessionLocal()
repository = VectorRepository(session)

# 이미지 경로 수집
def get_image_paths(root):
    image_paths = []
    for dirpath, _, files in os.walk(root):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(dirpath, file))
    return image_paths

def get_or_create_image_id(img_path, repository):
    # DB에 이미지가 이미 있으면 id 반환, 없으면 추가 후 id 반환
    image_obj = repository.get_image_by_path(img_path)
    if image_obj:
        return image_obj.id
    else:
        new_img = repository.add_image(img_path)
        return new_img.id

def load_image(img_path):
    # 실제 이미지 로딩 및 BGR 변환
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[경고] 이미지 로딩 실패: {img_path}")
            return None
        return img
    except Exception as e:
        print(f"[에러] 이미지 로딩 중 예외: {img_path}, {e}")
        return None

@celery_app.task
def process_image(img_path):
    session = SessionLocal()
    repository = VectorRepository(session)
    print(f"[처리중] {img_path}")
    image = load_image(img_path)
    if image is None:
        session.close()
        return
    image_id = get_or_create_image_id(img_path, repository)
    try:
        # 1. ArcFace 원본 벡터 추출 및 저장
        vec_origin = pipeline.run(image, transform_method=None)
        if vec_origin is not None:
            repository.add_embedding_512(
                image_id=image_id,
                vector_type='origin',
                parameters={},
                embedding=vec_origin,
                created_at=None
            )
        # 2. Wavelet 변환(level/mode별)
        for level in WAVELET_LEVELS:
            for mode in WAVELET_MODES:
                vec_wavelet = pipeline.run(image, transform_method='wavelet', transform_params={'level': level, 'mode': mode})
                if vec_wavelet is not None:
                    repository.add_embedding_256(
                        image_id=image_id,
                        vector_type='wavelet',
                        parameters={'level': level, 'mode': mode},
                        embedding=vec_wavelet,
                        created_at=None
                    )
        # 3. DCT 변환(keep_dim/mode별)
        for keep_dim in DCT_KEEP_DIMS:
            for mode in DCT_MODES:
                vec_dct = pipeline.run(image, transform_method='dct', transform_params={'keep_dim': keep_dim, 'mode': mode})
                if vec_dct is not None:
                    if keep_dim == 128:
                        repository.add_embedding_128(
                            image_id=image_id,
                            vector_type='dct',
                            parameters={'keep_dim': keep_dim, 'mode': mode},
                            embedding=vec_dct,
                            created_at=None
                        )
                    elif keep_dim == 256:
                        repository.add_embedding_256(
                            image_id=image_id,
                            vector_type='dct',
                            parameters={'keep_dim': keep_dim, 'mode': mode},
                            embedding=vec_dct,
                            created_at=None
                        )
        print(f"[완료] {img_path}")
    except Exception as e:
        print(f"[에러] {img_path}: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    image_paths = get_image_paths(DATASET_ROOT)
    session = SessionLocal()
    repository = VectorRepository(session)
    count = 0
    for img_path in image_paths:
        image_obj = repository.get_image_by_path(img_path)
        if image_obj is None:
            process_image.delay(img_path)
            count += 1
            continue
        image_id = image_obj.id
        # 1. ArcFace origin(원본) 중복 체크
        origin_exists = repository.get_embeddings_512(image_id=image_id, vector_type='origin', param_filter={})
        if not origin_exists:
            process_image.delay(img_path)
            count += 1
            continue
        # 2. Wavelet 변환(level/mode별) 중복 체크
        wavelet_all_exists = True
        for level in WAVELET_LEVELS:
            for mode in WAVELET_MODES:
                exists = repository.get_embeddings_256(
                    image_id=image_id,
                    vector_type='wavelet',
                    param_filter={'level': level, 'mode': mode}
                )
                if not exists:
                    wavelet_all_exists = False
        # 3. DCT 변환(keep_dim/mode별) 중복 체크
        dct_all_exists = True
        for keep_dim in DCT_KEEP_DIMS:
            for mode in DCT_MODES:
                if keep_dim == 128:
                    exists = repository.get_embeddings_128(
                        image_id=image_id,
                        vector_type='dct',
                        param_filter={'keep_dim': keep_dim, 'mode': mode}
                    )
                elif keep_dim == 256:
                    exists = repository.get_embeddings_256(
                        image_id=image_id,
                        vector_type='dct',
                        param_filter={'keep_dim': keep_dim, 'mode': mode}
                    )
                else:
                    exists = []
                if not exists:
                    dct_all_exists = False
        # 하나라도 미처리면 큐에 등록
        if not (wavelet_all_exists and dct_all_exists):
            process_image.delay(img_path)
            count += 1
    session.close()
    print(f"총 {count}개의 미처리 작업이 큐에 등록되었습니다.") 