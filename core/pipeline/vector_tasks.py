from core.celery_app import celery_app
from core.pipeline.vector_pipeline import VectorPipeline
import numpy as np
import cv2
import os
import sys

def _get_pipeline():
    return VectorPipeline()

def load_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[경고] 이미지 로딩 실패: {img_path}")
            return None
        return img
    except Exception as e:
        print(f"[에러] 이미지 로딩 중 예외: {img_path}, {e}")
        return None

@celery_app.task(name='core.pipeline.vector_tasks.process_image')
def process_image(image_path):
    print(f"[PYTHON] {sys.executable}")
    print(f"[ENV] {os.environ.get('CONDA_DEFAULT_ENV')}")
    print(f"[CWD] {os.getcwd()}")
    print(f"[태스크 시작] {image_path}")
    try:
        pipeline = _get_pipeline()
        image = load_image(image_path)
        if image is None:
            return f"이미지 로딩 실패: {image_path}"
        result = pipeline.run(image, meta={'image_path': image_path})
        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return [serialize(o) for o in obj]
            elif isinstance(obj, list):
                return [serialize(o) for o in obj]
            elif obj is None:
                return "None"
            else:
                return obj
        result = serialize(result)
        print(f"[DEBUG] 반환값: {result}, 타입: {type(result)}")
        return result
    except Exception as e:
        print(f"[에러] {image_path}: {e}")
        return str(e) 