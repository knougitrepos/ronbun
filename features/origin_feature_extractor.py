# origin 특징추출 관련 함수 예시

# pip install insightface onnxruntime-gpu # GPU 사용 시
# pip install insightface onnxruntime # CPU 사용 시

import cv2
import insightface
import numpy as np
from typing import Optional

# ArcFace 모델 로드 (최초 1회만 실행)
try:
    app = insightface.app.FaceAnalysis(providers=None)  # providers=None이면 insightface가 자동으로 GPU/CPU 선택
    app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=0: 첫 번째 GPU, -1: CPU
except Exception as e:
    print(f"모델 로드 실패: {e}")
    app = None

def extract_arcface_feature(image_path: str) -> Optional[np.ndarray]:
    """
    주어진 이미지 경로에서 얼굴을 탐지하고 ArcFace 임베딩을 추출합니다.
    """
    if app is None:
        print("ArcFace 모델이 준비되지 않았습니다.")
        return None
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"이미지를 로드할 수 없습니다: {image_path}")
            return None
        faces = app.get(img)
        if not faces:
            print(f"얼굴을 탐지할 수 없습니다: {image_path}")
            return None
        # 첫 번째 얼굴의 임베딩 사용
        embedding = faces[0].embedding
        return embedding
    except Exception as e:
        print(f"특징 추출 중 오류 발생 ({image_path}): {e}")
        return None