from abc import ABC, abstractmethod
import numpy as np
from typing import Any
import cv2
from features.face_analysis_loader import get_face_analysis_app

# FeatureExtractor 추상 클래스
class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, image: Any) -> np.ndarray:
        pass

# ArcFaceFeatureExtractor 구현체
class ArcFaceFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.app = get_face_analysis_app()

    def extract(self, image: Any) -> np.ndarray:
        # image: np.ndarray (BGR)
        faces = self.app.get(image)
        if not faces:
            raise ValueError("No face detected in the image")
        return faces[0].embedding

# (예시) 추후 EfficientNet 등 다른 모델도 FeatureExtractor 상속 구현 가능

# 서비스 계층 함수: 이미지 바이트에서 임베딩 추출
# (기본 ArcFace 사용, 추후 파라미터로 extractor 선택 가능)
def extract_embedding_from_image(image_bytes, extractor: FeatureExtractor = None):
    """
    이미지 바이트를 입력받아 얼굴 임베딩 벡터를 추출하는 서비스 함수.
    """
    if extractor is None:
        extractor = ArcFaceFeatureExtractor()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return extractor.extract(img) 