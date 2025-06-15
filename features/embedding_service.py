from features.face_analysis_loader import get_face_analysis_app
import cv2
import numpy as np

def extract_embedding_from_image(image_bytes):
    """
    이미지 바이트를 입력받아 얼굴 임베딩 벡터를 추출하는 서비스 함수.
    """
    app = get_face_analysis_app()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    faces = app.get(img)
    if not faces:
        raise ValueError("No face detected in the image")
    # 첫 번째로 탐지된 얼굴의 임베딩을 사용
    embedding = faces[0].embedding
    return embedding 