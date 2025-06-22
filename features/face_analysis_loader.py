# -------------------------------------------------------------
# features/face_analysis_loader.py
# -------------------------------------------------------------
# ArcFace 기반 얼굴 임베딩/분석 모델을 애플리케이션 전체에서
# 단 한 번만 메모리에 로드하여 효율적으로 재사용할 수 있게 하는
# '모델 로더' 역할의 유틸리티 파일입니다.
# - 무거운 딥러닝 모델(ArcFace 등)을 매 요청마다 새로 로드하지 않고,
#   앱 시작 시 한 번만 로드하여 전역 변수(face_app)로 관리합니다.
# - FastAPI, Flask 등 웹 서비스 환경에서 성능과 자원 효율성을 극대화합니다.
# - 서비스 계층, 라우트 등에서 get_face_analysis_app()을 통해
#   ArcFace 모델 객체를 안전하게 가져와 사용할 수 있습니다.
# -------------------------------------------------------------

import insightface
from insightface.app import FaceAnalysis

# 전역 ArcFace 모델 객체 (앱 전체에서 공유)
face_app = None

def initialize_face_analysis():
    """
    애플리케이션 시작 시 호출될 함수.
    ArcFace 기반 FaceAnalysis 모델을 전역 변수에 한 번만 로드합니다.
    - name='buffalo_l': ArcFace 기반 사전학습 모델 지정
    - providers=['CPUExecutionProvider']: CPU에서 실행 (GPU 사용 시 변경)
    - det_size=(640, 640): 얼굴 검출 입력 이미지 크기
    """


    #  640x640은 얼굴을 찾기 위한 크기이고, 실제 ArcFace 임베딩 추출은 검출된 얼굴을 112x112로 크롭한 후에 수행됩니다.
    global face_app
    if face_app is None:
        print("Initializing FaceAnalysis model...")
        face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("FaceAnalysis model initialized.")

def get_face_analysis_app():
    """
    전역에서 ArcFace 모델 객체를 안전하게 반환하는 함수.
    - 이미 로드되어 있으면 그대로 반환
    - 아직 로드되지 않았다면 initialize_face_analysis()로 로드 후 반환
    - 서비스 계층, 라우트 등에서 얼굴 임베딩 추출, 검출 등에 활용
    """
    if face_app is None:
        initialize_face_analysis()
    return face_app 