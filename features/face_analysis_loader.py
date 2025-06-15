import insightface
from insightface.app import FaceAnalysis

face_app = None

def initialize_face_analysis():
    """
    애플리케이션 시작 시 호출될 함수.
    FaceAnalysis 모델을 초기화하여 전역 변수에 할당합니다.
    """
    global face_app
    if face_app is None:
        print("Initializing FaceAnalysis model...")
        face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("FaceAnalysis model initialized.")

def get_face_analysis_app():
    """
    애플리케이션 전역에서 사용될 모델 객체를 반환합니다.
    """
    if face_app is None:
        initialize_face_analysis()
    return face_app 