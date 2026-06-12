# FastAPI 프로젝트 진입점: 앱, 라우트, 템플릿, 정적파일 관리
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os
import subprocess
from sqlalchemy import func
from core.schemas import *  # 모든 테이블 강제 import
from DB.db_utils import SessionLocal
from core.database import VectorRepository
import socket
import sys

from routes.extract_router import router as extract_router
from routes.recall_router import router as recall_router
from routes.similarity_router import router as similarity_router
from routes.datasets_router import router as datasets_router
from DB.db_utils import reset_db
from features.face_analysis_loader import initialize_face_analysis

app = FastAPI(
    title="특징추출 및 벡터 검색 시스템 API",
    description="이 API는 특징추출, 데이터셋 다운로드, 유사도/recall 테스트 등 주요 기능의 웹 라우트와 문서화를 제공합니다.",
    version="1.0.0",
)

# 정적 파일(css 등) 경로 추가
app.mount("/static", StaticFiles(directory="static"), name="static")

# 템플릿 디렉토리 설정
templates = Jinja2Templates(directory="templates")

# 앱 시작 시 모델 초기화
initialize_face_analysis()

@app.get(
    "/",
    response_class=HTMLResponse,
    description="메인 페이지. 각 기능별 이동 버튼 제공.",
)
async def index(request: Request):
    """메인 페이지: 각 기능별 이동 버튼 제공"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/reset-db", description="DB 전체 초기화(모든 데이터 삭제 및 테이블 재생성)")
async def reset_db_route():
    reset_db()
    return {"message": "DB가 초기화되었습니다."}


# 라우트 모듈 등록
app.include_router(extract_router)
app.include_router(recall_router)
app.include_router(similarity_router)
app.include_router(datasets_router)

@app.get("/automation", response_class=HTMLResponse, description="자동화 페이지. Celery/MLflow 등 실험 자동화 관리.")
async def automation_page(request: Request):
    return templates.TemplateResponse("automation.html", {"request": request})

@app.post("/automation/run-batch-transform")
async def automation_run_batch_transform():
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        subprocess.Popen(
            [sys.executable, "scripts/batch_vector_transform_and_save.py"],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {"message": "벡터 변환 작업 실행을 시작했습니다."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/automation/mlflow/start")
async def automation_mlflow_start():
    try:
        # 이미 실행 중인지 체크(포트 5001)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 5001))
        sock.close()
        if result == 0:
            return {"message": "MLflow UI가 이미 실행 중입니다."}
        project_root = os.path.dirname(os.path.abspath(__file__))
        subprocess.Popen(
            [sys.executable, "-m", "mlflow", "ui", "--port", "5001", "--backend-store-uri", "./mlruns", "--host", "127.0.0.1"],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {"message": "MLflow UI 실행을 시작했습니다."}
    except Exception as e:
        return {"error": str(e)}

def is_port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except Exception:
        return False

@app.get("/automation/status", response_class=HTMLResponse, description="Celery/MLflow 상태 모니터링.")
async def automation_status(request: Request):
    # Celery 상태 확인
    try:
        result = subprocess.run(
            ["celery", "-A", "core.celery_app.celery_app", "status"],
            capture_output=True, text=True, timeout=3
        )
        celery_status = result.stdout.strip() or "Celery 워커 미실행"
    except Exception as e:
        celery_status = f"확인 실패: {e}"

    # MLflow 상태 확인
    mlflow_running = is_port_open("localhost", 5001)
    mlflow_url = "http://localhost:5001"

    # 최근 작업 진행률 계산
    session = SessionLocal()
    repo = VectorRepository(session)
    total_images = session.query(func.count(Image.id)).scalar() or 1
    progress_list = []
    pca_done = len(repo.get_embeddings_256(vector_type='pca', param_filter={'n_components': 256}))
    progress_list.append(f"PCA(256): {pca_done}/{total_images} ({pca_done/total_images*100:.1f}%)")
    pq_done = session.query(func.count(EmbeddingPQ.id)).scalar() or 0
    progress_list.append(f"PQ: {pq_done}/{total_images} ({pq_done/total_images*100:.1f}%)")
    session.close()

    return templates.TemplateResponse(
        "automation_status.html",
        {
            "request": request,
            "status": {
                "celery": celery_status,
                "mlflow_running": mlflow_running,
                "mlflow_url": mlflow_url,
                "progress_list": progress_list,
            }
        }
    )
