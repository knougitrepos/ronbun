from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os

from routes.extract import router as extract_router
from routes.recall import router as recall_router
from routes.similarity import router as similarity_router
from routes.datasets import router as datasets_router

app = FastAPI(
    title="특징추출 및 벡터 검색 시스템 API",
    description="이 API는 특징추출, 데이터셋 다운로드, 유사도/recall 테스트 등 주요 기능의 웹 라우트와 문서화를 제공합니다.",
    version="1.0.0",
)

# 정적 파일(css 등) 경로 추가
app.mount("/static", StaticFiles(directory="static"), name="static")

# 템플릿 디렉토리 설정
templates = Jinja2Templates(directory="templates")


@app.get(
    "/",
    response_class=HTMLResponse,
    description="메인 페이지. 각 기능별 이동 버튼 제공.",
)
async def index(request: Request):
    """메인 페이지: 각 기능별 이동 버튼 제공"""
    return templates.TemplateResponse("index.html", {"request": request})


# 라우트 모듈 등록
app.include_router(extract_router)
app.include_router(recall_router)
app.include_router(similarity_router)
app.include_router(datasets_router)
