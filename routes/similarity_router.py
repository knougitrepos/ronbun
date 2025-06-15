# 벡터 유사도 테스트 라우트 관리
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

router = APIRouter()

@router.get("/similarity", response_class=HTMLResponse, description="유사도 테스트 페이지. 벡터 간 유사도 측정 기능 제공.")
async def similarity(request: Request):
    """유사도 테스트 페이지: 벡터 간 유사도 측정 기능 제공"""
    return templates.TemplateResponse("similarity.html", {"request": request})
