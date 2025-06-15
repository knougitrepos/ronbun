# 벡터 검색 recall 테스트 라우트 관리
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

router = APIRouter()

@router.get("/recall", response_class=HTMLResponse, description="recall 테스트 페이지. 벡터 검색 recall 성능 확인.")
async def recall(request: Request):
    """recall 테스트 페이지: 벡터 검색 recall 성능 확인"""
    return templates.TemplateResponse("recall.html", {"request": request})
