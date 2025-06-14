from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

router = APIRouter()

@router.get("/datasets", response_class=HTMLResponse, description="데이터셋 다운로드 페이지. 데이터셋 선택 및 다운로드 기능 제공.")
async def datasets(request: Request):
    """데이터셋 다운로드 페이지: 데이터셋 선택 및 다운로드 기능 제공"""
    return templates.TemplateResponse("datasets.html", {"request": request})
