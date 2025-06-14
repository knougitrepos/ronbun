from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

router = APIRouter()

@router.get("/extract", response_class=HTMLResponse, description="특징추출 페이지. 다양한 특징추출 모델 및 변환 옵션 제공.")
async def extract(request: Request):
    """특징추출 페이지: 다양한 특징추출 모델 및 변환 옵션 제공"""
    return templates.TemplateResponse("extract.html", {"request": request})
