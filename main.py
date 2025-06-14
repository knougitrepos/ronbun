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

app = FastAPI()

# 템플릿 디렉토리 설정
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 라우트 모듈 등록
app.include_router(extract_router)
app.include_router(recall_router)
app.include_router(similarity_router)
app.include_router(datasets_router)
