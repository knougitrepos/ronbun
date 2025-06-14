from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

router = APIRouter()

@router.get("/similarity", response_class=HTMLResponse)
async def similarity(request: Request):
    return templates.TemplateResponse("similarity.html", {"request": request})
