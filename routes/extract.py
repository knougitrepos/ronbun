from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

router = APIRouter()

@router.get("/extract", response_class=HTMLResponse)
async def extract(request: Request):
    return templates.TemplateResponse("extract.html", {"request": request})
