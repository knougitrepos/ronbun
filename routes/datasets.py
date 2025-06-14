from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

router = APIRouter()

@router.get("/datasets", response_class=HTMLResponse)
async def datasets(request: Request):
    return templates.TemplateResponse("datasets.html", {"request": request})
