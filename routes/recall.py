from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

router = APIRouter()

@router.get("/recall", response_class=HTMLResponse)
async def recall(request: Request):
    return templates.TemplateResponse("recall.html", {"request": request})
