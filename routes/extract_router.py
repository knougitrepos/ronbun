import os

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from core.database import SessionLocal
from core.pipeline.transformers.pca import PCATransformer
from core.pipeline.transformers.pq import PQTransformer
from core.pipeline.vector_pipeline import VectorPipeline
from core.schemas import Embedding256, Embedding512, EmbeddingPQ, Image
from features.embedding_service import ArcFaceFeatureExtractor

templates = Jinja2Templates(directory="templates")
router = APIRouter()


@router.get(
    "/extract",
    response_class=HTMLResponse,
    description="ArcFace origin, PCA-256D, PQ thesis experiment extraction page.",
)
async def extract(request: Request):
    return templates.TemplateResponse("extract.html", {"request": request})


def collect_image_paths(dataset_root="downloaded_datasets"):
    image_paths = []
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(root, file))
    return image_paths


def extract_and_store_origin_features(image_paths):
    pipeline = VectorPipeline(extractor=ArcFaceFeatureExtractor())
    for image_path in image_paths:
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARN] Could not read image: {image_path}")
                continue
            embedding = pipeline.extract_features(image)
            pipeline.load_to_db(
                embedding,
                {"image_path": image_path},
                dim=512,
                vector_type="origin",
                parameters={},
            )
        except Exception as exc:
            print(f"[ERROR] Origin extraction failed: {image_path}, {exc}")


@router.post("/extract_features/origin")
async def extract_origin_features(background_tasks: BackgroundTasks):
    image_paths = collect_image_paths()
    background_tasks.add_task(extract_and_store_origin_features, image_paths)
    return RedirectResponse(url="/extract?queued=origin&count=%d" % len(image_paths), status_code=303)


@router.post("/extract_features/pca")
async def extract_pca_features():
    session = SessionLocal()
    origin_rows = session.query(Embedding512).filter(Embedding512.vector_type == "origin").all()
    if not origin_rows:
        session.close()
        return RedirectResponse(url="/extract?done=pca&count=0", status_code=303)

    vectors = np.stack([np.asarray(row.embedding, dtype=np.float32) for row in origin_rows])
    transformer = PCATransformer(n_components=256)
    transformed, log = transformer.fit_transform(vectors)
    transformer.save_codebook()

    pipeline = VectorPipeline()
    count = 0
    for row, vector in zip(origin_rows, transformed):
        image = session.query(Image).filter(Image.id == row.image_id).first()
        if image is None:
            continue
        pipeline.load_to_db(
            vector,
            {"image_path": image.image_path},
            dim=256,
            vector_type="pca",
            parameters={"n_components": 256},
            log=log,
        )
        count += 1

    session.close()
    return RedirectResponse(url=f"/extract?done=pca&count={count}", status_code=303)


@router.post("/extract_features/pq")
async def extract_pq_features():
    session = SessionLocal()
    pca_rows = session.query(Embedding256).filter(Embedding256.vector_type == "pca").all()
    if not pca_rows:
        session.close()
        return RedirectResponse(url="/extract?done=pq&count=0", status_code=303)

    vectors = np.stack([np.asarray(row.embedding, dtype=np.float32) for row in pca_rows])
    transformer = PQTransformer(d=256, M=16, nbits=8)
    codes, log = transformer.fit_transform(vectors)
    transformer.save_codebook()

    count = 0
    for row, code in zip(pca_rows, codes):
        session.add(
            EmbeddingPQ(
                image_id=row.image_id,
                vector_type="pq",
                parameters={"source": "pca", "n_components": 256, "M": 16, "nbits": 8},
                codes=bytes(code),
                log=log,
            )
        )
        count += 1
    session.commit()
    session.close()
    return RedirectResponse(url=f"/extract?done=pq&count={count}", status_code=303)


@router.get("/extract_features/check_pca")
async def check_pca_exists():
    session = SessionLocal()
    has_pca = session.query(Embedding256).filter(Embedding256.vector_type == "pca").first() is not None
    session.close()
    return JSONResponse(content={"has_pca": has_pca})
