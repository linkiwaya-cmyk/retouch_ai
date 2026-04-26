"""
Photo Retouching Backend — FastAPI entry point.
POST /process-image  →  runs full pipeline, returns processed image.
"""

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
import cv2

from pipeline import RetouchPipeline
from utils import decode_image, encode_image_to_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── per-user usage counter (in-memory) ────────────────────────────────────────
_usage: dict[str, int] = {}
MAX_PER_USER = 1000


def _check_limit(user_id: str) -> None:
    count = _usage.get(user_id, 0)
    if count >= MAX_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=f"Limit reached: {MAX_PER_USER} images per user.",
        )
    _usage[user_id] = count + 1


# ── app lifecycle ──────────────────────────────────────────────────────────────
pipeline: RetouchPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading RetouchPipeline …")
    pipeline = RetouchPipeline()
    pipeline.load_models()
    logger.info("RetouchPipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Photo Retouch API", version="1.0.0", lifespan=lifespan)


# ── endpoint ───────────────────────────────────────────────────────────────────
@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    user_id: str = Form(default="anonymous"),
):
    """
    Accept an image file, run the retouching pipeline, return the result.
    """
    _check_limit(user_id)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        img_bgr = decode_image(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    try:
        result_bgr = pipeline.run(img_bgr)  # type: ignore[union-attr]
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}")

    out_bytes = encode_image_to_bytes(result_bgr, ext=".jpg", quality=95)
    return StreamingResponse(
        io.BytesIO(out_bytes),
        media_type="image/jpeg",
        headers={"Content-Disposition": 'attachment; filename="retouched.jpg"'},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None}


@app.get("/usage/{user_id}")
async def usage(user_id: str):
    return {"user_id": user_id, "used": _usage.get(user_id, 0), "limit": MAX_PER_USER}
