"""
main.py — FastAPI backend для AI ретуши.

POST /process-image:
  - принимает файл любого формата (JPEG, PNG, HEIC)
  - возвращает JPEG quality=98, оригинальное разрешение
  - логирует размеры входа/выхода
"""

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from pipeline import RetouchPipeline
from utils import decode_image, encode_image_to_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Usage limits (in-memory) ───────────────────────────────────────────────────
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


# ── Lifecycle ──────────────────────────────────────────────────────────────────
pipeline: RetouchPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading RetouchPipeline…")
    pipeline = RetouchPipeline()
    pipeline.load_models()
    logger.info("RetouchPipeline ready.")
    yield
    logger.info("Shutdown.")


app = FastAPI(title="Photo Retouch API", version="2.0.0", lifespan=lifespan)


# ── Endpoint ───────────────────────────────────────────────────────────────────

@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    user_id: str = Form(default="anonymous"),
):
    _check_limit(user_id)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    original_filename = file.filename or "photo.jpg"
    input_mb = len(raw) / 1024 / 1024
    logger.info(
        "IN  file=%s user=%s size=%.2f MB",
        original_filename, user_id, input_mb,
    )

    # Decode (EXIF ориентация применяется автоматически)
    try:
        img_bgr, meta = decode_image(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    h, w = img_bgr.shape[:2]
    logger.info("IN  resolution: %dx%d", w, h)

    # Process
    try:
        result_bgr = pipeline.run(img_bgr)  # type: ignore
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}")

    rh, rw = result_bgr.shape[:2]
    if rh != h or rw != w:
        logger.error("Resolution changed! %dx%d → %dx%d", w, h, rw, rh)

    # Encode — quality=98, сохраняем EXIF и ICC
    out_bytes = encode_image_to_bytes(result_bgr, meta, quality=98)
    output_mb = len(out_bytes) / 1024 / 1024
    logger.info(
        "OUT resolution: %dx%d size=%.2f MB (input was %.2f MB)",
        rw, rh, output_mb, input_mb,
    )

    # Имя выходного файла: retouched_<original>
    stem = original_filename.rsplit(".", 1)[0]
    out_filename = f"retouched_{stem}.jpg"

    return StreamingResponse(
        io.BytesIO(out_bytes),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{out_filename}"',
            "X-Input-Resolution": f"{w}x{h}",
            "X-Output-Resolution": f"{rw}x{rh}",
            "X-Input-Size-MB": f"{input_mb:.2f}",
            "X-Output-Size-MB": f"{output_mb:.2f}",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None}


@app.get("/usage/{user_id}")
async def usage(user_id: str):
    return {
        "user_id": user_id,
        "used": _usage.get(user_id, 0),
        "limit": MAX_PER_USER,
    }
