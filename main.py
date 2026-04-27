"""
main.py v3 — FastAPI backend, quality=100, подробные логи
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

_usage: dict[str, int] = {}
MAX_PER_USER = 1000

pipeline: RetouchPipeline | None = None


def _check_limit(user_id: str) -> None:
    count = _usage.get(user_id, 0)
    if count >= MAX_PER_USER:
        raise HTTPException(status_code=429, detail="Limit reached.")
    _usage[user_id] = count + 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading RetouchPipeline…")
    pipeline = RetouchPipeline()
    pipeline.load_models()
    logger.info("RetouchPipeline ready.")
    yield
    logger.info("Shutdown.")


app = FastAPI(title="Retouch API v3", version="3.0.0", lifespan=lifespan)


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

    # ── Decode ─────────────────────────────────────────────────────────────
    try:
        img_bgr, meta = decode_image(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode: {exc}")

    h, w = img_bgr.shape[:2]

    logger.info("=" * 60)
    logger.info("IN  file     : %s", original_filename)
    logger.info("IN  format   : %s", meta.get("format", "?"))
    logger.info("IN  size     : %.2f MB", input_mb)
    logger.info("IN  resolution: %dx%d", w, h)
    logger.info("IN  user     : %s  (used %d/%d)", user_id, _usage.get(user_id, 0), MAX_PER_USER)

    # ── Process ────────────────────────────────────────────────────────────
    try:
        result_bgr, stats = pipeline.run(img_bgr)   # type: ignore
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}")

    rh, rw = result_bgr.shape[:2]

    # ── Encode quality=100 ─────────────────────────────────────────────────
    out_bytes = encode_image_to_bytes(result_bgr, meta, quality=100)
    output_mb = len(out_bytes) / 1024 / 1024

    logger.info("OUT resolution: %dx%d", rw, rh)
    logger.info("OUT size      : %.2f MB (in=%.2f MB)", output_mb, input_mb)
    logger.info("OUT quality   : 100, subsampling=0 (4:4:4)")
    logger.info("RETOUCH strength : %s", stats.get("retouch_strength"))
    logger.info("RETOUCH smooth   : %.2f", stats.get("smooth_strength", 0))
    logger.info("RETOUCH db       : %.2f", stats.get("db_strength", 0))
    logger.info("RETOUCH faces    : %d", stats.get("faces_found", 0))
    logger.info("RETOUCH codeformer: %s (fidelity=%.2f blend=%.2f)",
                stats.get("codeformer_applied"), stats.get("cf_fidelity"), stats.get("cf_blend"))
    logger.info("RETOUCH db_applied: %s", stats.get("db_applied"))
    logger.info("RETOUCH smooth_applied: %s", stats.get("smooth_applied"))
    logger.info("=" * 60)

    if rh != h or rw != w:
        logger.error("RESOLUTION CHANGED! %dx%d → %dx%d", w, h, rw, rh)

    stem = original_filename.rsplit(".", 1)[0]
    out_filename = f"retouched_{stem}.jpg"

    return StreamingResponse(
        io.BytesIO(out_bytes),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{out_filename}"',
            "X-Input-Resolution": f"{w}x{h}",
            "X-Output-Resolution": f"{rw}x{rh}",
            "X-Input-MB": f"{input_mb:.2f}",
            "X-Output-MB": f"{output_mb:.2f}",
            "X-Faces-Found": str(stats.get("faces_found", 0)),
            "X-Retouch-Strength": stats.get("retouch_strength", "?"),
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None,
        "retouch_strength": pipeline and "loaded",
    }


@app.get("/usage/{user_id}")
async def usage(user_id: str):
    return {"user_id": user_id, "used": _usage.get(user_id, 0), "limit": MAX_PER_USER}
