"""
main.py v4 — FastAPI, quality=100, подробные логи
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


def _check_limit(uid: str):
    c = _usage.get(uid, 0)
    if c >= MAX_PER_USER:
        raise HTTPException(429, "Limit reached.")
    _usage[uid] = c + 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = RetouchPipeline()
    pipeline.load_models()
    logger.info("Pipeline ready.")
    yield


app = FastAPI(title="Retouch API v4", lifespan=lifespan)


@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    user_id: str = Form(default="anonymous"),
):
    _check_limit(user_id)
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file.")

    fname = file.filename or "photo.jpg"
    in_mb = len(raw) / 1024 / 1024

    try:
        img_bgr, meta = decode_image(raw)
    except Exception as e:
        raise HTTPException(400, f"Cannot decode image: {e}")

    H, W = img_bgr.shape[:2]

    logger.info("=" * 55)
    logger.info("IN  file      : %s", fname)
    logger.info("IN  format    : %s", meta.get("format", "?"))
    logger.info("IN  size      : %.2f MB", in_mb)
    logger.info("IN  resolution: %dx%d", W, H)
    logger.info("IN  user      : %s (%d/%d)", user_id, _usage.get(user_id,0), MAX_PER_USER)

    try:
        result, stats = pipeline.run(img_bgr)   # type: ignore
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(500, f"Error: {e}")

    rH, rW = result.shape[:2]
    out_bytes = encode_image_to_bytes(result, meta, quality=100)
    out_mb = len(out_bytes) / 1024 / 1024

    logger.info("OUT resolution: %dx%d", rW, rH)
    logger.info("OUT size      : %.2f MB (was %.2f MB)", out_mb, in_mb)
    logger.info("OUT quality   : 100 subsampling=0")
    logger.info("RETOUCH faces : %d", stats.get("faces",0))
    logger.info("RETOUCH strength   : %s", stats.get("strength"))
    logger.info("RETOUCH smooth     : %.2f", stats.get("smooth",0))
    logger.info("RETOUCH db         : %.2f", stats.get("db",0))
    logger.info("RETOUCH tone       : %.2f", stats.get("tone",0))
    logger.info("RETOUCH codeformer : applied=%s fidelity=%.2f blend=%.2f",
                stats.get("cf_applied"), stats.get("cf_fidelity",0), stats.get("cf_blend",0))
    logger.info("=" * 55)

    if rH != H or rW != W:
        logger.error("RESOLUTION CHANGED: %dx%d → %dx%d", W, H, rW, rH)

    stem = fname.rsplit(".", 1)[0]
    out_name = f"retouched_{stem}.jpg"

    return StreamingResponse(
        io.BytesIO(out_bytes),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{out_name}"',
            "X-Input-MB": f"{in_mb:.2f}",
            "X-Output-MB": f"{out_mb:.2f}",
            "X-Resolution": f"{rW}x{rH}",
            "X-Faces": str(stats.get("faces", 0)),
            "X-Retouch": stats.get("strength", "?"),
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "ready": pipeline is not None}


@app.get("/usage/{uid}")
async def usage(uid: str):
    return {"used": _usage.get(uid, 0), "limit": MAX_PER_USER}
