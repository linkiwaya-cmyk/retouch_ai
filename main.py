"""main.py v8"""
import io, logging, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pipeline import RetouchPipeline
from utils import decode_image, encode_image_to_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_usage: dict[str, int] = {}
MAX_PER_USER = 1000
pipeline: RetouchPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = RetouchPipeline()
    pipeline.load_models()
    logger.info("Pipeline ready.")
    yield


app = FastAPI(title="Retouch API v8", lifespan=lifespan)


@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    user_id: str = Form(default="anonymous"),
):
    c = _usage.get(user_id, 0)
    if c >= MAX_PER_USER:
        raise HTTPException(429, "Limit reached.")
    _usage[user_id] = c + 1

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file.")

    fname  = file.filename or "photo.jpg"
    in_mb  = len(raw) / 1024 / 1024
    t_all  = time.time()

    t0 = time.time()
    try:
        img_bgr, meta = decode_image(raw)
    except Exception as e:
        raise HTTPException(400, f"Cannot decode: {e}")
    t_decode = time.time() - t0

    H, W = img_bgr.shape[:2]
    logger.info("=" * 55)
    logger.info("IN  file       : %s", fname)
    logger.info("IN  format     : %s", meta.get("format","?"))
    logger.info("IN  size       : %.2f MB", in_mb)
    logger.info("IN  resolution : %dx%d", W, H)
    logger.info("IN  orientation: applied=%s", meta.get("orientation_applied"))
    logger.info("IN  decode     : %.2fs", t_decode)

    try:
        result, stats = pipeline.run(img_bgr)  # type: ignore
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(500, f"Error: {e}")

    rH, rW = result.shape[:2]

    t0 = time.time()
    out_bytes = encode_image_to_bytes(result, meta, quality=100)
    t_enc = time.time() - t0
    out_mb = len(out_bytes) / 1024 / 1024
    t_total = time.time() - t_all

    logger.info("OUT resolution : %dx%d%s", rW, rH,
                " ⚠️ CHANGED" if (rW!=W or rH!=H) else " ✓")
    logger.info("OUT size       : %.2f MB (in=%.2f MB)", out_mb, in_mb)
    logger.info("TIME decode    : %.2fs", t_decode)
    logger.info("TIME detect    : %.2fs", stats.get("t_detect",0))
    logger.info("TIME retouch   : %.2fs", stats.get("t_retouch",0))
    logger.info("TIME encode    : %.2fs", t_enc)
    logger.info("TIME total     : %.2fs ← must be <60s", t_total)
    logger.info("RETOUCH faces  : %d", stats.get("faces",0))
    logger.info("RETOUCH redness: %.2f  level: %.2f  micro: %.2f",
                stats.get("redness",0), stats.get("level",0), MICRO_STR if hasattr(pipeline,'_') else 0)
    logger.info("=" * 55)

    stem = fname.rsplit(".", 1)[0]
    out_name = f"retouched_{stem}.jpg"

    return StreamingResponse(
        io.BytesIO(out_bytes),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{out_name}"',
            "X-Total-Time": f"{t_total:.1f}s",
            "X-Input-MB":   f"{in_mb:.2f}",
            "X-Output-MB":  f"{out_mb:.2f}",
            "X-Faces":      str(stats.get("faces",0)),
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "ready": pipeline is not None}


@app.get("/usage/{uid}")
async def usage(uid: str):
    return {"used": _usage.get(uid, 0), "limit": MAX_PER_USER}


# Экспортируем для логов
from pipeline import MICRO_STR  # noqa
