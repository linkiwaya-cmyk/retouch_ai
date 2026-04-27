"""
utils.py v9 — EXIF один раз, HEIC, quality=100
"""
from __future__ import annotations
import io, logging, subprocess, tempfile, time
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

_HEIC_OK = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIC_OK = True
    logger.info("HEIC: OK")
except Exception as e:
    logger.warning("HEIC: %s", e)


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB") if img.mode != "RGB" else img


def decode_image(raw: bytes) -> tuple[np.ndarray, dict]:
    """
    Декодирует → BGR uint8.
    EXIF ориентация применяется ОДИН РАЗ через ImageOps.exif_transpose().
    После этого img_bgr всегда в правильной ориентации.
    """
    t0 = time.time()
    meta = {"exif": None, "icc_profile": None, "original_size": None, "format": "JPEG",
            "orientation_applied": True}

    def finish(pil: Image.Image, fmt: str):
        meta["format"] = fmt
        meta["exif"]        = pil.info.get("exif")
        meta["icc_profile"] = pil.info.get("icc_profile")
        pil = ImageOps.exif_transpose(_to_rgb(pil))   # ОДИН РАЗ
        meta["original_size"] = pil.size
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        logger.info("Decoded %s %dx%d in %.2fs", fmt, pil.size[0], pil.size[1], time.time()-t0)
        return bgr, meta

    # 1. pillow_heif — primary image only
    if _HEIC_OK:
        try:
            import pillow_heif as ph
            hf = ph.open_heif(io.BytesIO(raw), convert_hdr_to_8bit=True)
            return finish(hf[0].to_pillow(), "HEIC")
        except Exception as e:
            logger.warning("open_heif: %s", e)

    # 2. PIL universal
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
        return finish(pil, pil.format or "JPEG")
    except Exception as e:
        logger.warning("PIL: %s", e)

    # 3. cv2 fallback
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        meta["original_size"] = (img.shape[1], img.shape[0])
        meta["orientation_applied"] = False
        logger.info("Decoded cv2 %dx%d", img.shape[1], img.shape[0])
        return img, meta

    # 4. pillow_heif read_heif
    if _HEIC_OK:
        try:
            import pillow_heif as ph
            h = ph.read_heif(raw)
            pil = Image.frombytes(h.mode, h.size, h.data, "raw").convert("RGB")
            return finish(pil, "HEIC")
        except Exception as e:
            logger.warning("read_heif: %s", e)

    # 5. ImageMagick
    try:
        with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as f:
            f.write(raw); src = f.name
        dst = src + "_out.jpg"
        r = subprocess.run(["convert", "-auto-orient", src, dst],
                           capture_output=True, timeout=30)
        if r.returncode == 0 and Path(dst).exists():
            img = cv2.imread(dst)
            Path(src).unlink(missing_ok=True)
            Path(dst).unlink(missing_ok=True)
            if img is not None:
                meta["original_size"] = (img.shape[1], img.shape[0])
                logger.info("Decoded ImageMagick %dx%d", img.shape[1], img.shape[0])
                return img, meta
    except Exception as e:
        logger.warning("ImageMagick: %s", e)

    raise ValueError(f"Cannot decode. HEIC={'OK' if _HEIC_OK else 'DISABLED'}.")


def encode_image_to_bytes(img: np.ndarray, meta: dict, quality=100) -> bytes:
    H, W = img.shape[:2]
    t0 = time.time()
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    kw: dict = {"format": "JPEG", "quality": quality,
                "subsampling": 0, "optimize": False, "progressive": False}
    if meta.get("icc_profile"):
        kw["icc_profile"] = meta["icc_profile"]
    if meta.get("exif"):
        kw["exif"] = _reset_orient(meta["exif"])
    buf = io.BytesIO()
    pil.save(buf, **kw)
    data = buf.getvalue()
    logger.info("Encoded %dx%d → %.2fMB in %.2fs", W, H, len(data)/1024/1024, time.time()-t0)
    return data


def _reset_orient(exif: bytes) -> bytes:
    try:
        import piexif
        d = piexif.load(exif)
        if "0th" in d: d["0th"][piexif.ImageIFD.Orientation] = 1
        return piexif.dump(d)
    except Exception:
        return exif


# masks
def feather(m: np.ndarray, r: int) -> np.ndarray:
    if r < 1: return m.astype(np.float32)
    k = r*2+1
    return np.clip(cv2.GaussianBlur(m.astype(np.float32),(k,k),r/2.),0,1)

def dilate(m: np.ndarray, k=5, n=1) -> np.ndarray:
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    return cv2.dilate((m*255).astype(np.uint8),ke,iterations=n).astype(np.float32)/255.

def erode(m: np.ndarray, k=3, n=1) -> np.ndarray:
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    return cv2.erode((m*255).astype(np.uint8),ke,iterations=n).astype(np.float32)/255.

def blend(src: np.ndarray, dst: np.ndarray, a: np.ndarray) -> np.ndarray:
    a = np.clip(a,0,1)[...,np.newaxis]
    return np.clip(src.astype(np.float32)*a + dst.astype(np.float32)*(1-a),0,255).astype(np.uint8)

def crop(img: np.ndarray, x1,y1,x2,y2) -> np.ndarray:
    H,W = img.shape[:2]
    return img[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
