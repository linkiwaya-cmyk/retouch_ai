"""
utils.py v6

HEIC fix: "Too many auxiliary image references" — берём ТОЛЬКО primary image
через pillow_heif напрямую, игнорируем depth/live-photo auxiliary layers.
"""
from __future__ import annotations

import io
import logging
import time
import numpy as np
import cv2
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# ── HEIC регистрация на уровне модуля ─────────────────────────────────────────
_HEIC_OK = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIC_OK = True
    logger.info("HEIC/HEIF: enabled via pillow-heif")
except ImportError:
    logger.warning("HEIC disabled — pip install pillow-heif")
except Exception as e:
    logger.warning("HEIC register failed: %s", e)


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """PIL RGB/RGBA → BGR uint8"""
    if pil_img.mode == "RGBA":
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def decode_image(raw_bytes: bytes) -> tuple[np.ndarray, dict]:
    """
    Декодирует JPEG / PNG / HEIC / HEIF / WebP → BGR uint8.
    HEIC: берёт только primary image, игнорирует auxiliary (depth/live-photo).
    """
    t0 = time.time()
    meta = {"exif": None, "icc_profile": None, "original_size": None, "format": "JPEG"}

    # ── Попытка 1: HEIC через pillow_heif напрямую (primary image only) ───
    # Это решает "Too many auxiliary image references"
    if _HEIC_OK:
        try:
            import pillow_heif
            # read_heif берёт primary image и игнорирует auxiliary layers
            heif_file = pillow_heif.open_heif(raw_bytes, convert_hdr_to_8bit=True)
            # Берём только первый (primary) image
            primary = heif_file[0]
            pil_img = Image.frombytes(
                primary.mode,
                primary.size,
                primary.data,
                "raw",
                primary.mode,
            )
            # EXIF ориентация
            if hasattr(primary, "info") and primary.info.get("exif"):
                meta["exif"] = primary.info["exif"]
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            meta["format"] = "HEIC"
            meta["original_size"] = pil_img.size
            img_bgr = _pil_to_bgr(pil_img)
            logger.info("Decoded HEIC (primary): %dx%d in %.2fs", *pil_img.size, time.time()-t0)
            return img_bgr, meta
        except Exception as e:
            # Если HEIC не сработал — пробуем дальше
            if "heic" in str(type(e).__name__).lower() or "heif" in str(e).lower() or "auxiliary" in str(e).lower():
                logger.warning("HEIC primary decode failed: %s — trying PIL fallback", e)
            # иначе просто продолжаем

    # ── Попытка 2: PIL универсальный (JPEG, PNG, WebP, HEIC если зарегистрирован) ──
    try:
        buf = io.BytesIO(raw_bytes)
        pil_img = Image.open(buf)
        pil_img.load()
        meta["format"] = pil_img.format or "JPEG"
        meta["exif"] = pil_img.info.get("exif")
        meta["icc_profile"] = pil_img.info.get("icc_profile")
        pil_img = ImageOps.exif_transpose(pil_img)
        meta["original_size"] = pil_img.size
        img_bgr = _pil_to_bgr(pil_img)
        logger.info("Decoded %s: %dx%d in %.2fs", meta["format"], *pil_img.size, time.time()-t0)
        return img_bgr, meta
    except Exception as e:
        logger.warning("PIL decode failed: %s", e)

    # ── Попытка 3: cv2 (JPEG/PNG без EXIF) ────────────────────────────────
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        meta["original_size"] = (img.shape[1], img.shape[0])
        logger.info("Decoded via cv2: %dx%d in %.2fs", img.shape[1], img.shape[0], time.time()-t0)
        return img, meta

    # ── Попытка 4: HEIC через pillow_heif.read_heif (старый API) ──────────
    if _HEIC_OK:
        try:
            import pillow_heif
            heif = pillow_heif.read_heif(raw_bytes)
            pil_img = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            meta["format"] = "HEIC"
            meta["original_size"] = pil_img.size
            img_bgr = _pil_to_bgr(pil_img)
            logger.info("Decoded HEIC (read_heif): %dx%d", *pil_img.size)
            return img_bgr, meta
        except Exception as e2:
            logger.warning("HEIC read_heif failed: %s", e2)

    raise ValueError(
        f"Cannot decode image. "
        f"Supported: JPEG, PNG, HEIC, HEIF, WebP. "
        f"HEIC support: {'enabled' if _HEIC_OK else 'DISABLED'}."
    )


def encode_image_to_bytes(img_bgr: np.ndarray, meta: dict, quality: int = 100) -> bytes:
    """JPEG quality=100, subsampling=0 (4:4:4). Сохраняет ICC и EXIF."""
    t0 = time.time()
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(img_rgb)
    kwargs: dict = {
        "format": "JPEG", "quality": quality,
        "subsampling": 0, "optimize": False, "progressive": False,
    }
    if meta.get("icc_profile"):
        kwargs["icc_profile"] = meta["icc_profile"]
    if meta.get("exif"):
        kwargs["exif"] = _reset_exif_orientation(meta["exif"])
    buf = io.BytesIO()
    pil_out.save(buf, **kwargs)
    data = buf.getvalue()
    logger.info("Encoded: %dx%d → %.2f MB in %.2fs", W, H, len(data)/1024/1024, time.time()-t0)
    return data


def _reset_exif_orientation(exif_bytes: bytes) -> bytes:
    try:
        import piexif
        d = piexif.load(exif_bytes)
        if "0th" in d:
            d["0th"][piexif.ImageIFD.Orientation] = 1
        return piexif.dump(d)
    except Exception:
        return exif_bytes


# ── Mask helpers ───────────────────────────────────────────────────────────────

def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return mask.astype(np.float32)
    k = radius * 2 + 1
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (k, k), radius / 2.0), 0, 1)


def dilate_mask(mask: np.ndarray, ksize: int = 5, iters: int = 1) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate((mask * 255).astype(np.uint8), k, iterations=iters).astype(np.float32) / 255.0


def erode_mask(mask: np.ndarray, ksize: int = 3, iters: int = 1) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode((mask * 255).astype(np.uint8), k, iterations=iters).astype(np.float32) / 255.0


def blend_layers(src: np.ndarray, dst: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha, 0, 1)[..., np.newaxis]
    return np.clip(src.astype(np.float32) * a + dst.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)


def safe_crop(img: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    H, W = img.shape[:2]
    return img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
