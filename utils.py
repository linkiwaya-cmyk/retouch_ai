"""
utils.py — image I/O, HEIC/HEIF, EXIF, quality=100
"""
from __future__ import annotations
import io
import logging
import numpy as np
import cv2
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# HEIC — критически важно для iPhone
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    logger.info("HEIC/HEIF: enabled")
except ImportError:
    logger.warning("HEIC/HEIF: DISABLED — pip install pillow-heif")


def decode_image(raw_bytes: bytes) -> tuple[np.ndarray, dict]:
    """
    Декодирует любой формат → BGR uint8.
    Автоматически исправляет EXIF ориентацию (iPhone).
    """
    meta = {"exif": None, "icc_profile": None, "original_size": None, "format": "JPEG"}

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        meta["format"] = pil_img.format or "JPEG"
        meta["exif"] = pil_img.info.get("exif")
        meta["icc_profile"] = pil_img.info.get("icc_profile")

        # Исправляем ориентацию iPhone до конвертации
        pil_img = ImageOps.exif_transpose(pil_img)

        if pil_img.mode == "RGBA":
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[3])
            pil_img = bg
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        meta["original_size"] = pil_img.size  # (W, H)
        w, h = pil_img.size
        logger.info("Decoded: %s %dx%d", meta["format"], w, h)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), meta

    except Exception as exc:
        logger.warning("PIL decode failed: %s — cv2 fallback", exc)

    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image (unsupported format).")
    meta["original_size"] = (img.shape[1], img.shape[0])
    return img, meta


def encode_image_to_bytes(img_bgr: np.ndarray, meta: dict, quality: int = 100) -> bytes:
    """
    JPEG quality=100, subsampling=0 (4:4:4).
    Сохраняет ICC и EXIF (orientation сброшен в 1).
    """
    h, w = img_bgr.shape[:2]
    logger.info("Encoding: %dx%d quality=%d", w, h, quality)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(img_rgb)

    kwargs: dict = {
        "format": "JPEG",
        "quality": quality,
        "subsampling": 0,
        "optimize": False,
        "progressive": False,
    }
    if meta.get("icc_profile"):
        kwargs["icc_profile"] = meta["icc_profile"]
    if meta.get("exif"):
        kwargs["exif"] = _reset_orientation(meta["exif"])

    buf = io.BytesIO()
    pil_out.save(buf, **kwargs)
    data = buf.getvalue()
    logger.info("Output: %.2f MB", len(data) / 1024 / 1024)
    return data


def _reset_orientation(exif_bytes: bytes) -> bytes:
    try:
        import piexif
        d = piexif.load(exif_bytes)
        d.get("0th", {})[piexif.ImageIFD.Orientation] = 1
        return piexif.dump(d)
    except Exception:
        return exif_bytes


# ── Mask helpers ───────────────────────────────────────────────────────────────

def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    k = radius * 2 + 1
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (k, k), radius / 2), 0, 1)


def dilate_mask(mask: np.ndarray, ksize: int = 5, iters: int = 1) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask, k, iterations=iters)


def erode_mask(mask: np.ndarray, ksize: int = 5, iters: int = 1) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(mask, k, iterations=iters)


def blend_layers(src: np.ndarray, dst: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """src * alpha + dst * (1-alpha). alpha = float32 [0..1], broadcast over channels."""
    a = alpha[..., np.newaxis]
    return np.clip(src.astype(np.float32) * a + dst.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)


def safe_crop(img: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    H, W = img.shape[:2]
    return img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
