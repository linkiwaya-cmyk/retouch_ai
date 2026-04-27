"""
utils.py v8

Главные исправления:
1. FLIP FIX — EXIF orientation применяется ОДИН раз при decode,
   при encode orientation сбрасывается в 1 (уже применён).
   cv2 не трогает EXIF — поэтому все операции после decode безопасны.

2. HEIC — 4 метода, primary image only.
"""
from __future__ import annotations

import io
import logging
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

_HEIC_OK = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIC_OK = True
    logger.info("HEIC: pillow_heif OK")
except ImportError:
    logger.warning("pillow_heif not installed")
except Exception as e:
    logger.warning("pillow_heif register: %s", e)


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def decode_image(raw_bytes: bytes) -> tuple[np.ndarray, dict]:
    """
    Декодирует изображение → BGR uint8.

    FLIP FIX:
    - ImageOps.exif_transpose() применяется ОДИН раз здесь
    - После этого img_bgr всегда в правильной ориентации
    - При encode orientation в EXIF сбрасывается в 1
    - cv2 операции не трогают ориентацию — всё корректно
    """
    t0 = time.time()
    meta = {
        "exif": None,
        "icc_profile": None,
        "original_size": None,
        "format": "JPEG",
        "orientation_applied": False,
    }

    def _finalize_pil(pil_img: Image.Image, fmt: str) -> tuple[np.ndarray, dict]:
        meta["format"] = fmt
        meta["exif"] = pil_img.info.get("exif")
        meta["icc_profile"] = pil_img.info.get("icc_profile")
        # ПРИМЕНЯЕМ EXIF ORIENTATION ОДИН РАЗ
        pil_img = ImageOps.exif_transpose(pil_img)
        meta["orientation_applied"] = True
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB") if pil_img.mode != "RGBA" else (
                lambda: (lambda bg: (bg.paste(pil_img, mask=pil_img.split()[3]), bg)[1])(
                    Image.new("RGB", pil_img.size, (255, 255, 255))
                )
            )()
        if pil_img.mode == "RGBA":
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[3])
            pil_img = bg
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        meta["original_size"] = pil_img.size
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        logger.info("Decoded %s %dx%d in %.2fs", fmt, pil_img.size[0], pil_img.size[1], time.time()-t0)
        return img_bgr, meta

    # ── 1. pillow_heif.open_heif — primary image ───────────────────────────
    if _HEIC_OK:
        try:
            import pillow_heif
            hf = pillow_heif.open_heif(io.BytesIO(raw_bytes), convert_hdr_to_8bit=True)
            pil_img = hf[0].to_pillow()
            return _finalize_pil(pil_img, "HEIC")
        except Exception as e:
            logger.warning("open_heif[0] failed: %s", e)

    # ── 2. PIL универсальный ───────────────────────────────────────────────
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.load()
        return _finalize_pil(pil_img, pil_img.format or "JPEG")
    except Exception as e:
        logger.warning("PIL failed: %s", e)

    # ── 3. cv2 (JPEG/PNG, без EXIF ориентации) ────────────────────────────
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        meta["original_size"] = (img.shape[1], img.shape[0])
        # cv2 не применяет EXIF — пробуем вручную
        img = _apply_exif_rotation_cv2(raw_bytes, img)
        logger.info("Decoded cv2 %dx%d", img.shape[1], img.shape[0])
        return img, meta

    # ── 4. ImageMagick fallback ────────────────────────────────────────────
    img = _imagemagick(raw_bytes)
    if img is not None:
        meta["format"] = "HEIC"
        meta["original_size"] = (img.shape[1], img.shape[0])
        return img, meta

    raise ValueError(
        f"Cannot decode. Supported: JPEG PNG HEIC HEIF WebP. "
        f"HEIC: {'OK' if _HEIC_OK else 'DISABLED'}."
    )


def _apply_exif_rotation_cv2(raw_bytes: bytes, img: np.ndarray) -> np.ndarray:
    """Применяет EXIF ориентацию к cv2 изображению вручную."""
    try:
        import piexif
        exif = piexif.load(raw_bytes)
        orientation = exif.get("0th", {}).get(piexif.ImageIFD.Orientation, 1)
        if orientation == 3:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif orientation == 6:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 8:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass
    return img


def _imagemagick(raw_bytes: bytes) -> np.ndarray | None:
    try:
        with tempfile.NamedTemporaryFile(suffix=".heic", delete=False) as f:
            f.write(raw_bytes)
            src = f.name
        dst = src.replace(".heic", "_out.jpg")
        r = subprocess.run(["convert", "-auto-orient", src, dst],
                           capture_output=True, timeout=30)
        if r.returncode == 0 and Path(dst).exists():
            img = cv2.imread(dst)
            Path(src).unlink(missing_ok=True)
            Path(dst).unlink(missing_ok=True)
            if img is not None:
                logger.info("HEIC via ImageMagick OK")
                return img
    except Exception as e:
        logger.warning("ImageMagick: %s", e)
    return None


def encode_image_to_bytes(img_bgr: np.ndarray, meta: dict, quality: int = 100) -> bytes:
    """
    JPEG quality=100, subsampling=0.
    EXIF: orientation сброшен в 1 (уже применён при decode).
    """
    t0 = time.time()
    H, W = img_bgr.shape[:2]
    pil_out = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
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
    logger.info("Encoded %dx%d → %.2fMB in %.2fs", W, H, len(data)/1024/1024, time.time()-t0)
    return data


def _reset_exif_orientation(exif_bytes: bytes) -> bytes:
    """Сбрасывает Orientation=1 — ориентация уже применена пикселями."""
    try:
        import piexif
        d = piexif.load(exif_bytes)
        if "0th" in d:
            d["0th"][piexif.ImageIFD.Orientation] = 1
        return piexif.dump(d)
    except Exception:
        return exif_bytes


# ── Helpers ────────────────────────────────────────────────────────────────────

def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return mask.astype(np.float32)
    k = radius * 2 + 1
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (k, k), radius / 2.0), 0, 1)


def dilate_mask(mask: np.ndarray, ksize: int = 5, iters: int = 1) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate((mask*255).astype(np.uint8), k, iterations=iters).astype(np.float32)/255.0


def erode_mask(mask: np.ndarray, ksize: int = 3, iters: int = 1) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode((mask*255).astype(np.uint8), k, iterations=iters).astype(np.float32)/255.0


def blend_layers(src: np.ndarray, dst: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha, 0, 1)[..., np.newaxis]
    return np.clip(src.astype(np.float32)*a + dst.astype(np.float32)*(1-a), 0, 255).astype(np.uint8)


def safe_crop(img: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    H, W = img.shape[:2]
    return img[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
