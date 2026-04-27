"""
utils.py v7 — HEIC bulletproof, quality=100

HEIC fallback chain:
  1. pillow_heif.open_heif  — primary image only (fixes "too many auxiliary")
  2. PIL Image.open         — универсальный
  3. cv2.imdecode           — JPEG/PNG
  4. subprocess imagemagick — последний резерв
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

# ── HEIC регистрация ───────────────────────────────────────────────────────────
_HEIC_OK = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIC_OK = True
    logger.info("HEIC: pillow_heif registered")
except ImportError:
    logger.warning("pillow_heif not installed — pip install pillow-heif")
except Exception as e:
    logger.warning("pillow_heif register failed: %s", e)


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _try_imagemagick(raw_bytes: bytes) -> np.ndarray | None:
    """Fallback через ImageMagick convert."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".heic", delete=False) as f:
            f.write(raw_bytes)
            src = f.name
        dst = src.replace(".heic", ".jpg")
        r = subprocess.run(
            ["convert", src, "-quality", "95", dst],
            capture_output=True, timeout=30,
        )
        if r.returncode == 0 and Path(dst).exists():
            img = cv2.imread(dst)
            Path(src).unlink(missing_ok=True)
            Path(dst).unlink(missing_ok=True)
            if img is not None:
                logger.info("HEIC decoded via ImageMagick")
                return img
    except Exception as e:
        logger.warning("ImageMagick fallback failed: %s", e)
    return None


def decode_image(raw_bytes: bytes) -> tuple[np.ndarray, dict]:
    """
    Декодирует любой формат → BGR uint8.
    HEIC: 4 последовательных метода, берём только primary image.
    """
    t0 = time.time()
    meta = {"exif": None, "icc_profile": None, "original_size": None, "format": "JPEG"}

    # ── 1. pillow_heif.open_heif — primary image only ─────────────────────
    if _HEIC_OK:
        try:
            import pillow_heif
            hf = pillow_heif.open_heif(io.BytesIO(raw_bytes), convert_hdr_to_8bit=True)
            primary = hf[0]  # только primary, игнорируем depth/live-photo layers
            pil_img = primary.to_pillow()
            meta["exif"] = pil_img.info.get("exif")
            meta["icc_profile"] = pil_img.info.get("icc_profile")
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            meta["format"] = "HEIC"
            meta["original_size"] = pil_img.size
            img = _pil_to_bgr(pil_img)
            logger.info("Decoded HEIC (open_heif primary): %dx%d %.2fs", *pil_img.size, time.time()-t0)
            return img, meta
        except Exception as e:
            logger.warning("open_heif failed: %s", e)

    # ── 2. PIL универсальный (JPEG, PNG, WebP, HEIC если зарегистрирован) ──
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.load()
        meta["format"] = pil_img.format or "JPEG"
        meta["exif"] = pil_img.info.get("exif")
        meta["icc_profile"] = pil_img.info.get("icc_profile")
        pil_img = ImageOps.exif_transpose(pil_img)
        meta["original_size"] = pil_img.size
        img = _pil_to_bgr(pil_img)
        logger.info("Decoded via PIL: %s %dx%d %.2fs", meta["format"], *pil_img.size, time.time()-t0)
        return img, meta
    except Exception as e:
        logger.warning("PIL failed: %s", e)

    # ── 3. cv2 (JPEG/PNG) ─────────────────────────────────────────────────
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        meta["original_size"] = (img.shape[1], img.shape[0])
        logger.info("Decoded via cv2: %dx%d %.2fs", img.shape[1], img.shape[0], time.time()-t0)
        return img, meta

    # ── 4. pillow_heif.read_heif (старый API) ─────────────────────────────
    if _HEIC_OK:
        try:
            import pillow_heif
            heif = pillow_heif.read_heif(raw_bytes)
            pil_img = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
            pil_img = ImageOps.exif_transpose(pil_img.convert("RGB"))
            meta["format"] = "HEIC"
            meta["original_size"] = pil_img.size
            img = _pil_to_bgr(pil_img)
            logger.info("Decoded HEIC (read_heif): %dx%d", *pil_img.size)
            return img, meta
        except Exception as e:
            logger.warning("read_heif failed: %s", e)

    # ── 5. ImageMagick ────────────────────────────────────────────────────
    img = _try_imagemagick(raw_bytes)
    if img is not None:
        meta["format"] = "HEIC"
        meta["original_size"] = (img.shape[1], img.shape[0])
        return img, meta

    raise ValueError(
        "Cannot decode image. Supported: JPEG, PNG, HEIC, HEIF, WebP. "
        f"HEIC: {'enabled' if _HEIC_OK else 'DISABLED — pip install pillow-heif'}."
    )


def encode_image_to_bytes(img_bgr: np.ndarray, meta: dict, quality: int = 100) -> bytes:
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
    out = cv2.dilate((mask * 255).astype(np.uint8), k, iterations=iters)
    return out.astype(np.float32) / 255.0


def erode_mask(mask: np.ndarray, ksize: int = 3, iters: int = 1) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    out = cv2.erode((mask * 255).astype(np.uint8), k, iterations=iters)
    return out.astype(np.float32) / 255.0


def blend_layers(src: np.ndarray, dst: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha, 0, 1)[..., np.newaxis]
    return np.clip(src.astype(np.float32) * a + dst.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)


def safe_crop(img: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    H, W = img.shape[:2]
    return img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
