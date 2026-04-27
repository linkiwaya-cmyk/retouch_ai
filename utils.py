"""
utils.py — image I/O, HEIC/HEIF fix, EXIF, quality=100

HEIC FIX: pillow_heif.register_heif_opener() вызывается на уровне модуля,
до любого Image.open() — это критично для iPhone фото.
"""
from __future__ import annotations

import io
import logging
import numpy as np
import cv2
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# ── HEIC — регистрируем СРАЗУ при импорте модуля ──────────────────────────────
_HEIC_OK = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIC_OK = True
    logger.info("HEIC/HEIF support: ENABLED")
except ImportError:
    logger.warning("HEIC/HEIF support: DISABLED — run: pip install pillow-heif")
except Exception as e:
    logger.warning("HEIC/HEIF register failed: %s", e)


def decode_image(raw_bytes: bytes) -> tuple[np.ndarray, dict]:
    """
    Декодирует JPEG / PNG / HEIC / HEIF / WebP → BGR uint8.
    Автоматически исправляет EXIF ориентацию (критично для iPhone).

    Returns:
        img_bgr: np.ndarray (H, W, 3)
        meta: dict {exif, icc_profile, original_size, format}
    """
    meta = {
        "exif": None,
        "icc_profile": None,
        "original_size": None,
        "format": "JPEG",
    }

    # ── Попытка 1: PIL (поддерживает HEIC если зарегистрирован) ───────────
    try:
        buf = io.BytesIO(raw_bytes)
        pil_img = Image.open(buf)
        pil_img.load()  # принудительная загрузка (важно для HEIC)

        meta["format"] = pil_img.format or "JPEG"
        meta["exif"] = pil_img.info.get("exif")
        meta["icc_profile"] = pil_img.info.get("icc_profile")

        # Исправляем ориентацию (iPhone снимает с rotation в EXIF)
        pil_img = ImageOps.exif_transpose(pil_img)

        # Конвертируем в RGB
        if pil_img.mode == "RGBA":
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[3])
            pil_img = bg
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        meta["original_size"] = pil_img.size  # (W, H)
        w, h = pil_img.size
        logger.info("Decoded: %s %dx%d", meta["format"], w, h)

        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img_bgr, meta

    except Exception as exc:
        logger.warning("PIL decode failed: %s", exc)

    # ── Попытка 2: cv2 fallback (JPEG/PNG без EXIF) ────────────────────────
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        meta["original_size"] = (img.shape[1], img.shape[0])
        logger.info("Decoded via cv2: %dx%d", img.shape[1], img.shape[0])
        return img, meta

    # ── Попытка 3: HEIC через pillow_heif напрямую ─────────────────────────
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
            logger.info("Decoded HEIC directly: %dx%d", pil_img.size[0], pil_img.size[1])
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), meta
        except Exception as exc2:
            logger.warning("HEIC direct decode failed: %s", exc2)

    raise ValueError(
        "Cannot decode image. Supported formats: JPEG, PNG, HEIC, HEIF, WebP. "
        f"HEIC support: {'enabled' if _HEIC_OK else 'DISABLED (pip install pillow-heif)'}."
    )


def encode_image_to_bytes(img_bgr: np.ndarray, meta: dict, quality: int = 100) -> bytes:
    """
    JPEG quality=100, subsampling=0 (4:4:4), optimize=False, progressive=False.
    Сохраняет ICC профиль и EXIF (orientation=1).
    """
    h, w = img_bgr.shape[:2]
    logger.info("Encoding: %dx%d quality=%d", w, h, quality)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(img_rgb)

    kwargs: dict = {
        "format": "JPEG",
        "quality": quality,
        "subsampling": 0,      # 4:4:4
        "optimize": False,
        "progressive": False,
    }

    if meta.get("icc_profile"):
        kwargs["icc_profile"] = meta["icc_profile"]

    if meta.get("exif"):
        kwargs["exif"] = _reset_exif_orientation(meta["exif"])

    buf = io.BytesIO()
    pil_out.save(buf, **kwargs)
    data = buf.getvalue()
    logger.info("Output: %.2f MB", len(data) / 1024 / 1024)
    return data


def _reset_exif_orientation(exif_bytes: bytes) -> bytes:
    """Сбрасывает Orientation=1 чтобы не было двойного поворота."""
    try:
        import piexif
        d = piexif.load(exif_bytes)
        if "0th" in d and piexif.ImageIFD.Orientation in d["0th"]:
            d["0th"][piexif.ImageIFD.Orientation] = 1
        return piexif.dump(d)
    except Exception:
        return exif_bytes  # возвращаем как есть


# ── Mask helpers ───────────────────────────────────────────────────────────────

def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return mask.astype(np.float32)
    k = radius * 2 + 1
    return np.clip(
        cv2.GaussianBlur(mask.astype(np.float32), (k, k), radius / 2.0),
        0.0, 1.0
    )


def dilate_mask(mask: np.ndarray, ksize: int = 5, iters: int = 1) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask.astype(np.uint8), k, iterations=iters).astype(np.float32)


def erode_mask(mask: np.ndarray, ksize: int = 3, iters: int = 1) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(mask.astype(np.uint8), k, iterations=iters).astype(np.float32)


def blend_layers(src: np.ndarray, dst: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """alpha=1 → src, alpha=0 → dst"""
    a = np.clip(alpha, 0, 1)[..., np.newaxis]
    return np.clip(
        src.astype(np.float32) * a + dst.astype(np.float32) * (1.0 - a),
        0, 255
    ).astype(np.uint8)


def safe_crop(img: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    H, W = img.shape[:2]
    return img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
