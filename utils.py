"""
utils.py — image I/O с сохранением EXIF, ориентации, ICC профиля.
Поддержка: JPEG, PNG, HEIC/HEIF, WebP.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# HEIC поддержка
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
    logger.info("HEIC/HEIF: enabled")
except ImportError:
    HEIF_SUPPORTED = False
    logger.warning("HEIC/HEIF: disabled. pip install pillow-heif")


# ── Decode ─────────────────────────────────────────────────────────────────────

def decode_image(raw_bytes: bytes) -> tuple[np.ndarray, dict]:
    """
    Декодирует байты → BGR ndarray + метаданные.
    Автоматически корректирует EXIF ориентацию (iPhone).

    Returns:
        img_bgr: np.ndarray (H, W, 3) uint8
        meta: dict с exif, icc_profile, original_size, original_filename
    """
    meta = {
        "exif": None,
        "icc_profile": None,
        "original_size": None,  # (W, H) PIL формат
        "format": "JPEG",
    }

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        meta["format"] = pil_img.format or "JPEG"

        # Сохраняем метаданные ДО exif_transpose
        meta["exif"] = pil_img.info.get("exif", None)
        meta["icc_profile"] = pil_img.info.get("icc_profile", None)

        # Корректируем ориентацию по EXIF (iPhone снимает повёрнуто)
        pil_img = ImageOps.exif_transpose(pil_img)

        # Конвертируем в RGB
        if pil_img.mode == "RGBA":
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[3])
            pil_img = bg
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        meta["original_size"] = pil_img.size  # (W, H)
        logger.info(
            "Decoded: format=%s size=%dx%d",
            meta["format"], pil_img.size[0], pil_img.size[1],
        )

        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img_bgr, meta

    except Exception as exc:
        logger.warning("PIL decode failed (%s), fallback to cv2", exc)

    # cv2 fallback
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось декодировать изображение.")
    meta["original_size"] = (img.shape[1], img.shape[0])
    return img, meta


# ── Encode ─────────────────────────────────────────────────────────────────────

def encode_image_to_bytes(
    img_bgr: np.ndarray,
    meta: dict,
    quality: int = 98,
) -> bytes:
    """
    Кодирует BGR → JPEG байты.
    Сохраняет ICC профиль и EXIF (с orientation=1 после transpose).
    quality=98, subsampling=0 (4:4:4) — максимальное качество.
    """
    h, w = img_bgr.shape[:2]
    logger.info("Encoding output: %dx%d quality=%d", w, h, quality)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(img_rgb)

    save_kwargs: dict = {
        "format": "JPEG",
        "quality": quality,
        "subsampling": 0,    # 4:4:4 — максимальное качество хроминанса
        "optimize": False,
    }

    # ICC профиль
    if meta.get("icc_profile"):
        save_kwargs["icc_profile"] = meta["icc_profile"]

    # EXIF — сбрасываем Orientation=1 (уже применён)
    if meta.get("exif"):
        exif_out = _reset_orientation_in_exif(meta["exif"])
        if exif_out:
            save_kwargs["exif"] = exif_out

    buf = io.BytesIO()
    pil_out.save(buf, **save_kwargs)
    result = buf.getvalue()

    logger.info(
        "Output size: %.2f MB (input was approx original)",
        len(result) / 1024 / 1024,
    )
    return result


def _reset_orientation_in_exif(exif_bytes: bytes) -> bytes | None:
    """Устанавливает Orientation=1 в EXIF чтобы не было двойного поворота."""
    try:
        import piexif
        exif_dict = piexif.load(exif_bytes)
        ifd = exif_dict.get("0th", {})
        if piexif.ImageIFD.Orientation in ifd:
            ifd[piexif.ImageIFD.Orientation] = 1
        return piexif.dump(exif_dict)
    except Exception:
        return exif_bytes


# ── Mask helpers ───────────────────────────────────────────────────────────────

def dilate_mask(mask: np.ndarray, ksize: int = 5, iterations: int = 2) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask, kernel, iterations=iterations)


def feather_mask(mask: np.ndarray, radius: int = 15) -> np.ndarray:
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), radius / 2)
    return np.clip(blurred, 0.0, 1.0)


def blend_with_mask(
    src: np.ndarray,
    dst: np.ndarray,
    mask_f: np.ndarray,
) -> np.ndarray:
    """src * mask + dst * (1 - mask). mask_f = float32 [0..1]"""
    m = mask_f[..., np.newaxis]
    return np.clip(
        src.astype(np.float32) * m + dst.astype(np.float32) * (1.0 - m),
        0, 255,
    ).astype(np.uint8)


def safe_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]
    return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
