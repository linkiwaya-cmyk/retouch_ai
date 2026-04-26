"""
Utility functions: image encoding/decoding, color conversions, mask helpers.
"""

from __future__ import annotations

import cv2
import numpy as np


# ── image I/O ──────────────────────────────────────────────────────────────────

def decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode raw bytes → BGR ndarray (uint8)."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None — unsupported format or corrupted data.")
    return img


def encode_image_to_bytes(img_bgr: np.ndarray, ext: str = ".jpg", quality: int = 95) -> bytes:
    """Encode BGR ndarray → compressed bytes."""
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    ok, buf = cv2.imencode(ext, img_bgr, params)
    if not ok:
        raise RuntimeError("cv2.imencode failed.")
    return buf.tobytes()


# ── color space helpers ────────────────────────────────────────────────────────

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def bgr_to_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def lab_to_bgr(img_lab: np.ndarray) -> np.ndarray:
    out = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ── mask helpers ───────────────────────────────────────────────────────────────

def dilate_mask(mask: np.ndarray, ksize: int = 5, iterations: int = 2) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask, kernel, iterations=iterations)


def erode_mask(mask: np.ndarray, ksize: int = 5, iterations: int = 1) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(mask, kernel, iterations=iterations)


def feather_mask(mask: np.ndarray, radius: int = 15) -> np.ndarray:
    """Gaussian blur on a float mask [0..1] for smooth blending."""
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), radius / 2)
    return np.clip(blurred, 0.0, 1.0)


def blend_with_mask(src: np.ndarray, dst: np.ndarray, mask_f: np.ndarray) -> np.ndarray:
    """
    Alpha-blend src and dst using a float mask [0..1].
    mask_f == 1  →  use src
    mask_f == 0  →  use dst
    """
    m = mask_f[..., np.newaxis]  # broadcast over channels
    return np.clip(src.astype(np.float32) * m + dst.astype(np.float32) * (1 - m), 0, 255).astype(np.uint8)


# ── geometry ───────────────────────────────────────────────────────────────────

def safe_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2]


def paste_region(canvas: np.ndarray, patch: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = canvas.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    ph, pw = patch.shape[:2]
    patch_resized = cv2.resize(patch, (x2c - x1c, y2c - y1c), interpolation=cv2.INTER_LANCZOS4)
    result = canvas.copy()
    result[y1c:y2c, x1c:x2c] = patch_resized
    return result
