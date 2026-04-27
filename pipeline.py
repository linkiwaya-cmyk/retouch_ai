"""
RetouchPipeline v6 — заметная профессиональная ретушь, быстро (<60 сек)

Изменения от v5:
- Усиленные параметры D&B и tone (ретушь теперь видна)
- Строгий лимит blemish inpaint (max 3%)
- Правильная frequency separation (high-freq 100% нетронут)
- Детальное логирование времени каждого этапа
- CodeFormer отключён полностью
- Только быстрые OpenCV операции

Принцип: как Photoshop D&B layers — работаем ТОЛЬКО с low-freq тоном,
high-freq (поры, текстура) возвращаем нетронутым.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from utils import blend_layers, feather_mask, dilate_mask, erode_mask, safe_crop

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARSING_DIR = os.environ.get("BISENET_WEIGHT_DIR", str(Path.home() / ".cache/facexlib"))
FACE_CONF   = float(os.environ.get("FACE_CONF", "0.5"))

# ── Пресеты силы ретуши ────────────────────────────────────────────────────────
# Усилены по сравнению с v5 — ретушь теперь будет заметна
_PRESETS = {
    #          db    tone   blemish_pct
    "light":  (0.22, 0.28,  0.03),
    "medium": (0.38, 0.45,  0.03),
    "strong": (0.55, 0.62,  0.04),
}
_pname = os.environ.get("RETOUCH_STRENGTH", "medium")
_p = _PRESETS.get(_pname, _PRESETS["medium"])

DB_STR          = float(os.environ.get("DB_STRENGTH",    _p[0]))
TONE_STR        = float(os.environ.get("TONE_STRENGTH",  _p[1]))
BLEMISH_MAX_PCT = float(os.environ.get("BLEMISH_MAX_PCT", _p[2]))

# BiSeNet labels
_SKIN   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13}
_NOSKIN = {11, 12, 17, 18}  # глаза, губы — строго НЕ трогаем


# ══════════════════════════════════════════════════════════════════════════════
# Face Detector
# ══════════════════════════════════════════════════════════════════════════════

class FaceDetector:
    def __init__(self):
        self._app = None
        self._retina = None
        self._init()

    def _init(self):
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
            self._app = app
            logger.info("FaceDetector: InsightFace buffalo_l")
        except Exception as e:
            logger.warning("InsightFace failed (%s) → RetinaFace fallback", e)
            try:
                from facexlib.detection import init_detection_model
                self._retina = init_detection_model(
                    "retinaface_resnet50", half=False, device=str(DEVICE)
                )
                logger.info("FaceDetector: RetinaFace")
            except Exception as e2:
                logger.error("No face detector: %s", e2)

    def detect(self, img: np.ndarray) -> list[dict]:
        t0 = time.time()
        if self._app:
            faces = self._app.get(img)
            out = [
                {"bbox": f.bbox.astype(int).tolist(), "score": float(f.det_score)}
                for f in faces if f.det_score >= FACE_CONF
            ]
        elif self._retina:
            with torch.no_grad():
                boxes = self._retina.detect_faces(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), conf_threshold=FACE_CONF
                )
            out = [
                {"bbox": [int(b[0]),int(b[1]),int(b[2]),int(b[3])], "score": float(b[4])}
                for b in boxes
            ]
        else:
            return []
        out.sort(
            key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]),
            reverse=True,
        )
        logger.info("Face detection: %d faces in %.2fs", len(out), time.time()-t0)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Face Parser → skin mask
# ══════════════════════════════════════════════════════════════════════════════

class FaceParser:
    def __init__(self):
        self._net = None
        self._init()

    def _init(self):
        try:
            from facexlib.parsing import init_parsing_model
            self._net = init_parsing_model(
                model_name="bisenet", device=str(DEVICE), model_rootpath=PARSING_DIR
            )
            self._net.eval()
            logger.info("FaceParser: BiSeNet OK")
        except Exception as e:
            logger.warning("BiSeNet failed: %s → HSV fallback", e)

    def skin_mask(self, face_bgr: np.ndarray) -> np.ndarray:
        """float32 [0..1]. 1=кожа. Глаза/губы/волосы исключены."""
        return self._bisenet(face_bgr) if self._net else self._hsv(face_bgr)

    def _bisenet(self, face: np.ndarray) -> np.ndarray:
        H, W = face.shape[:2]
        x = cv2.resize(face, (512, 512))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            seg = self._net(t)[0].squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
        mask = np.zeros((512, 512), np.uint8)
        for lbl in _SKIN:
            mask[seg == lbl] = 255
        for lbl in _NOSKIN:
            mask[seg == lbl] = 0
        return cv2.resize(mask, (W, H), cv2.INTER_NEAREST).astype(np.float32) / 255.0

    def _hsv(self, face: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array([0, 15, 60]), np.array([25, 170, 255]))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return m.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# Frequency Separation
# ══════════════════════════════════════════════════════════════════════════════

def freq_sep(channel: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """
    low  = GaussianBlur(radius)  — тон
    high = channel - low          — текстура (может быть отрицательным!)
    Рекомбинация: clip(new_low + high, 0, 255)
    """
    k = radius * 2 + 1
    low = cv2.GaussianBlur(channel.astype(np.float32), (k, k), radius / 2.0)
    return low, channel.astype(np.float32) - low


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Точечный inpaint дефектов (строгий лимит площади)
# ══════════════════════════════════════════════════════════════════════════════

def remove_blemishes(face: np.ndarray, skin: np.ndarray) -> np.ndarray:
    t0 = time.time()
    H, W = face.shape[:2]
    max_px = int(H * W * BLEMISH_MAX_PCT)

    r = max(21, int(min(H, W) * 0.04))
    if r % 2 == 0: r += 1

    L = cv2.cvtColor(face, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32)
    local_mean = cv2.GaussianBlur(L, (r, r), 0)
    diff = np.abs(L - local_mean)

    skin_px = diff[skin > 0.5]
    if len(skin_px) < 100:
        return face

    thresh = max(np.percentile(skin_px, 90), 10.0)
    blemish = ((diff > thresh) & (skin > 0.5)).astype(np.uint8) * 255
    blemish = cv2.morphologyEx(blemish, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    blemish = cv2.dilate(blemish, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    n_px = int(blemish.sum() / 255)
    if n_px == 0 or n_px > max_px:
        if n_px > max_px:
            logger.info("Blemish: %d px > limit %d px — skip", n_px, max_px)
        return face

    result = cv2.inpaint(face, blemish, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    logger.info("Blemish inpaint: %d px (%.1f%%) in %.2fs", n_px, n_px/H/W*100, time.time()-t0)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Выравнивание тона кожи (усиленное)
# ══════════════════════════════════════════════════════════════════════════════

def even_skin_tone(face: np.ndarray, skin_soft: np.ndarray) -> np.ndarray:
    """
    Выравнивает неровный тон (покраснения, пятна, тени).
    Работает ТОЛЬКО с low-freq слоем в Lab.
    High-freq (текстура) возвращается нетронутой.
    """
    t0 = time.time()
    if TONE_STR < 0.01:
        return face

    H, W = face.shape[:2]
    # Большой радиус = крупные зоны тона, не мелкие детали
    r = max(45, int(min(H, W) * 0.10))
    if r % 2 == 0: r += 1

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)
    m = skin_soft > 0.25
    if m.sum() < 200:
        return face

    out = lab.copy()

    # L — яркость (убираем крупные пятна света/тени)
    L_low, L_high = freq_sep(lab[:, :, 0], r)
    L_mean = float(L_low[m].mean())
    correction_L = (L_mean - L_low) * skin_soft * TONE_STR
    out[:, :, 0] = np.clip(L_low + correction_L + L_high, 0, 255)

    # a — красный/зелёный (убираем покраснения, но слабее)
    a_low, a_high = freq_sep(lab[:, :, 1], r)
    a_mean = float(a_low[m].mean())
    correction_a = (a_mean - a_low) * skin_soft * TONE_STR * 0.55
    out[:, :, 1] = np.clip(a_low + correction_a + a_high, 0, 255)

    result = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_Lab2BGR)
    logger.info("Tone evening done in %.2fs", time.time()-t0)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Dodge & Burn (усиленный, правильный)
# ══════════════════════════════════════════════════════════════════════════════

def dodge_and_burn(face: np.ndarray, skin_soft: np.ndarray) -> np.ndarray:
    """
    Профессиональный D&B:
    1. freq_sep L канала → low + high
    2. D&B ТОЛЬКО на low_freq (тон)
    3. high_freq (текстура, поры) возвращается 100% нетронутым
    4. Только по skin_soft mask

    Это именно то что делает ретушёр на D&B слоях в Photoshop.
    """
    t0 = time.time()
    if DB_STR < 0.01:
        return face

    H, W = face.shape[:2]
    m = skin_soft > 0.2
    if m.sum() < 200:
        return face

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:, :, 0]

    # Крупные зоны света/тени (большой радиус)
    r_coarse = max(55, int(min(H, W) * 0.12))
    if r_coarse % 2 == 0: r_coarse += 1

    # Средние неровности
    r_mid = max(25, int(min(H, W) * 0.055))
    if r_mid % 2 == 0: r_mid += 1

    # ── Coarse D&B ─────────────────────────────────────────────────────────
    L_low_c, L_high_c = freq_sep(L, r_coarse)
    L_mean_c = float(L_low_c[m].mean())
    # Dodge тёмных, Burn светлых → выравниваем тон
    correction_c = (L_mean_c - L_low_c) * skin_soft * DB_STR
    L_low_c_db = L_low_c + correction_c

    # ── Mid-tone D&B ────────────────────────────────────────────────────────
    L_low_m, L_high_m = freq_sep(L, r_mid)
    L_mean_m = float(L_low_m[m].mean())
    correction_m = (L_mean_m - L_low_m) * skin_soft * DB_STR * 0.45
    L_low_m_db = L_low_m + correction_m

    # ── Рекомбинируем ──────────────────────────────────────────────────────
    # Берём coarse коррекцию + mid коррекцию + HIGH-FREQ НЕТРОНУТЫЙ
    L_db = L_low_c_db + (L_low_m_db - L_low_m) + L_high_c
    lab[:, :, 0] = np.clip(L_db, 0, 255)

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    logger.info("D&B done in %.2fs (db=%.2f)", time.time()-t0, DB_STR)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Face helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pad_bbox(bbox, H, W, pad=0.30):
    x1, y1, x2, y2 = bbox
    bw, bh = x2-x1, y2-y1
    return [
        max(0, x1-int(bw*pad)), max(0, y1-int(bh*pad)),
        min(W, x2+int(bw*pad)), min(H, y2+int(bh*pad)),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class RetouchPipeline:
    def __init__(self):
        self.detector: Optional[FaceDetector] = None
        self.parser:   Optional[FaceParser]   = None

    def load_models(self):
        logger.info("Loading FaceDetector…")
        self.detector = FaceDetector()
        logger.info("Loading FaceParser…")
        self.parser = FaceParser()
        logger.info(
            "Pipeline v6 ready | strength=%s db=%.2f tone=%.2f blemish_max=%.0f%%",
            _pname, DB_STR, TONE_STR, BLEMISH_MAX_PCT * 100,
        )

    def run(self, img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        t_total = time.time()
        H, W = img_bgr.shape[:2]
        result = img_bgr.copy()
        stats = {
            "faces": 0, "strength": _pname,
            "db": DB_STR, "tone": TONE_STR,
            "codeformer": "disabled",
        }

        # ── Face detection ─────────────────────────────────────────────────
        t0 = time.time()
        faces = self.detector.detect(img_bgr)
        stats["faces"] = len(faces)
        stats["detect_time"] = round(time.time()-t0, 2)

        if not faces:
            logger.info("No faces detected — pipeline skipped")
            return result, stats

        # ── Process each face ──────────────────────────────────────────────
        t_retouch = time.time()
        for fi in faces:
            x1, y1, x2, y2 = _pad_bbox(fi["bbox"], H, W, pad=0.30)
            crop = safe_crop(img_bgr, x1, y1, x2, y2)
            if crop.size == 0:
                continue

            fH, fW = crop.shape[:2]
            logger.info("Face crop: %dx%d", fW, fH)

            try:
                # 1. Skin mask (строгая)
                t_s = time.time()
                skin_raw = self.parser.skin_mask(crop)
                skin_strict = erode_mask(skin_raw, ksize=3, iters=1)
                skin_soft = feather_mask(
                    dilate_mask(skin_strict, ksize=5, iters=1),
                    radius=max(8, int(min(fH, fW) * 0.012)),
                )
                logger.info("Skin mask: %.2fs", time.time()-t_s)

                # 2. Blemish removal (строгий лимит)
                processed = remove_blemishes(crop, skin_strict)

                # 3. Tone evening (усиленное)
                processed = even_skin_tone(processed, skin_soft)

                # 4. Dodge & Burn (усиленный)
                processed = dodge_and_burn(processed, skin_soft)

                # 5. Composite — бесшовная вставка в оригинал
                pad_px = max(20, int(min(fH, fW) * 0.06))
                comp = np.zeros((fH, fW), np.float32)
                comp[pad_px:-pad_px, pad_px:-pad_px] = 1.0
                comp = feather_mask(comp, radius=pad_px)
                result[y1:y2, x1:x2] = blend_layers(processed, crop, comp)

            except Exception as e:
                logger.exception("Face processing error: %s", e)

        stats["retouch_time"] = round(time.time()-t_retouch, 2)
        stats["total_time"]   = round(time.time()-t_total, 2)
        logger.info(
            "Pipeline done: faces=%d total=%.2fs (detect=%.2fs retouch=%.2fs)",
            stats["faces"], stats["total_time"],
            stats.get("detect_time", 0), stats.get("retouch_time", 0),
        )
        return result, stats
