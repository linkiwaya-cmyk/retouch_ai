"""
RetouchPipeline v5 — Photoshop-style beauty retouch

Принципы:
- CodeFormer ОТКЛЮЧЁН полностью
- НЕТ глобального blur/bilateral
- Frequency separation: high-freq (текстура) сохраняется 100%
- Inpaint только точечных дефектов (max 3% площади)
- D&B только в low-freq, только по skin mask
- Строгая skin mask: глаза/брови/губы/волосы/фон — исключены

Настройки .env:
  RETOUCH_STRENGTH=medium   # light / medium / strong
  CODEFORMER_BLEND=0        # 0 = отключён
"""
from __future__ import annotations

import logging
import os
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

# Пресеты
_PRESETS = {
    #          db    tone  blemish_max_pct
    "light":  (0.12, 0.15, 0.02),
    "medium": (0.20, 0.25, 0.03),
    "strong": (0.30, 0.38, 0.04),
}
_pname = os.environ.get("RETOUCH_STRENGTH", "medium")
_p = _PRESETS.get(_pname, _PRESETS["medium"])

DB_STR          = float(os.environ.get("DB_STRENGTH",   _p[0]))
TONE_STR        = float(os.environ.get("TONE_STRENGTH", _p[1]))
BLEMISH_MAX_PCT = float(os.environ.get("BLEMISH_MAX_PCT", _p[2]))  # max % площади для inpaint

# BiSeNet labels
_SKIN   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13}
_NOSKIN = {11, 12, 17, 18}  # глаза, губы — строго исключаем


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
            logger.info("FaceDetector: InsightFace")
        except Exception as e:
            logger.warning("InsightFace failed (%s) → RetinaFace", e)
            try:
                from facexlib.detection import init_detection_model
                self._retina = init_detection_model(
                    "retinaface_resnet50", half=False, device=str(DEVICE)
                )
                logger.info("FaceDetector: RetinaFace")
            except Exception as e2:
                logger.error("No face detector available: %s", e2)

    def detect(self, img: np.ndarray) -> list[dict]:
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
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Face Parser → строгая skin mask
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
        """float32 [0..1]. 1=кожа. Строго исключает глаза/брови/губы/волосы."""
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
        # Строго исключаем нежелательные зоны
        for lbl in _NOSKIN:
            mask[seg == lbl] = 0
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.float32) / 255.0

    def _hsv(self, face: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array([0, 15, 60]), np.array([25, 170, 255]))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return m.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# Frequency Separation (правильная реализация)
# ══════════════════════════════════════════════════════════════════════════════

def freq_separate(channel: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """
    low  = GaussianBlur(radius)     — тон, цвет
    high = channel - low             — текстура, поры (может быть отрицательным)
    recombine: np.clip(new_low + high, 0, 255)
    """
    k = radius * 2 + 1
    low = cv2.GaussianBlur(channel.astype(np.float32), (k, k), radius / 2.0)
    high = channel.astype(np.float32) - low
    return low, high


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Точечное удаление дефектов (строгий лимит)
# ══════════════════════════════════════════════════════════════════════════════

def remove_blemishes(face: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """
    Inpaint только реальных точечных дефектов.
    Лимит: не более BLEMISH_MAX_PCT% площади лица.
    Если маска слишком большая — пропускаем (это не дефекты, а особенности).
    """
    H, W = face.shape[:2]
    total_px = H * W
    max_px = int(total_px * BLEMISH_MAX_PCT)

    # Радиус под разрешение
    r = max(21, int(min(H, W) * 0.04))
    if r % 2 == 0: r += 1

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype(np.float32)

    local_mean = cv2.GaussianBlur(L, (r, r), 0)
    diff = np.abs(L - local_mean)

    skin_px = diff[skin > 0.5]
    if len(skin_px) < 100:
        return face

    # Адаптивный порог — топ 10% аномалий
    thresh = np.percentile(skin_px, 90)
    thresh = max(thresh, 10.0)

    blemish = ((diff > thresh) & (skin > 0.5)).astype(np.uint8) * 255

    # Убираем шум, небольшое расширение
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blemish = cv2.morphologyEx(blemish, cv2.MORPH_OPEN, k3)
    blemish = cv2.dilate(blemish, k5, iterations=1)

    n_px = int(blemish.sum() / 255)

    if n_px == 0:
        return face

    # СТРОГИЙ ЛИМИТ — если маска слишком большая, это не дефекты
    if n_px > max_px:
        logger.info(
            "Blemish mask %d px > limit %d px (%.1f%%) — skipping",
            n_px, max_px, BLEMISH_MAX_PCT * 100
        )
        return face

    result = cv2.inpaint(face, blemish, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    logger.info("Blemish inpaint: %d px (%.1f%% of face)", n_px, n_px/total_px*100)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Выравнивание тона (только low-freq, только кожа)
# ══════════════════════════════════════════════════════════════════════════════

def even_skin_tone(face: np.ndarray, skin_soft: np.ndarray, strength: float = TONE_STR) -> np.ndarray:
    """
    Выравнивает неровный тон кожи (покраснения, пятна).
    Работает ТОЛЬКО с low-frequency слоем в Lab пространстве.
    High-frequency (текстура) возвращается НЕТРОНУТОЙ.
    """
    if strength < 0.01:
        return face

    H, W = face.shape[:2]
    # Большой радиус = работаем только с крупными неравномерностями тона
    r = max(41, int(min(H, W) * 0.09))
    if r % 2 == 0: r += 1

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)

    m = skin_soft > 0.3
    if m.sum() < 200:
        return face

    result_lab = lab.copy()

    # L канал — яркость
    L_low, L_high = freq_separate(lab[:, :, 0], r)
    L_mean = float(L_low[m].mean())
    # Сдвигаем low-freq к среднему — убираем крупные пятна/тени
    correction = (L_mean - L_low) * skin_soft * strength
    result_lab[:, :, 0] = np.clip(L_low + correction + L_high, 0, 255)

    # a канал — красный/зелёный (покраснения)
    a_low, a_high = freq_separate(lab[:, :, 1], r)
    a_mean = float(a_low[m].mean())
    a_correction = (a_mean - a_low) * skin_soft * strength * 0.5
    result_lab[:, :, 1] = np.clip(a_low + a_correction + a_high, 0, 255)

    return cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Dodge & Burn (правильный — только low-freq)
# ══════════════════════════════════════════════════════════════════════════════

def dodge_and_burn(face: np.ndarray, skin_soft: np.ndarray, strength: float = DB_STR) -> np.ndarray:
    """
    Профессиональный Dodge & Burn:
    1. Разделяем L канал на low-freq + high-freq
    2. D&B применяем ТОЛЬКО к low-freq (тон)
    3. high-freq (текстура, поры) возвращаем 100% НЕТРОНУТОЙ
    4. Работаем только по skin_soft маске

    Это именно то что делает ретушёр в Photoshop на D&B слоях.
    """
    if strength < 0.01:
        return face

    H, W = face.shape[:2]

    # Радиус для крупных зон света/тени
    r_low = max(51, int(min(H, W) * 0.11))
    if r_low % 2 == 0: r_low += 1

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:, :, 0]

    m = skin_soft > 0.2
    if m.sum() < 200:
        return face

    # ── Разделяем L ────────────────────────────────────────────────────────
    L_low, L_high = freq_separate(L, r_low)

    # ── D&B на low-freq ────────────────────────────────────────────────────
    L_mean = float(L_low[m].mean())

    # Отклонение от среднего: тёмные участки -> dodge, светлые -> burn
    deviation = L_mean - L_low

    # Применяем коррекцию с feathered skin mask
    L_low_db = L_low + deviation * skin_soft * strength

    # ── Рекомбинируем: low_corrected + HIGH_FREQ_ORIGINAL ──────────────────
    # HIGH-FREQ НЕТРОНУТ — текстура и поры сохранены полностью
    L_result = np.clip(L_low_db + L_high, 0, 255)

    lab[:, :, 0] = L_result
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# Face helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pad_bbox(bbox, H, W, pad=0.30):
    x1, y1, x2, y2 = bbox
    bw, bh = x2-x1, y2-y1
    return [
        max(0, x1 - int(bw*pad)), max(0, y1 - int(bh*pad)),
        min(W, x2 + int(bw*pad)), min(H, y2 + int(bh*pad)),
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
            "Pipeline v5 ready | strength=%s db=%.2f tone=%.2f blemish_max=%.1f%%",
            _pname, DB_STR, TONE_STR, BLEMISH_MAX_PCT * 100,
        )

    def run(self, img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        H, W = img_bgr.shape[:2]
        result = img_bgr.copy()
        stats = {
            "faces": 0,
            "strength": _pname,
            "db": DB_STR,
            "tone": TONE_STR,
            "blemish_max_pct": BLEMISH_MAX_PCT,
            "codeformer": "disabled",
        }

        faces = self.detector.detect(img_bgr)
        stats["faces"] = len(faces)
        logger.info("Faces detected: %d", len(faces))

        if not faces:
            logger.info("No faces — pipeline skipped")
            return result, stats

        for fi in faces:
            x1, y1, x2, y2 = _pad_bbox(fi["bbox"], H, W, pad=0.30)
            crop = safe_crop(img_bgr, x1, y1, x2, y2)
            if crop.size == 0:
                continue

            fH, fW = crop.shape[:2]
            logger.info("Face crop: %dx%d at [%d,%d,%d,%d]", fW, fH, x1, y1, x2, y2)

            try:
                # ── 1. Строгая skin mask ───────────────────────────────────
                skin_raw = self.parser.skin_mask(crop)

                # Эродируем — убираем края маски где возможны артефакты
                skin_strict = erode_mask(skin_raw, ksize=3, iters=1)

                # Мягкая версия для плавных переходов
                skin_soft = feather_mask(
                    dilate_mask(skin_strict, ksize=5, iters=1),
                    radius=max(8, int(min(fH, fW) * 0.012)),
                )

                # ── 2. Точечное удаление дефектов (строгий лимит) ─────────
                processed = remove_blemishes(crop, skin_strict)

                # ── 3. Выравнивание тона (low-freq, только кожа) ──────────
                processed = even_skin_tone(processed, skin_soft, strength=TONE_STR)

                # ── 4. Dodge & Burn (low-freq, текстура нетронута) ─────────
                processed = dodge_and_burn(processed, skin_soft, strength=DB_STR)

                # ── 5. Composite — бесшовная вставка ──────────────────────
                # Маска вставки: плавные края, центр = 1.0
                pad_px = max(20, int(min(fH, fW) * 0.06))
                comp_mask = np.zeros((fH, fW), np.float32)
                comp_mask[pad_px:-pad_px, pad_px:-pad_px] = 1.0
                comp_mask = feather_mask(comp_mask, radius=pad_px)

                final = blend_layers(processed, crop, comp_mask)
                result[y1:y2, x1:x2] = final

            except Exception as e:
                logger.exception("Face processing error: %s", e)

        logger.info("Pipeline done: faces=%d strength=%s", stats["faces"], _pname)
        return result, stats
