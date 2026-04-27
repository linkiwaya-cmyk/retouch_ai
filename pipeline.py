"""
RetouchPipeline v7 — новый подход, максимально близко к эталону

Анализ эталона IMG_8746 vs исходника:
- Убраны красные пятна/покраснения
- Выровнен тон (темные участки немного осветлены)
- Мелкие дефекты убраны
- Кожа ровнее, но текстура и поры ПОЛНОСТЬЮ сохранены
- Глаза, брови, губы — нетронуты, резкие

Новый подход (отличается от v6):
1. Skin mask через BiSeNet — строгая
2. Targeted redness removal — убираем покраснения точечно в a-канале
3. Low-freq luminosity leveling — выравниваем яркость крупных зон
4. Selective inpaint blemishes — точечные дефекты
5. Gentle contrast micro-boost — чуть поднимаем локальный контраст на коже
6. Composite с оригиналом через skin mask

Принцип: меняем ТОЛЬКО то что отличает эталон от оригинала.
НЕТ blur. НЕТ smoothing. НЕТ CodeFormer.
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

# ── Параметры (настраиваются через .env) ──────────────────────────────────────
# Насколько сильно убираем покраснения (0.0-1.0)
REDNESS_STR  = float(os.environ.get("REDNESS_STRENGTH",  "0.55"))
# Насколько сильно выравниваем яркость зон (D&B)
LEVEL_STR    = float(os.environ.get("LEVEL_STRENGTH",    "0.42"))
# Micro-contrast boost на коже
MICRO_STR    = float(os.environ.get("MICRO_STRENGTH",    "0.25"))
# Max площадь blemish inpaint
BLEMISH_PCT  = float(os.environ.get("BLEMISH_MAX_PCT",   "0.03"))

# BiSeNet labels
_SKIN   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13}
_NOSKIN = {11, 12, 17, 18}  # глаза, губы — НЕ трогаем


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
            logger.warning("InsightFace failed: %s", e)
            try:
                from facexlib.detection import init_detection_model
                self._retina = init_detection_model(
                    "retinaface_resnet50", half=False, device=str(DEVICE)
                )
                logger.info("FaceDetector: RetinaFace fallback")
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
            out = [{"bbox": [int(b[0]),int(b[1]),int(b[2]),int(b[3])], "score": float(b[4])} for b in boxes]
        else:
            return []
        out.sort(key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]), reverse=True)
        logger.info("Detected %d faces in %.2fs", len(out), time.time()-t0)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Face Parser
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
            logger.warning("BiSeNet failed: %s", e)

    def skin_mask(self, face: np.ndarray) -> np.ndarray:
        return self._bisenet(face) if self._net else self._hsv(face)

    def _bisenet(self, face: np.ndarray) -> np.ndarray:
        H, W = face.shape[:2]
        x = cv2.resize(face, (512, 512))
        x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
             - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        t = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            seg = self._net(t)[0].squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
        mask = np.zeros((512,512), np.uint8)
        for lbl in _SKIN:   mask[seg == lbl] = 255
        for lbl in _NOSKIN: mask[seg == lbl] = 0
        return cv2.resize(mask, (W,H), cv2.INTER_NEAREST).astype(np.float32)/255.0

    def _hsv(self, face: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array([0,15,60]), np.array([25,170,255]))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)).astype(np.float32)/255.0


# ══════════════════════════════════════════════════════════════════════════════
# Core: частотное разделение
# ══════════════════════════════════════════════════════════════════════════════

def freq_sep(ch: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """low = blur(r), high = ch - low. Рекомбинация: clip(new_low + high, 0,255)"""
    k = r*2+1
    low = cv2.GaussianBlur(ch.astype(np.float32), (k,k), r/2.0)
    return low, ch.astype(np.float32) - low


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Убираем покраснения (самое заметное отличие эталона)
# ══════════════════════════════════════════════════════════════════════════════

def remove_redness(face: np.ndarray, skin_soft: np.ndarray, strength: float = REDNESS_STR) -> np.ndarray:
    """
    Убирает покраснения и неровный цветовой тон кожи.
    Работает с a-каналом (красный/зелёный) в Lab.
    Выравнивает low-freq a-канал к среднему значению по коже.
    High-freq a (детали цвета) сохраняется.
    """
    if strength < 0.01:
        return face

    H, W = face.shape[:2]
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)
    m = skin_soft > 0.3

    if m.sum() < 200:
        return face

    # Большой радиус — убираем только крупные цветовые пятна
    r_color = max(45, int(min(H, W) * 0.09))
    if r_color % 2 == 0: r_color += 1

    # a-канал: покраснения
    a_low, a_high = freq_sep(lab[:,:,1], r_color)
    a_mean = float(a_low[m].mean())
    # Выравниваем к среднему — убираем локальные покраснения
    a_correction = (a_mean - a_low) * skin_soft * strength
    lab[:,:,1] = np.clip(a_low + a_correction + a_high, 0, 255)

    # b-канал: желтый/синий тон (слабее)
    b_low, b_high = freq_sep(lab[:,:,2], r_color)
    b_mean = float(b_low[m].mean())
    b_correction = (b_mean - b_low) * skin_soft * strength * 0.35
    lab[:,:,2] = np.clip(b_low + b_correction + b_high, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Выравнивание яркости зон кожи (Dodge & Burn)
# ══════════════════════════════════════════════════════════════════════════════

def level_skin_luminosity(face: np.ndarray, skin_soft: np.ndarray, strength: float = LEVEL_STR) -> np.ndarray:
    """
    Выравнивает яркость зон кожи — как Dodge & Burn на отдельном слое.
    Осветляет тёмные участки, слегка затемняет пересветы.
    Работает ТОЛЬКО с low-freq L-канала.
    High-freq (текстура, поры) возвращается 100% нетронутой.
    """
    if strength < 0.01:
        return face

    H, W = face.shape[:2]
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:,:,0]
    m = skin_soft > 0.2

    if m.sum() < 200:
        return face

    # Крупные зоны (coarse D&B)
    r_coarse = max(55, int(min(H, W) * 0.11))
    if r_coarse % 2 == 0: r_coarse += 1

    L_low_c, L_high_c = freq_sep(L, r_coarse)
    L_mean_c = float(L_low_c[m].mean())
    # Dodge тёмных участков кожи, Burn пересветов
    correction_c = (L_mean_c - L_low_c) * skin_soft * strength
    L_low_c_new = L_low_c + correction_c

    # Средние зоны (mid-tone D&B)
    r_mid = max(25, int(min(H, W) * 0.05))
    if r_mid % 2 == 0: r_mid += 1

    L_low_m, L_high_m = freq_sep(L, r_mid)
    L_mean_m = float(L_low_m[m].mean())
    correction_m = (L_mean_m - L_low_m) * skin_soft * strength * 0.40
    L_low_m_new = L_low_m + correction_m

    # Рекомбинируем: новый low + ОРИГИНАЛЬНЫЙ high-freq (текстура)
    L_result = L_low_c_new + (L_low_m_new - L_low_m) + L_high_c
    lab[:,:,0] = np.clip(L_result, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Micro-contrast boost (локальный контраст кожи)
# ══════════════════════════════════════════════════════════════════════════════

def micro_contrast_boost(face: np.ndarray, skin_soft: np.ndarray, strength: float = MICRO_STR) -> np.ndarray:
    """
    Лёгкий unsharp mask только на коже — кожа выглядит более чёткой и детальной.
    Усиливает mid-frequency детали (не мелкую текстуру, а структуру кожи).
    """
    if strength < 0.01:
        return face

    H, W = face.shape[:2]
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:,:,0]

    # mid-freq radius
    r = max(7, int(min(H, W) * 0.015))
    if r % 2 == 0: r += 1

    L_blur = cv2.GaussianBlur(L, (r*2+1, r*2+1), r/2.0)
    # Unsharp mask: усиливаем разницу между оригиналом и blur
    sharpened = L + (L - L_blur) * skin_soft * strength
    lab[:,:,0] = np.clip(sharpened, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Точечный inpaint дефектов
# ══════════════════════════════════════════════════════════════════════════════

def remove_blemishes(face: np.ndarray, skin: np.ndarray) -> np.ndarray:
    H, W = face.shape[:2]
    max_px = int(H * W * BLEMISH_PCT)

    r = max(21, int(min(H, W) * 0.04))
    if r % 2 == 0: r += 1

    L = cv2.cvtColor(face, cv2.COLOR_BGR2Lab)[:,:,0].astype(np.float32)
    diff = np.abs(L - cv2.GaussianBlur(L, (r,r), 0))

    skin_px = diff[skin > 0.5]
    if len(skin_px) < 100:
        return face

    thresh = max(np.percentile(skin_px, 90), 10.0)
    blemish = ((diff > thresh) & (skin > 0.5)).astype(np.uint8) * 255
    blemish = cv2.morphologyEx(blemish, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    blemish = cv2.dilate(blemish, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=1)

    n_px = int(blemish.sum() / 255)
    if n_px == 0 or n_px > max_px:
        if n_px > max_px:
            logger.info("Blemish: %d px > limit %d — skip", n_px, max_px)
        return face

    result = cv2.inpaint(face, blemish, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    logger.info("Blemish inpaint: %d px (%.1f%%)", n_px, n_px/H/W*100)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pad_bbox(bbox, H, W, pad=0.30):
    x1,y1,x2,y2 = bbox
    bw,bh = x2-x1, y2-y1
    return [max(0,x1-int(bw*pad)), max(0,y1-int(bh*pad)),
            min(W,x2+int(bw*pad)), min(H,y2+int(bh*pad))]


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
            "Pipeline v7 ready | redness=%.2f level=%.2f micro=%.2f blemish=%.0f%%",
            REDNESS_STR, LEVEL_STR, MICRO_STR, BLEMISH_PCT*100,
        )

    def run(self, img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        t_total = time.time()
        H, W = img_bgr.shape[:2]
        result = img_bgr.copy()
        stats = {
            "faces": 0,
            "redness": REDNESS_STR, "level": LEVEL_STR, "micro": MICRO_STR,
        }

        # Face detection
        t0 = time.time()
        faces = self.detector.detect(img_bgr)
        stats["faces"] = len(faces)
        stats["t_detect"] = round(time.time()-t0, 2)

        if not faces:
            logger.info("No faces — skipped")
            return result, stats

        t_retouch = time.time()
        for fi in faces:
            x1,y1,x2,y2 = _pad_bbox(fi["bbox"], H, W, pad=0.30)
            crop = safe_crop(img_bgr, x1, y1, x2, y2)
            if crop.size == 0:
                continue

            fH, fW = crop.shape[:2]
            logger.info("Face: %dx%d at [%d,%d,%d,%d]", fW,fH, x1,y1,x2,y2)

            try:
                # 1. Skin mask
                t_s = time.time()
                skin_raw    = self.parser.skin_mask(crop)
                skin_strict = erode_mask(skin_raw, ksize=3, iters=1)
                skin_soft   = feather_mask(
                    dilate_mask(skin_strict, ksize=5, iters=1),
                    radius=max(8, int(min(fH,fW)*0.012)),
                )
                logger.info("Skin mask: %.2fs", time.time()-t_s)

                # 2. Blemish inpaint (точечно)
                proc = remove_blemishes(crop, skin_strict)

                # 3. Убираем покраснения (ключевой шаг для эталона)
                t_s = time.time()
                proc = remove_redness(proc, skin_soft, REDNESS_STR)
                logger.info("Redness removal: %.2fs", time.time()-t_s)

                # 4. Dodge & Burn (выравниваем яркость зон)
                t_s = time.time()
                proc = level_skin_luminosity(proc, skin_soft, LEVEL_STR)
                logger.info("Luminosity leveling: %.2fs", time.time()-t_s)

                # 5. Micro-contrast boost
                proc = micro_contrast_boost(proc, skin_soft, MICRO_STR)

                # 6. Composite — вставляем только на область лица
                pad_px = max(20, int(min(fH,fW)*0.06))
                comp = np.zeros((fH,fW), np.float32)
                comp[pad_px:-pad_px, pad_px:-pad_px] = 1.0
                comp = feather_mask(comp, radius=pad_px)

                result[y1:y2, x1:x2] = blend_layers(proc, crop, comp)

            except Exception as e:
                logger.exception("Face error: %s", e)

        stats["t_retouch"] = round(time.time()-t_retouch, 2)
        stats["t_total"]   = round(time.time()-t_total, 2)
        logger.info(
            "Done: faces=%d total=%.2fs (detect=%.2fs retouch=%.2fs)",
            stats["faces"], stats["t_total"],
            stats.get("t_detect",0), stats.get("t_retouch",0),
        )
        return result, stats
