"""
RetouchPipeline v8

Цель: результат максимально близкий к эталону.

Анализ эталона vs исходника:
  1. Убраны покраснения и цветовые пятна на коже — REDNESS_STR=0.70
  2. Выровнена яркость (тёмные участки осветлены) — LEVEL_STR=0.52
  3. Точечно убраны дефекты — inpaint до 4% площади
  4. Текстура кожи (поры) ПОЛНОСТЬЮ сохранена — high-freq не трогаем
  5. Глаза/брови/губы/волосы — строго исключены из маски

FLIP FIX:
  - EXIF ориентация применяется ОДИН РАЗ в utils.decode_image()
  - Кроп вставляется обратно по тем же координатам
  - Никаких дополнительных rotate/flip нигде

CodeFormer: ОТКЛЮЧЁН.
Blur по лицу: ОТСУТСТВУЕТ.
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

# ── Параметры ретуши — УСИЛЕНЫ под эталон ─────────────────────────────────────
REDNESS_STR = float(os.environ.get("REDNESS_STRENGTH", "0.70"))   # убираем покраснения
LEVEL_STR   = float(os.environ.get("LEVEL_STRENGTH",   "0.52"))   # D&B яркость зон
MICRO_STR   = float(os.environ.get("MICRO_STRENGTH",   "0.28"))   # micro-contrast
BLEMISH_PCT = float(os.environ.get("BLEMISH_MAX_PCT",  "0.04"))   # max % inpaint

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
            logger.info("FaceDetector: InsightFace")
        except Exception as e:
            logger.warning("InsightFace: %s → RetinaFace", e)
            try:
                from facexlib.detection import init_detection_model
                self._retina = init_detection_model(
                    "retinaface_resnet50", half=False, device=str(DEVICE)
                )
                logger.info("FaceDetector: RetinaFace")
            except Exception as e2:
                logger.error("No detector: %s", e2)

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
            logger.warning("BiSeNet: %s", e)

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
        for lbl in _SKIN:   mask[seg==lbl] = 255
        for lbl in _NOSKIN: mask[seg==lbl] = 0
        return cv2.resize(mask, (W,H), cv2.INTER_NEAREST).astype(np.float32)/255.0

    def _hsv(self, face: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array([0,15,60]), np.array([25,170,255]))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)).astype(np.float32)/255.0


# ══════════════════════════════════════════════════════════════════════════════
# Frequency Separation
# ══════════════════════════════════════════════════════════════════════════════

def fs(ch: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """low=blur(r), high=ch-low. Recombine: clip(new_low+high, 0,255)"""
    k = r*2+1
    low = cv2.GaussianBlur(ch.astype(np.float32), (k,k), r/2.0)
    return low, ch.astype(np.float32) - low


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Точечный inpaint дефектов
# ══════════════════════════════════════════════════════════════════════════════

def remove_blemishes(face: np.ndarray, skin: np.ndarray) -> np.ndarray:
    H, W = face.shape[:2]
    max_px = int(H * W * BLEMISH_PCT)

    r = max(21, int(min(H,W)*0.04))
    if r%2==0: r+=1

    L = cv2.cvtColor(face, cv2.COLOR_BGR2Lab)[:,:,0].astype(np.float32)
    diff = np.abs(L - cv2.GaussianBlur(L, (r,r), 0))
    skin_px = diff[skin > 0.5]
    if len(skin_px) < 100:
        return face

    thresh = max(np.percentile(skin_px, 90), 10.0)
    mask = ((diff > thresh) & (skin > 0.5)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=1)

    n = int(mask.sum()/255)
    if n == 0 or n > max_px:
        if n > max_px:
            logger.info("Blemish %d px > limit %d — skip", n, max_px)
        return face

    result = cv2.inpaint(face, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    logger.info("Blemish: %d px (%.1f%%)", n, n/H/W*100)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Убираем покраснения и неровный цвет кожи
# ══════════════════════════════════════════════════════════════════════════════

def remove_redness(face: np.ndarray, skin_soft: np.ndarray) -> np.ndarray:
    """
    Выравнивает a и b каналы Lab по low-freq.
    a = красный/зелёный → убираем покраснения
    b = жёлтый/синий → убираем неровный тон
    High-freq цвета сохраняется.
    """
    if REDNESS_STR < 0.01:
        return face

    H, W = face.shape[:2]
    r = max(45, int(min(H,W)*0.09))
    if r%2==0: r+=1

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)
    m = skin_soft > 0.3
    if m.sum() < 200:
        return face

    # a-канал (покраснения)
    a_low, a_high = fs(lab[:,:,1], r)
    a_mean = float(a_low[m].mean())
    lab[:,:,1] = np.clip(a_low + (a_mean - a_low)*skin_soft*REDNESS_STR + a_high, 0, 255)

    # b-канал (тон)
    b_low, b_high = fs(lab[:,:,2], r)
    b_mean = float(b_low[m].mean())
    lab[:,:,2] = np.clip(b_low + (b_mean - b_low)*skin_soft*REDNESS_STR*0.40 + b_high, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Dodge & Burn — выравнивание яркости зон кожи
# ══════════════════════════════════════════════════════════════════════════════

def level_luminosity(face: np.ndarray, skin_soft: np.ndarray) -> np.ndarray:
    """
    Выравнивает яркость крупных зон кожи (Dodge & Burn).
    ТОЛЬКО low-freq L канала.
    High-freq (текстура, поры) возвращается 100% нетронутой.
    """
    if LEVEL_STR < 0.01:
        return face

    H, W = face.shape[:2]
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:,:,0]
    m = skin_soft > 0.2
    if m.sum() < 200:
        return face

    # Coarse: крупные зоны
    r_c = max(55, int(min(H,W)*0.11))
    if r_c%2==0: r_c+=1
    L_low_c, L_high_c = fs(L, r_c)
    L_mean_c = float(L_low_c[m].mean())
    L_low_c_new = L_low_c + (L_mean_c - L_low_c) * skin_soft * LEVEL_STR

    # Mid: средние неровности
    r_m = max(25, int(min(H,W)*0.05))
    if r_m%2==0: r_m+=1
    L_low_m, _ = fs(L, r_m)
    L_mean_m = float(L_low_m[m].mean())
    L_low_m_new = L_low_m + (L_mean_m - L_low_m) * skin_soft * LEVEL_STR * 0.42

    # Рекомбинируем: new_low_coarse + mid_correction + HIGH_FREQ_ORIGINAL
    L_result = L_low_c_new + (L_low_m_new - L_low_m) + L_high_c
    lab[:,:,0] = np.clip(L_result, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Micro-contrast (кожа выглядит более живой)
# ══════════════════════════════════════════════════════════════════════════════

def micro_contrast(face: np.ndarray, skin_soft: np.ndarray) -> np.ndarray:
    """Unsharp mask на mid-freq только по коже."""
    if MICRO_STR < 0.01:
        return face

    H, W = face.shape[:2]
    r = max(7, int(min(H,W)*0.015))
    if r%2==0: r+=1

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:,:,0]
    L_blur = cv2.GaussianBlur(L, (r*2+1, r*2+1), r/2.0)
    lab[:,:,0] = np.clip(L + (L - L_blur) * skin_soft * MICRO_STR, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


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
            "Pipeline v8 | redness=%.2f level=%.2f micro=%.2f blemish=%.0f%%",
            REDNESS_STR, LEVEL_STR, MICRO_STR, BLEMISH_PCT*100,
        )

    def run(self, img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        FLIP SAFE:
        img_bgr уже в правильной ориентации (EXIF применён в decode_image).
        Кроп вставляется по тем же пиксельным координатам → нет flip.
        """
        t0 = time.time()
        H, W = img_bgr.shape[:2]
        result = img_bgr.copy()
        stats = {"faces": 0, "redness": REDNESS_STR, "level": LEVEL_STR}

        # Face detection
        t_d = time.time()
        faces = self.detector.detect(img_bgr)
        stats["faces"] = len(faces)
        stats["t_detect"] = round(time.time()-t_d, 2)

        if not faces:
            logger.info("No faces — skipped")
            return result, stats

        t_r = time.time()
        for fi in faces:
            # Координаты bbox в уже-ориентированном изображении
            x1,y1,x2,y2 = _pad_bbox(fi["bbox"], H, W, pad=0.30)
            crop = safe_crop(img_bgr, x1, y1, x2, y2)
            if crop.size == 0:
                continue

            fH, fW = crop.shape[:2]
            logger.info("Face: %dx%d at [%d,%d,%d,%d]", fW,fH, x1,y1,x2,y2)

            try:
                # 1. Skin mask
                skin_raw  = self.parser.skin_mask(crop)
                skin_strict = erode_mask(skin_raw, ksize=3, iters=1)
                skin_soft   = feather_mask(
                    dilate_mask(skin_strict, ksize=5, iters=1),
                    radius=max(8, int(min(fH,fW)*0.012)),
                )

                # 2. Blemish inpaint (точечно)
                proc = remove_blemishes(crop, skin_strict)

                # 3. Убираем покраснения (главный эффект)
                proc = remove_redness(proc, skin_soft)

                # 4. Dodge & Burn (яркость зон)
                proc = level_luminosity(proc, skin_soft)

                # 5. Micro-contrast
                proc = micro_contrast(proc, skin_soft)

                # 6. Composite — бесшовная вставка
                pad_px = max(20, int(min(fH,fW)*0.06))
                comp = np.zeros((fH,fW), np.float32)
                comp[pad_px:-pad_px, pad_px:-pad_px] = 1.0
                comp = feather_mask(comp, radius=pad_px)

                # Вставляем по ТОМУ ЖЕ bbox — нет flip
                result[y1:y2, x1:x2] = blend_layers(proc, crop, comp)

            except Exception as e:
                logger.exception("Face error: %s", e)

        stats["t_retouch"] = round(time.time()-t_r, 2)
        stats["t_total"]   = round(time.time()-t0, 2)
        logger.info("Done: faces=%d total=%.2fs", stats["faces"], stats["t_total"])
        return result, stats
