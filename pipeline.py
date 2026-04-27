"""
RetouchPipeline v4 — Professional Beauty Retouch

ПОЛНОСТЬЮ ПЕРЕПИСАН. Никакого пластика, никакого blur.

Логика как у профессионального ретушёра в Photoshop:

  1. Skin mask (BiSeNet) — только кожа, не трогаем:
     глаза, брови, губы, волосы, фон

  2. Blemish removal — ТОЧЕЧНЫЙ inpaint только дефектов
     (не по всему лицу)

  3. Frequency separation (правильная):
     - low-freq  = Gaussian blur большого радиуса (ТОЛЬКО тон)
     - high-freq = original - low (ТЕКСТУРА, поры — 100%)
     - корректируем ТОЛЬКО low-freq
     - high-freq возвращаем НЕТРОНУТЫМ

  4. Skin tone / color evening
     - выравниваем неровный тон ТОЛЬКО в low-freq
     - работаем в Lab (только L и a каналы)
     - не меняем цвет кожи глобально

  5. Dodge & Burn (правильный)
     - локальная работа светом в low-freq
     - dodge: осветляем тёмные участки кожи
     - burn: затемняем слишком светлые
     - только по skin_mask
     - рекомбинируем с оригинальной текстурой

  6. CodeFormer — МИНИМАЛЬНО (fidelity=0.97, blend=0.05)
     или полностью отключить через CODEFORMER_BLEND=0

  7. Composite с feather — бесшовная вставка в оригинал

Настройки в .env:
  RETOUCH_STRENGTH=medium   # light/medium/strong
  CODEFORMER_BLEND=0.05     # 0 = отключить CodeFormer
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from codeformer_loader import load_codeformer
from utils import (
    blend_layers, feather_mask, dilate_mask, erode_mask, safe_crop
)

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CF_WEIGHT = os.environ.get("CODEFORMER_WEIGHT", str(Path.home() / ".cache/codeformer/codeformer.pth"))
PARSING_DIR = os.environ.get("BISENET_WEIGHT_DIR", str(Path.home() / ".cache/facexlib"))
FACE_CONF = float(os.environ.get("FACE_CONF", "0.5"))

# Пресеты силы ретуши
_PRESETS = {
    #              blemish  smooth  db    tone  cf_blend
    "light":      (0.80,   0.20,  0.10, 0.15,  0.03),
    "medium":     (0.85,   0.30,  0.18, 0.25,  0.05),
    "strong":     (0.88,   0.42,  0.28, 0.38,  0.05),
}
_pname = os.environ.get("RETOUCH_STRENGTH", "medium")
_p = _PRESETS.get(_pname, _PRESETS["medium"])

BLEMISH_PERC   = float(os.environ.get("BLEMISH_PERCENTILE", _p[0]))
SMOOTH_STR     = float(os.environ.get("SKIN_SMOOTH_STRENGTH", _p[1]))
DB_STR         = float(os.environ.get("DB_STRENGTH", _p[2]))
TONE_STR       = float(os.environ.get("TONE_STRENGTH", _p[3]))
CF_BLEND       = float(os.environ.get("CODEFORMER_BLEND", _p[4]))
CF_FIDELITY    = float(os.environ.get("CF_FIDELITY", "0.97"))

_CF_SIZE = 512

# BiSeNet labels
_SKIN  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13}
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
            app = FaceAnalysis(name="buffalo_l",
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
            self._app = app
            logger.info("FaceDetector: InsightFace")
        except Exception as e:
            logger.warning("InsightFace failed (%s), trying RetinaFace", e)
            try:
                from facexlib.detection import init_detection_model
                self._retina = init_detection_model("retinaface_resnet50", half=False, device=str(DEVICE))
                logger.info("FaceDetector: RetinaFace")
            except Exception as e2:
                logger.error("No face detector: %s", e2)

    def detect(self, img: np.ndarray) -> list[dict]:
        if self._app:
            faces = self._app.get(img)
            out = [{"bbox": f.bbox.astype(int).tolist(), "score": float(f.det_score)}
                   for f in faces if f.det_score >= FACE_CONF]
        elif self._retina:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                boxes = self._retina.detect_faces(rgb, conf_threshold=FACE_CONF)
            out = [{"bbox": [int(b[0]),int(b[1]),int(b[2]),int(b[3])], "score": float(b[4])} for b in boxes]
        else:
            return []
        out.sort(key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]), reverse=True)
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
            self._net = init_parsing_model(model_name="bisenet", device=str(DEVICE),
                                           model_rootpath=PARSING_DIR)
            self._net.eval()
            logger.info("FaceParser: BiSeNet OK")
        except Exception as e:
            logger.warning("BiSeNet failed: %s — HSV fallback", e)

    def skin_mask(self, face_bgr: np.ndarray) -> np.ndarray:
        """float32 [0..1], HxW. 1=кожа, 0=не кожа."""
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
            mask[seg == lbl] = 0  # убираем глаза/губы
        return (cv2.resize(mask, (W, H), cv2.INTER_NEAREST).astype(np.float32) / 255.0)

    def _hsv(self, face: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array([0, 15, 60]), np.array([25, 170, 255]))
        return (cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)).astype(np.float32)/255.0)


# ══════════════════════════════════════════════════════════════════════════════
# CORE: Правильная Frequency Separation
# ══════════════════════════════════════════════════════════════════════════════

def frequency_separate(img: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Разделяет изображение на:
    - low_freq:  тон, цвет (Gaussian blur)
    - high_freq: текстура, поры (img - low_freq)

    high_freq содержит ОТРИЦАТЕЛЬНЫЕ значения — это нормально.
    При рекомбинации: result = new_low + high_freq
    """
    k = radius * 2 + 1
    low = cv2.GaussianBlur(img.astype(np.float32), (k, k), radius / 2.0)
    high = img.astype(np.float32) - low
    return low, high


def recombine(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(low + high, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Точечное удаление дефектов (inpaint)
# ══════════════════════════════════════════════════════════════════════════════

def remove_blemishes(face: np.ndarray, skin: np.ndarray, percentile: float = BLEMISH_PERC) -> np.ndarray:
    """
    Детектирует локальные аномалии яркости → inpaint.
    ТОЛЬКО точечные дефекты — не по всему лицу.
    """
    H, W = face.shape[:2]
    # Адаптивный радиус под разрешение
    r = max(25, int(min(H, W) * 0.05))
    if r % 2 == 0: r += 1

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab)
    L = lab[:,:,0].astype(np.float32)

    local_mean = cv2.GaussianBlur(L, (r, r), 0)
    diff = np.abs(L - local_mean)

    skin_pixels = diff[skin > 0.4]
    if len(skin_pixels) < 50:
        return face

    thresh = np.percentile(skin_pixels, percentile * 100)
    thresh = max(thresh, 8.0)  # минимальный порог

    blemish = ((diff > thresh) & (skin > 0.4)).astype(np.uint8) * 255

    # Морфология — убираем шум
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blemish = cv2.morphologyEx(blemish, cv2.MORPH_OPEN, k3)   # убираем точечный шум
    blemish = cv2.dilate(blemish, k5, iterations=1)             # расширяем края дефектов

    n_pixels = int(blemish.sum() / 255)
    if n_pixels == 0:
        return face

    # Ограничиваем — не более 8% площади лица (иначе это не дефект, а особенность)
    max_pixels = int(H * W * 0.08)
    if n_pixels > max_pixels:
        logger.info("Blemish mask too large (%d px), skipping inpaint", n_pixels)
        return face

    result = cv2.inpaint(face, blemish, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    logger.info("Blemish inpaint: %d pixels", n_pixels)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Выравнивание тона кожи (только low-freq, только кожа)
# ══════════════════════════════════════════════════════════════════════════════

def even_skin_tone(face: np.ndarray, skin: np.ndarray, strength: float = TONE_STR) -> np.ndarray:
    """
    Выравнивает неровный тон (покраснения, пятна) через low-freq коррекцию.
    Работает в Lab. Текстура (high-freq) сохраняется полностью.
    """
    if strength < 0.01:
        return face

    H, W = face.shape[:2]
    r_low = max(35, int(min(H, W) * 0.08))
    if r_low % 2 == 0: r_low += 1

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)

    # Разделяем L и a на low+high
    L_low, L_high = frequency_separate(lab[:,:,0], r_low)
    a_low, a_high = frequency_separate(lab[:,:,1], r_low)

    # Глобальное среднее по коже
    m = skin > 0.5
    if m.sum() < 100:
        return face

    L_mean = float(L_low[m].mean())
    a_mean = float(a_low[m].mean())

    # Сдвигаем low-freq к среднему значению — только на коже
    L_low_new = L_low.copy()
    a_low_new = a_low.copy()
    L_low_new[m] = L_low[m] + (L_mean - L_low[m]) * strength
    a_low_new[m] = a_low[m] + (a_mean - a_low[m]) * strength * 0.6  # чуть слабее по цвету

    # Рекомбинируем с оригинальной текстурой
    lab[:,:,0] = np.clip(L_low_new + L_high, 0, 255)
    lab[:,:,1] = np.clip(a_low_new + a_high, 0, 255)

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Профессиональный Dodge & Burn
# ══════════════════════════════════════════════════════════════════════════════

def dodge_and_burn(face: np.ndarray, skin: np.ndarray, strength: float = DB_STR) -> np.ndarray:
    """
    Правильный Dodge & Burn:
    - Разделяем на low-freq и high-freq
    - Dodge & Burn применяем ТОЛЬКО к low-freq (тону)
    - high-freq (текстура, поры) возвращаем НЕТРОНУТОЙ 100%
    - Работает только по skin mask
    - Не трогает глаза, брови, губы
    """
    if strength < 0.01:
        return face

    H, W = face.shape[:2]

    # Радиус для low-freq — крупные зоны света/тени
    r_low = max(41, int(min(H, W) * 0.10))
    if r_low % 2 == 0: r_low += 1

    # Радиус для mid-freq — средние неровности
    r_mid = max(15, int(min(H, W) * 0.035))
    if r_mid % 2 == 0: r_mid += 1

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:,:,0]

    # ── Разделяем на low и high ────────────────────────────────────────────
    L_low, L_high = frequency_separate(L, r_low)

    # ── Dodge & Burn на low-freq ───────────────────────────────────────────
    m = skin > 0.3
    if m.sum() < 100:
        return face

    L_mean = float(L_low[m].mean())

    # Карта коррекции: тёмные участки -> dodge (+), светлые -> burn (-)
    correction = (L_mean - L_low) * strength

    # Применяем только на коже, с мягким feather
    skin_f = feather_mask(skin, radius=max(8, int(min(H, W) * 0.015)))
    L_low_corrected = L_low + correction * skin_f

    # ── Mid-freq Dodge & Burn (более мелкие неровности) ───────────────────
    L_mid, L_mid_high = frequency_separate(L, r_mid)
    L_mid_mean = float(L_mid[m].mean())
    mid_correction = (L_mid_mean - L_mid) * strength * 0.4
    L_mid_corrected = L_mid + mid_correction * skin_f
    # mid high-freq
    L_mid_high_new = L - L_mid_corrected  # обновляем mid high
    # Возвращаем только коррекцию mid-freq
    L_combined_low = L_low_corrected + (L_mid_corrected - L_mid)

    # ── Рекомбинируем: low_corrected + high_original ───────────────────────
    # HIGH-FREQ (текстура) НЕТРОНУТА
    L_result = np.clip(L_combined_low + L_high, 0, 255)

    lab[:,:,0] = L_result
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Мягкое сглаживание цвета кожи (не яркости, не текстуры)
# ══════════════════════════════════════════════════════════════════════════════

def smooth_skin_color(face: np.ndarray, skin: np.ndarray, strength: float = SMOOTH_STR) -> np.ndarray:
    """
    Очень мягкое сглаживание ТОЛЬКО цветового неравенства кожи.
    НЕ размывает текстуру — работает только с low-freq a и b каналами (Lab).
    L канал (яркость/текстура) не трогаем.
    """
    if strength < 0.01:
        return face

    H, W = face.shape[:2]
    r = max(25, int(min(H, W) * 0.06))
    if r % 2 == 0: r += 1

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)

    # Работаем только с a и b (цвет), НЕ с L (яркость/текстура)
    for ch in [1, 2]:
        ch_data = lab[:,:,ch]
        ch_low, ch_high = frequency_separate(ch_data, r)
        # Bilateral сглаживание low-freq цвета
        ch_low_u8 = np.clip(ch_low, 0, 255).astype(np.uint8)
        ch_smooth = cv2.bilateralFilter(ch_low_u8, d=r//2*2+1,
                                        sigmaColor=20, sigmaSpace=20).astype(np.float32)
        # Применяем только на коже
        skin_f = feather_mask(skin, radius=max(6, r//4))
        ch_low_new = ch_low * (1 - skin_f * strength) + ch_smooth * (skin_f * strength)
        # Рекомбинируем с оригинальной текстурой цвета
        lab[:,:,ch] = np.clip(ch_low_new + ch_high, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: CodeFormer — минимально (fidelity=0.97, blend=0.05)
# ══════════════════════════════════════════════════════════════════════════════

def apply_codeformer(
    face: np.ndarray,
    net,
    skin: np.ndarray,
    fidelity: float = CF_FIDELITY,
    blend: float = CF_BLEND,
) -> tuple[np.ndarray, bool]:
    if net is None or blend < 0.01:
        return face, False

    H, W = face.shape[:2]
    inp = cv2.resize(face, (_CF_SIZE, _CF_SIZE), cv2.INTER_LANCZOS4)
    inp_t = torch.from_numpy(
        (cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - 0.5) / 0.5
    ).permute(2,0,1).unsqueeze(0).to(DEVICE)

    try:
        with torch.no_grad():
            out = net(inp_t, w=fidelity, adain=True)
            if isinstance(out, (list, tuple)):
                out = out[0]
    except Exception as e:
        logger.warning("CodeFormer inference failed: %s", e)
        return face, False

    out_np = np.clip(out.squeeze(0).permute(1,2,0).cpu().numpy() * 0.5 + 0.5, 0, 1)
    cf = cv2.cvtColor((out_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cf = cv2.resize(cf, (W, H), cv2.INTER_LANCZOS4)

    # Смешиваем только на коже, очень мягко
    alpha = feather_mask(skin, radius=12) * blend
    result = blend_layers(cf, face, alpha)
    logger.info("CodeFormer applied: fidelity=%.2f blend=%.2f", fidelity, blend)
    return result, True


# ══════════════════════════════════════════════════════════════════════════════
# Face helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pad_bbox(bbox, H, W, pad=0.35):
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
        self.parser: Optional[FaceParser] = None
        self.codeformer = None

    def load_models(self):
        logger.info("Loading FaceDetector…")
        self.detector = FaceDetector()
        logger.info("Loading FaceParser…")
        self.parser = FaceParser()
        logger.info("Loading CodeFormer (fidelity=%.2f blend=%.2f)…", CF_FIDELITY, CF_BLEND)
        self.codeformer = load_codeformer(device=DEVICE)  # None если не удалось
        logger.info(
            "Pipeline ready | strength=%s smooth=%.2f db=%.2f tone=%.2f cf=%.2f",
            _pname, SMOOTH_STR, DB_STR, TONE_STR, CF_BLEND
        )

    def run(self, img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        H, W = img_bgr.shape[:2]
        result = img_bgr.copy()
        stats = {
            "faces": 0, "cf_applied": False,
            "strength": _pname,
            "smooth": SMOOTH_STR, "db": DB_STR, "tone": TONE_STR,
            "cf_blend": CF_BLEND, "cf_fidelity": CF_FIDELITY,
        }

        faces = self.detector.detect(img_bgr)
        stats["faces"] = len(faces)
        logger.info("Faces detected: %d", len(faces))

        if not faces:
            logger.info("No faces — skipping retouch")
            return result, stats

        for face_info in faces:
            x1,y1,x2,y2 = _pad_bbox(face_info["bbox"], H, W, pad=0.35)
            crop = safe_crop(img_bgr, x1, y1, x2, y2)
            if crop.size == 0:
                continue

            fH, fW = crop.shape[:2]
            logger.info("Face crop: %dx%d", fW, fH)

            try:
                # ── 1. Skin mask ───────────────────────────────────────────
                skin = self.parser.skin_mask(crop)

                # Убираем края маски — более точная маска = меньше артефактов
                skin = erode_mask(skin, ksize=3, iters=1)
                skin_soft = feather_mask(
                    dilate_mask(skin, ksize=5, iters=1), radius=10
                )

                # ── 2. Blemish removal ─────────────────────────────────────
                processed = remove_blemishes(crop, skin, percentile=BLEMISH_PERC)

                # ── 3. Skin tone evening ───────────────────────────────────
                processed = even_skin_tone(processed, skin_soft, strength=TONE_STR)

                # ── 4. Dodge & Burn (только low-freq, текстура сохранена) ──
                processed = dodge_and_burn(processed, skin_soft, strength=DB_STR)

                # ── 5. Цветовое сглаживание (только a,b каналы, не L) ─────
                processed = smooth_skin_color(processed, skin_soft, strength=SMOOTH_STR)

                # ── 6. CodeFormer (минимально) ─────────────────────────────
                processed, cf_ok = apply_codeformer(
                    processed, self.codeformer, skin_soft, CF_FIDELITY, CF_BLEND
                )
                if cf_ok:
                    stats["cf_applied"] = True

                # ── 7. Composite — бесшовная вставка ──────────────────────
                pad_px = max(20, int(min(fH, fW) * 0.07))
                comp = np.zeros((fH, fW), np.float32)
                comp[pad_px:-pad_px, pad_px:-pad_px] = 1.0
                comp = feather_mask(comp, radius=pad_px)

                blended = blend_layers(processed, crop, comp)
                result[y1:y2, x1:x2] = blended

            except Exception as e:
                logger.exception("Face retouch failed: %s", e)

        logger.info("Done: faces=%d cf=%s", stats["faces"], stats["cf_applied"])
        return result, stats
