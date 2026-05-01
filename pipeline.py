"""
RetouchPipeline v15 — MediaPipe Beauty Retouch

Архитектура по рекомендации айтишника:

1. MediaPipe Face Mesh   → точная маска лица (468 точек)
2. MediaPipe Selfie Seg  → маска тела
3. Исключения: глаза, губы, брови, волосы через landmarks
4. Защита: родинки, тату, пирсинг
5. Обработка ТОЛЬКО по маске:
   - Redness reduction (LAB a-канал)
   - Tone evening (CLAHE по маске)
   - Soft smoothing (BilateralFilter / GuidedFilter)
   - Dodge & Burn (frequency separation)
6. Feathered composite с оригиналом

Настройки .env:
  REDNESS_STRENGTH=0.75
  TONE_STRENGTH=0.65
  SMOOTH_STRENGTH=0.60
  DB_STRENGTH=0.55
  MOLE_PROTECTION=0.90
"""
from __future__ import annotations

import logging, os, time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils import blend, feather, dilate, erode

logger = logging.getLogger(__name__)

# ── Параметры ──────────────────────────────────────────────────────────────────
REDNESS_STRENGTH  = float(os.environ.get("REDNESS_STRENGTH",  "0.75"))
TONE_STRENGTH     = float(os.environ.get("TONE_STRENGTH",     "0.65"))
SMOOTH_STRENGTH   = float(os.environ.get("SMOOTH_STRENGTH",   "0.60"))
DB_STRENGTH       = float(os.environ.get("DB_STRENGTH",       "0.55"))
MOLE_PROTECTION   = float(os.environ.get("MOLE_PROTECTION",   "0.90"))
MAX_PROC_SIZE     = int(os.environ.get("MAX_PROC_SIZE",        "2048"))


# ══════════════════════════════════════════════════════════════════════════════
# MediaPipe Segmentation
# ══════════════════════════════════════════════════════════════════════════════

class MediaPipeSegmenter:
    """
    Использует MediaPipe Face Mesh + Selfie Segmentation
    для точной маски кожи лица и тела.
    """
    def __init__(self):
        self._face_mesh  = None
        self._selfie_seg = None
        self._load()

    def _load(self):
        try:
            import mediapipe as mp

            # Face Mesh — 468 точек лица
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=4,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )

            # Selfie Segmentation — маска тела
            self._selfie_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1  # 1 = более точная модель
            )
            logger.info("MediaPipe: Face Mesh + Selfie Segmentation loaded")

        except ImportError:
            logger.error("mediapipe not installed: pip install mediapipe")
        except Exception as e:
            logger.error("MediaPipe load failed: %s", e)

    def get_face_skin_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Точная маска кожи лица через MediaPipe Face Mesh.
        Исключает глаза, губы, брови, зубы.
        Возвращает float32 [0..1].
        """
        if self._face_mesh is None:
            return self._fallback_face_mask(img_bgr)

        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            result = self._face_mesh.process(img_rgb)
        except Exception as e:
            logger.warning("FaceMesh failed: %s", e)
            return self._fallback_face_mask(img_bgr)

        if not result.multi_face_landmarks:
            logger.info("No faces in FaceMesh")
            return np.zeros((H, W), np.float32)

        import mediapipe as mp

        # Индексы точек для ИСКЛЮЧЕНИЯ из маски кожи
        # (глаза, брови, губы, зубы)
        LEFT_EYE   = list(mp.solutions.face_mesh.FACEMESH_LEFT_EYE)
        RIGHT_EYE  = list(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)
        LEFT_BROW  = list(mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW)
        RIGHT_BROW = list(mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW)
        LIPS       = list(mp.solutions.face_mesh.FACEMESH_LIPS)

        # Собираем все индексы исключений
        exclude_indices = set()
        for conn_list in [LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, LIPS]:
            for conn in conn_list:
                exclude_indices.add(conn[0])
                exclude_indices.add(conn[1])

        # Контур лица
        FACE_OVAL = list(mp.solutions.face_mesh.FACEMESH_FACE_OVAL)
        oval_indices = set()
        for conn in FACE_OVAL:
            oval_indices.add(conn[0])
            oval_indices.add(conn[1])

        face_mask = np.zeros((H, W), np.uint8)

        for face_lm in result.multi_face_landmarks:
            pts = face_lm.landmark

            # 1. Рисуем контур лица (заполненный полигон)
            oval_pts = np.array([
                [int(pts[i].x * W), int(pts[i].y * H)]
                for i in sorted(oval_indices)
            ], dtype=np.int32)
            if len(oval_pts) > 2:
                cv2.fillPoly(face_mask, [oval_pts], 255)

            # 2. Исключаем зоны глаз, бровей, губ
            for name, conn_list in [
                ("left_eye", LEFT_EYE), ("right_eye", RIGHT_EYE),
                ("left_brow", LEFT_BROW), ("right_brow", RIGHT_BROW),
                ("lips", LIPS),
            ]:
                ex_indices = set()
                for conn in conn_list:
                    ex_indices.add(conn[0])
                    ex_indices.add(conn[1])
                ex_pts = np.array([
                    [int(pts[i].x * W), int(pts[i].y * H)]
                    for i in sorted(ex_indices)
                ], dtype=np.int32)
                if len(ex_pts) > 2:
                    # Немного расширяем зону исключения
                    hull = cv2.convexHull(ex_pts)
                    cv2.fillPoly(face_mask, [hull], 0)

        return face_mask.astype(np.float32) / 255.0

    def get_body_skin_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Маска тела через MediaPipe Selfie Segmentation.
        Возвращает float32 [0..1].
        """
        if self._selfie_seg is None:
            return self._fallback_body_mask(img_bgr)

        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            result = self._selfie_seg.process(img_rgb)
        except Exception as e:
            logger.warning("SelfieSegmentation failed: %s", e)
            return self._fallback_body_mask(img_bgr)

        if result.segmentation_mask is None:
            return self._fallback_body_mask(img_bgr)

        # segmentation_mask: float [0..1], 1 = человек
        body_mask = result.segmentation_mask  # (H, W) float32

        # Оставляем только кожу (убираем одежду через HSV)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        skin_hsv = cv2.inRange(hsv, np.array([0,15,50]), np.array([30,180,255]))
        skin_ycr = cv2.inRange(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb),
            np.array([0,133,77]), np.array([255,173,127])
        )
        skin_color = cv2.bitwise_and(skin_hsv, skin_ycr).astype(np.float32) / 255.0

        # Тело + кожа
        body_skin = body_mask * skin_color

        # Морфология
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        body_skin_u8 = (body_skin * 255).astype(np.uint8)
        body_skin_u8 = cv2.morphologyEx(body_skin_u8, cv2.MORPH_OPEN, k)
        body_skin_u8 = cv2.morphologyEx(body_skin_u8, cv2.MORPH_CLOSE, k)

        return body_skin_u8.astype(np.float32) / 255.0

    def _fallback_face_mask(self, img: np.ndarray) -> np.ndarray:
        """Fallback: HSV если MediaPipe недоступен."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array([0,15,60]), np.array([25,170,255]))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)).astype(np.float32)/255.

    def _fallback_body_mask(self, img: np.ndarray) -> np.ndarray:
        return self._fallback_face_mask(img)


# ══════════════════════════════════════════════════════════════════════════════
# Mole / Tattoo / Piercing Protection
# ══════════════════════════════════════════════════════════════════════════════

def build_protection_mask(img: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """
    Детектирует родинки, тату, пирсинг — тёмные элементы с чёткими краями.
    Возвращает float32 [0..1]: 1 = защищать, 0 = можно обрабатывать.
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Локальный контраст
    r = max(21, int(min(H,W)*0.04)); r += r%2==0
    local_mean = cv2.GaussianBlur(gray, (r,r), 0)
    dark_diff   = local_mean - gray

    # Чёткость краёв (Laplacian)
    er = max(5, int(min(H,W)*0.01)); er += er%2==0
    edges = np.abs(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_32F))
    local_e = cv2.GaussianBlur(edges, (er*2+1,er*2+1), 0)

    sp = skin > 0.3
    if sp.sum() < 100:
        return np.zeros((H,W), np.float32)

    dt = max(float(np.percentile(dark_diff[sp], 88)), 6.)
    et = max(float(np.percentile(local_e[sp],   82)), 2.)

    protect = ((dark_diff > dt) & (local_e > et) & sp).astype(np.uint8) * 255
    protect = cv2.morphologyEx(protect, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))

    # Компоненты — фильтруем по размеру
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(protect, 8)
    mask = np.zeros((H,W), np.uint8)
    max_area = int(H * W * 0.03)   # не более 3% изображения
    for i in range(1, n):
        ar = stats[i, cv2.CC_STAT_AREA]
        mw = stats[i, cv2.CC_STAT_WIDTH]
        mh = stats[i, cv2.CC_STAT_HEIGHT]
        if ar < 3 or ar > max_area: continue
        mask[lbl==i] = 255

    # Расширяем и сглаживаем
    ex = max(4, int(min(H,W)*0.01))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ex*2+1,ex*2+1)))
    n_elem = int((mask>127).sum() / max(1, np.pi*ex**2))
    logger.info("Protection mask: ~%d elements", n_elem)
    return feather(mask.astype(np.float32)/255., r=max(3,ex))


# ══════════════════════════════════════════════════════════════════════════════
# Processing Functions
# ══════════════════════════════════════════════════════════════════════════════

def reduce_redness(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Нейтрализует красноту в LAB a-канале только по маске кожи.
    Не трогает родинки/тату (они в protect_mask уже исключены из mask).
    """
    if REDNESS_STRENGTH < 0.01: return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    m = mask > 0.2
    if m.sum() < 100: return img
    a_mean = float(lab[:,:,1][m].mean())
    correction = (a_mean - lab[:,:,1]) * mask * REDNESS_STRENGTH
    lab[:,:,1] = np.clip(lab[:,:,1] + correction, 0, 255)
    logger.info("Redness: a_mean=%.1f strength=%.2f", a_mean, REDNESS_STRENGTH)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


def even_skin_tone(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Выравнивает тон кожи через CLAHE на L-канале + плавное выравнивание.
    Применяется ТОЛЬКО по маске кожи.
    """
    if TONE_STRENGTH < 0.01: return img
    H, W = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:,:,0]
    m = mask > 0.2
    if m.sum() < 100: return img

    # CLAHE с малым clip limit — мягкое выравнивание
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    L_u8  = np.clip(L, 0, 255).astype(np.uint8)
    L_eq  = clahe.apply(L_u8).astype(np.float32)

    # Применяем только на коже
    L_new = L * (1. - mask * TONE_STRENGTH) + L_eq * (mask * TONE_STRENGTH)
    lab[:,:,0] = np.clip(L_new, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


def soft_smooth(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Мягкое сглаживание через BilateralFilter + Guided Filter.
    Сохраняет текстуру и края.
    Применяется только по маске.
    """
    if SMOOTH_STRENGTH < 0.01: return img
    H, W = img.shape[:2]

    d = max(9, int(min(H,W) * 0.012))
    if d % 2 == 0: d += 1

    # Bilateral — сохраняет края
    smoothed = cv2.bilateralFilter(img, d=d, sigmaColor=30, sigmaSpace=30)

    # Guided filter (если доступен)
    try:
        import cv2.ximgproc
        r_gf = max(8, int(min(H,W)*0.01))
        guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for ch in range(3):
            smoothed[:,:,ch] = cv2.ximgproc.guidedFilter(
                guide=guide,
                src=smoothed[:,:,ch],
                radius=r_gf,
                eps=100,
            )
    except Exception:
        pass  # Bilateral уже достаточно

    # Blend только на коже
    result = blend(smoothed, img, mask * SMOOTH_STRENGTH)
    return result


def dodge_and_burn(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Frequency Separation Dodge & Burn.
    Low freq: выравниваем свет/тень
    High freq: текстура — 100% нетронута
    """
    if DB_STRENGTH < 0.01: return img
    H, W = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:,:,0]
    m = mask > 0.2
    if m.sum() < 100: return img

    # Coarse D&B
    r_c = max(55, int(min(H,W)*0.11)); r_c += r_c%2==0
    k_c = r_c*2+1
    L_low_c = cv2.GaussianBlur(L, (k_c,k_c), r_c/2.)
    L_high_c = L - L_low_c
    L_mean_c = float(L_low_c[m].mean())
    L_low_c_new = L_low_c + (L_mean_c - L_low_c) * mask * DB_STRENGTH

    # Mid D&B
    r_m = max(21, int(min(H,W)*0.045)); r_m += r_m%2==0
    k_m = r_m*2+1
    L_low_m = cv2.GaussianBlur(L, (k_m,k_m), r_m/2.)
    L_mean_m = float(L_low_m[m].mean())
    L_mid_corr = (L_mean_m - L_low_m) * mask * DB_STRENGTH * 0.45

    # Рекомбинируем — HIGH_FREQ нетронут
    L_result = L_low_c_new + L_mid_corr + L_high_c
    lab[:,:,0] = np.clip(L_result, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


def add_warmth(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Лёгкий тёплый персиковый тон на коже."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab[:,:,2] = np.clip(lab[:,:,2] + mask * 3.5, 0, 255)  # тепло
    lab[:,:,1] = np.clip(lab[:,:,1] - mask * 1.5, 0, 255)  # убираем чуть красноты
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# Scale helpers
# ══════════════════════════════════════════════════════════════════════════════

def _scale_down(img, mx):
    H,W=img.shape[:2]; s=mx/max(H,W)
    if s>=1.: return img,1.
    return cv2.resize(img,(int(W*s),int(H*s)),cv2.INTER_AREA),s

def _scale_mask_up(mask, H, W):
    if mask.shape[:2]==(H,W): return mask
    return cv2.resize(mask,(W,H),cv2.INTER_LINEAR)


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class RetouchPipeline:
    def __init__(self):
        self.segmenter: Optional[MediaPipeSegmenter] = None

    def load_models(self):
        logger.info("Loading MediaPipe segmenter…")
        self.segmenter = MediaPipeSegmenter()
        logger.info(
            "Pipeline v15 | redness=%.2f tone=%.2f smooth=%.2f db=%.2f mole=%.2f",
            REDNESS_STRENGTH, TONE_STRENGTH, SMOOTH_STRENGTH, DB_STRENGTH, MOLE_PROTECTION,
        )

    def run(self, img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Полный pipeline beauty retouch с MediaPipe масками.
        Обрабатывает лицо И тело.
        """
        t0 = time.time()
        origH, origW = img_bgr.shape[:2]
        stats = {"mediapipe": True}

        # Для скорости — работаем на уменьшенной копии для масок
        proc_img, scale = _scale_down(img_bgr, MAX_PROC_SIZE)
        pH, pW = proc_img.shape[:2]

        logger.info("Processing at %dx%d (scale=%.2f)", pW, pH, scale)

        # ── 1. Получаем маски ──────────────────────────────────────────────
        t_mask = time.time()

        face_mask = self.segmenter.get_face_skin_mask(proc_img)
        body_mask = self.segmenter.get_body_skin_mask(proc_img)

        # Объединяем лицо + тело
        skin_mask = np.clip(face_mask + body_mask, 0., 1.)

        # Защитная маска (родинки, тату, пирсинг)
        protect_mask = build_protection_mask(proc_img, skin_mask)

        # Итоговая маска для обработки = кожа минус защита
        final_mask = skin_mask * (1. - protect_mask * MOLE_PROTECTION)

        # Сглаживаем края маски
        final_mask_soft = feather(
            dilate(final_mask, k=5, n=1),
            r=max(8, int(min(pH,pW)*0.01))
        )

        skin_px = (final_mask > 0.3).sum()
        logger.info(
            "Masks: face=%.1f%% body=%.1f%% total=%d px in %.2fs",
            face_mask.mean()*100, body_mask.mean()*100,
            skin_px, time.time()-t_mask,
        )

        if skin_px < 1000:
            logger.warning("Skin mask too small — check MediaPipe")
            return img_bgr.copy(), stats

        # ── 2. Обрабатываем на proc_img ────────────────────────────────────
        t_ret = time.time()
        proc = proc_img.copy()

        t1=time.time(); proc = reduce_redness(proc, final_mask_soft)
        logger.info("Redness: %.2fs", time.time()-t1)

        t1=time.time(); proc = even_skin_tone(proc, final_mask_soft)
        logger.info("Tone: %.2fs", time.time()-t1)

        t1=time.time(); proc = soft_smooth(proc, final_mask_soft)
        logger.info("Smooth: %.2fs", time.time()-t1)

        t1=time.time(); proc = dodge_and_burn(proc, final_mask_soft)
        logger.info("D&B: %.2fs", time.time()-t1)

        proc = add_warmth(proc, final_mask_soft)

        logger.info("Retouch total: %.2fs", time.time()-t_ret)

        # ── 3. Масштабируем обратно к оригинальному размеру ───────────────
        if scale < 1.0:
            proc = cv2.resize(proc, (origW, origH), interpolation=cv2.INTER_LANCZOS4)
            final_mask_full = _scale_mask_up(final_mask_soft, origH, origW)
        else:
            final_mask_full = final_mask_soft

        # ── 4. Финальный blend с оригиналом ───────────────────────────────
        result = blend(proc, img_bgr, final_mask_full)

        stats["t_total"] = round(time.time()-t0, 2)
        stats["skin_coverage"] = round(float(skin_px)/(pH*pW)*100, 1)
        logger.info(
            "Done: skin=%.1f%% total=%.2fs",
            stats["skin_coverage"], stats["t_total"],
        )
        return result, stats
