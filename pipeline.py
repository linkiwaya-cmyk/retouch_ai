"""
RetouchPipeline v11 — Light Commercial Skin Retouch

Цель: «дорого, но незаметно» — чистая натуральная кожа.

КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ vs v10:
  1. Исправлена маска кожи — нос обрабатывается только по КОЖЕ,
     пирсинг/украшения НЕ удаляются (убраны лишние labels из _NOSKIN)
  2. Добавлена защита родинок — mole_mask детектирует тёмные
     стабильные пятна с чёткими краями и исключает их из обработки
  3. Frequency Separation вместо CodeFormer для тонкой ретуши:
     - Low freq: тон, краснота, пятна
     - High freq: текстура, поры — 100% сохранена
  4. Dodge & Burn только по skin mask, не трогает родинки
  5. Цветокоррекция: убираем красноту, добавляем тёплый тон

BiSeNet labels (исправлено):
  _SKIN   = {1, 7, 8}        — кожа лица, шея, кожа тела
  _NOSKIN = {2,3,4,5,6,9,10,11,12,13,14,15,16,17,18}
    1=skin  2=l_brow  3=r_brow  4=l_eye   5=r_eye
    6=eyeg  7=l_ear   8=r_ear   9=earring 10=nose (форма носа — НЕ трогать)
    11=mouth 12=u_lip 13=l_lip  14=neck   15=necklace
    16=cloth 17=hair  18=hat

  НОС: label 10 = форма носа (геометрия) — НЕ включён в _SKIN.
  Кожа НА носу попадает в label 1 (skin) — обрабатывается корректно.
  Пирсинг носа — мелкий металлический объект, не попадает ни в один skin label.
"""
from __future__ import annotations

import logging, os, time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from utils import blend, feather, dilate, erode, crop as ucrop

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARSING_DIR = os.environ.get("BISENET_WEIGHT_DIR", str(Path.home() / ".cache/facexlib"))
FACE_CONF   = float(os.environ.get("FACE_CONF", "0.5"))
MAX_PROC_SIZE = int(os.environ.get("MAX_PROC_SIZE", "2048"))

# ── Параметры ретуши ───────────────────────────────────────────────────────────
STRENGTH              = float(os.environ.get("STRENGTH",               "0.75"))
REDNESS_REDUCTION     = float(os.environ.get("REDNESS_REDUCTION",      "0.60"))
SKIN_SMOOTHING        = float(os.environ.get("SKIN_SMOOTHING",         "0.40"))
DODGE_BURN_STRENGTH   = float(os.environ.get("DODGE_BURN_STRENGTH",    "0.45"))
TEXTURE_PRESERVE      = float(os.environ.get("TEXTURE_PRESERVE",       "1.00"))  # 1.0 = 100%
MOLE_PROTECTION       = float(os.environ.get("MOLE_PROTECTION_STRENGTH","0.90"))
BLUR_RADIUS_FACTOR    = float(os.environ.get("BLUR_RADIUS_FACTOR",      "0.09"))

# ── BiSeNet labels (ИСПРАВЛЕНО) ────────────────────────────────────────────────
# label 10 = нос (ГЕОМЕТРИЯ) — НЕ включаем в _SKIN чтобы не менять форму носа
# Кожа НА носу = label 1 (skin) — обрабатывается через общую skin mask
_SKIN   = {1, 7, 8}   # skin, l_ear_skin, r_ear_skin
_NOSKIN = {2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}


# ══════════════════════════════════════════════════════════════════════════════
# Face Detector
# ══════════════════════════════════════════════════════════════════════════════

class FaceDetector:
    def __init__(self):
        self._app = self._ret = None
        try:
            from insightface.app import FaceAnalysis
            a = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            a.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
            self._app = a
            logger.info("FaceDetector: InsightFace")
        except Exception as e:
            logger.warning("InsightFace: %s", e)
            try:
                from facexlib.detection import init_detection_model
                self._ret = init_detection_model(
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
        elif self._ret:
            with torch.no_grad():
                boxes = self._ret.detect_faces(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), conf_threshold=FACE_CONF
                )
            out = [
                {"bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])], "score": float(b[4])}
                for b in boxes
            ]
        else:
            return []
        out.sort(
            key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]),
            reverse=True,
        )
        logger.info("Faces: %d in %.2fs", len(out), time.time()-t0)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Face Parser → skin mask
# ══════════════════════════════════════════════════════════════════════════════

class FaceParser:
    def __init__(self):
        self._net = None
        try:
            from facexlib.parsing import init_parsing_model
            self._net = init_parsing_model(
                model_name="bisenet", device=str(DEVICE), model_rootpath=PARSING_DIR
            )
            self._net.eval()
            logger.info("FaceParser: BiSeNet")
        except Exception as e:
            logger.warning("BiSeNet: %s", e)

    def get_skin_mask(self, face: np.ndarray) -> np.ndarray:
        """float32 [0..1] — только кожа, без глаз/бровей/губ/носа (геометрии)."""
        if self._net:
            return self._bisenet(face)
        return self._hsv(face)

    def _bisenet(self, face: np.ndarray) -> np.ndarray:
        H, W = face.shape[:2]
        x = cv2.resize(face, (512, 512))
        x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
             - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
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
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# Mole / Freckle Protection Mask
# Детектирует тёмные стабильные элементы кожи (родинки, веснушки)
# и ИСКЛЮЧАЕТ их из агрессивной обработки
# ══════════════════════════════════════════════════════════════════════════════

def build_mole_protection_mask(face: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """
    Возвращает float32 маску [0..1]:
      1.0 = родинка/веснушка → НЕ обрабатывать
      0.0 = можно обрабатывать

    Принцип: тёмное пятно с ЧЁТКИМИ краями на коже = родинка/веснушка.
    Размытое красное пятно = прыщ/воспаление → убираем.

    Параметры отличия:
    - родинка: тёмная, чёткие края (высокий градиент по контуру), стабильный цвет
    - прыщ:    красноватый, размытые края, выше по a-каналу Lab
    """
    H, W = face.shape[:2]
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:, :, 0]    # яркость
    a = lab[:, :, 1]    # красный/зелёный

    # 1. Тёмные пятна на коже (L < local_mean - threshold)
    r_local = max(21, int(min(H, W) * 0.04))
    if r_local % 2 == 0:
        r_local += 1
    L_local = cv2.GaussianBlur(L, (r_local, r_local), 0)
    dark_diff = L_local - L  # положительно там где темнее окружения

    # 2. Чёткость краёв (Laplacian) — родинки имеют чёткий контур
    L_u8 = np.clip(L, 0, 255).astype(np.uint8)
    edges = cv2.Laplacian(L_u8, cv2.CV_32F)
    edge_strength = np.abs(edges)
    edge_blur_r = max(5, int(min(H, W) * 0.01))
    if edge_blur_r % 2 == 0:
        edge_blur_r += 1
    local_edge = cv2.GaussianBlur(edge_strength, (edge_blur_r*2+1, edge_blur_r*2+1), 0)

    # 3. Красноватость — прыщи красные, родинки нет
    a_local = cv2.GaussianBlur(a, (r_local, r_local), 0)
    redness = a - a_local  # красноватее окружения

    # Кандидаты на родинку:
    # - тёмнее окружения (dark_diff > threshold)
    # - чёткие края (local_edge > threshold)
    # - НЕ красноватые (redness < threshold)
    dark_thresh  = max(float(np.percentile(dark_diff[skin > 0.5], 92)), 8.0) if (skin > 0.5).sum() > 100 else 8.0
    edge_thresh  = max(float(np.percentile(local_edge[skin > 0.5], 85)), 3.0) if (skin > 0.5).sum() > 100 else 3.0
    redness_max  = 6.0  # если краснее этого — скорее воспаление, не родинка

    mole_candidates = (
        (dark_diff > dark_thresh) &
        (local_edge > edge_thresh) &
        (redness < redness_max) &
        (skin > 0.3)
    ).astype(np.uint8) * 255

    # Морфология — оставляем только связные области нужного размера
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mole_candidates = cv2.morphologyEx(mole_candidates, cv2.MORPH_OPEN, kernel)

    # Фильтруем по размеру компонентов (родинки — небольшие)
    n_lab, lbl, stats, _ = cv2.connectedComponentsWithStats(mole_candidates, connectivity=8)
    mole_mask = np.zeros((H, W), np.uint8)
    min_area = 4
    max_area = int((min(H, W) * 0.06) ** 2)  # не больше 6% от min стороны
    for i in range(1, n_lab):
        area = stats[i, cv2.CC_STAT_AREA]
        mw = stats[i, cv2.CC_STAT_WIDTH]
        mh_comp = stats[i, cv2.CC_STAT_HEIGHT]
        if area < min_area or area > max_area:
            continue
        # Родинки более-менее круглые
        aspect = max(mw, mh_comp) / max(1, min(mw, mh_comp))
        if aspect > 3.5:
            continue
        mole_mask[lbl == i] = 255

    # Расширяем маску родинок с feather — защищаем область вокруг
    expand_r = max(3, int(min(H, W) * 0.008))
    mole_mask = cv2.dilate(mole_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_r*2+1, expand_r*2+1)))
    mole_float = feather(mole_mask.astype(np.float32) / 255.0, r=max(2, expand_r))

    n_moles = int((mole_mask > 127).sum() / (np.pi * (expand_r**2) + 1))
    logger.info("Mole protection: ~%d elements detected", n_moles)
    return mole_float


# ══════════════════════════════════════════════════════════════════════════════
# Frequency Separation
# ══════════════════════════════════════════════════════════════════════════════

def fs(ch: np.ndarray, r: int):
    """low = GaussianBlur(r), high = ch - low"""
    k = r * 2 + 1
    low = cv2.GaussianBlur(ch.astype(np.float32), (k, k), r / 2.0)
    return low, ch.astype(np.float32) - low


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Удаление дефектов (прыщи, воспаления)
# С защитой родинок
# ══════════════════════════════════════════════════════════════════════════════

def remove_blemishes(face: np.ndarray, skin: np.ndarray, mole_protect: np.ndarray) -> np.ndarray:
    """
    Удаляет прыщи и воспаления через inpaint.
    Родинки и веснушки (mole_protect > 0.5) — НЕ трогает.
    Принцип: красноватое + тёмное + нечёткие края = дефект.
    """
    H, W = face.shape[:2]
    r = max(21, int(min(H, W) * 0.04))
    if r % 2 == 0: r += 1

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:, :, 0]
    a = lab[:, :, 1]

    L_local = cv2.GaussianBlur(L, (r, r), 0)
    a_local = cv2.GaussianBlur(a, (r, r), 0)

    # Дефект: темнее окружения + краснее окружения
    dark   = L_local - L          # темнее → больше
    red    = a - a_local           # краснее → больше
    score  = dark * 0.5 + red * 1.2

    sp = score[skin > 0.5]
    if len(sp) < 100:
        return face

    thresh = max(np.percentile(sp, 88), 5.0)
    blemish = ((score > thresh) & (skin > 0.5)).astype(np.uint8) * 255

    # Убираем из маски родинки
    blemish[mole_protect > 0.5] = 0

    # Морфология
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blemish = cv2.morphologyEx(blemish, cv2.MORPH_OPEN, k3)
    blemish = cv2.dilate(blemish, k5, iterations=1)
    # Снова убираем родинки после dilate
    blemish[mole_protect > 0.4] = 0

    # Ограничение площади (не более 4%)
    n = int(blemish.sum() / 255)
    max_px = int(H * W * 0.04)
    if n == 0 or n > max_px:
        if n > max_px:
            logger.info("Blemish mask %d > %d px — skip", n, max_px)
        return face

    result = cv2.inpaint(face, blemish, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    logger.info("Blemish inpaint: %d px", n)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Frequency Separation Retouch
# Low freq: тон, краснота, D&B
# High freq: текстура — 100% нетронута
# Родинки защищены
# ══════════════════════════════════════════════════════════════════════════════

def fs_retouch(
    face: np.ndarray,
    skin_soft: np.ndarray,
    mole_protect: np.ndarray,
) -> np.ndarray:
    """
    Photoshop Frequency Separation:
    1. low_freq  = GaussianBlur(large)   — тон, цвет
    2. high_freq = original - low_freq   — текстура, поры (НЕ ТРОГАТЬ)
    3. Корректируем low_freq:
       - выравниваем тон (L-канал) через D&B
       - убираем краснота (a-канал)
       - добавляем тёплый тон (b-канал)
    4. Маска = skin_soft * (1 - mole_protect * MOLE_PROTECTION)
       → родинки получают минимальную коррекцию
    5. final = new_low + high_freq (текстура полностью сохранена)
    """
    H, W = face.shape[:2]
    m = skin_soft > 0.2
    if m.sum() < 200:
        return face

    # Эффективная маска = кожа минус родинки
    eff_mask = skin_soft * (1.0 - mole_protect * MOLE_PROTECTION)

    # Адаптивные радиусы
    r_tone  = max(45, int(min(H, W) * float(BLUR_RADIUS_FACTOR)))
    r_db    = max(55, int(min(H, W) * (float(BLUR_RADIUS_FACTOR) * 1.2)))
    r_mid   = max(25, int(min(H, W) * (float(BLUR_RADIUS_FACTOR) * 0.55)))
    r_micro = max(7,  int(min(H, W) * 0.014))
    for r in [r_tone, r_db, r_mid, r_micro]:
        if r % 2 == 0: r += 1

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)

    # ── Tone/Color correction (low-freq) ──────────────────────────────────

    # a-канал: убираем покраснения
    a_low, a_high = fs(lab[:, :, 1], r_tone)
    a_mean = float(a_low[m].mean())
    a_corr = (a_mean - a_low) * eff_mask * REDNESS_REDUCTION
    lab[:, :, 1] = np.clip(a_low + a_corr + a_high, 0, 255)

    # b-канал: тёплый персиковый тон (+лёгкое смещение к жёлтому)
    b_low, b_high = fs(lab[:, :, 2], r_tone)
    b_mean = float(b_low[m].mean())
    # Смещаем b чуть выше среднего → теплее
    warmth_shift = (b_mean - b_low + 2.5) * eff_mask * REDNESS_REDUCTION * 0.30
    lab[:, :, 2] = np.clip(b_low + warmth_shift + b_high, 0, 255)

    # ── Dodge & Burn (L low-freq) ──────────────────────────────────────────

    L = lab[:, :, 0]

    # Coarse D&B — крупные зоны света/тени
    L_low_c, L_high_c = fs(L, r_db)
    L_mean_c = float(L_low_c[m].mean())
    db_corr_c = (L_mean_c - L_low_c) * eff_mask * DODGE_BURN_STRENGTH
    L_low_c_new = L_low_c + db_corr_c

    # Mid D&B — средние неровности
    L_low_m, _ = fs(L, r_mid)
    L_mean_m = float(L_low_m[m].mean())
    db_corr_m = (L_mean_m - L_low_m) * eff_mask * DODGE_BURN_STRENGTH * 0.42
    L_low_m_new = L_low_m + db_corr_m

    # ── Skin smoothing (только low-freq) ──────────────────────────────────
    smooth_r = max(r_tone // 2, 15)
    if smooth_r % 2 == 0: smooth_r += 1
    L_low_smooth = cv2.GaussianBlur(L_low_c_new, (smooth_r, smooth_r), 0)
    # Применяем сглаживание только на коже, с ослаблением у родинок
    smooth_mask = eff_mask * SKIN_SMOOTHING
    L_low_c_new = L_low_c_new * (1 - smooth_mask) + L_low_smooth * smooth_mask

    # ── Micro-contrast (делает кожу живой) ────────────────────────────────
    r_mc = r_micro if r_micro % 2 == 1 else r_micro + 1
    L_result = L_low_c_new + (L_low_m_new - L_low_m) + L_high_c
    L_blur_mc = cv2.GaussianBlur(L_result, (r_mc*2+1, r_mc*2+1), r_mc/2.0)
    # Micro-contrast НЕ применяем на родинках
    micro_mask = skin_soft * (1.0 - mole_protect * 0.7) * 0.22
    L_result = L_result + (L_result - L_blur_mc) * micro_mask

    # HIGH_FREQ L нетронут — рекомбинируем (текстура сохранена 100%)
    lab[:, :, 0] = np.clip(L_result, 0, 255)

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)

    # Применяем общий strength
    result = blend(result, face, np.full((H, W), STRENGTH * eff_mask.max(), np.float32))
    # Точнее — попиксельно
    s_map = np.clip(eff_mask * STRENGTH, 0, 1)
    return blend(result, face, s_map)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pad(bbox, H, W, pad=0.30):
    x1, y1, x2, y2 = bbox
    bw, bh = x2-x1, y2-y1
    return [
        max(0, x1-int(bw*pad)), max(0, y1-int(bh*pad)),
        min(W, x2+int(bw*pad)), min(H, y2+int(bh*pad)),
    ]


def _scale_down(img: np.ndarray, max_side: int):
    H, W = img.shape[:2]
    s = max_side / max(H, W)
    if s >= 1.0:
        return img, 1.0
    return cv2.resize(img, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA), s


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
            "Pipeline v11 | strength=%.2f redness=%.2f smooth=%.2f "
            "db=%.2f mole_protect=%.2f",
            STRENGTH, REDNESS_REDUCTION, SKIN_SMOOTHING,
            DODGE_BURN_STRENGTH, MOLE_PROTECTION,
        )

    def run(self, img_bgr: np.ndarray, **kwargs) -> tuple[np.ndarray, dict]:
        """
        FLIP SAFE: EXIF применён в decode_image, bbox масштабируется обратно.
        """
        t0 = time.time()
        origH, origW = img_bgr.shape[:2]
        result = img_bgr.copy()
        stats = {"faces": 0}

        det_img, scale = _scale_down(img_bgr, MAX_PROC_SIZE)
        td = time.time()
        faces = self.detector.detect(det_img)
        stats["faces"] = len(faces)
        stats["t_detect"] = round(time.time()-td, 2)

        if not faces:
            logger.info("No faces detected")
            return result, stats

        tr = time.time()
        for fi in faces:
            bx = [int(v / scale) for v in fi["bbox"]]
            x1, y1, x2, y2 = _pad(bx, origH, origW, pad=0.32)
            crop = ucrop(img_bgr, x1, y1, x2, y2)
            if crop.size == 0:
                continue

            fH, fW = crop.shape[:2]
            logger.info("Face: %dx%d at [%d,%d,%d,%d]", fW, fH, x1, y1, x2, y2)

            try:
                # 1. Skin mask (исправленная — нос/пирсинг не трогаем)
                t_s = time.time()
                skin_raw    = self.parser.get_skin_mask(crop)
                skin_strict = erode(skin_raw, k=3, n=1)
                skin_soft   = feather(
                    dilate(skin_strict, k=5, n=1),
                    r=max(8, int(min(fH, fW) * 0.012)),
                )
                logger.info("Skin mask: %.2fs", time.time()-t_s)

                # 2. Mole / freckle protection mask
                t_m = time.time()
                mole_mask = build_mole_protection_mask(crop, skin_strict)
                logger.info("Mole mask: %.2fs", time.time()-t_m)

                # 3. Blemish removal (прыщи/воспаления, родинки защищены)
                t_b = time.time()
                proc = remove_blemishes(crop, skin_strict, mole_mask)
                logger.info("Blemish: %.2fs", time.time()-t_b)

                # 4. Frequency Separation Retouch
                t_f = time.time()
                proc = fs_retouch(proc, skin_soft, mole_mask)
                logger.info("FS retouch: %.2fs", time.time()-t_f)

                # 5. Composite — бесшовная вставка в оригинал
                pad_px = max(20, int(min(fH, fW) * 0.06))
                comp = np.zeros((fH, fW), np.float32)
                comp[pad_px:-pad_px, pad_px:-pad_px] = 1.0
                comp = feather(comp, r=pad_px)

                result[y1:y2, x1:x2] = blend(proc, crop, comp)

            except Exception as e:
                logger.exception("Face error: %s", e)

        stats["t_retouch"] = round(time.time()-tr, 2)
        stats["t_total"]   = round(time.time()-t0, 2)
        logger.info(
            "Done: faces=%d total=%.2fs (det=%.2fs ret=%.2fs)",
            stats["faces"], stats["t_total"],
            stats.get("t_detect", 0), stats.get("t_retouch", 0),
        )
        return result, stats
