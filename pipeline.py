"""
RetouchPipeline v12 — Light Commercial Retouch

ИСПРАВЛЕНИЯ vs v11:
  1. Убран двойной blend в конце fs_retouch — он гасил весь эффект
  2. Усилены параметры под эталон:
       REDNESS_REDUCTION  0.60 → 0.80
       DODGE_BURN_STRENGTH 0.45 → 0.65
       SKIN_SMOOTHING     0.40 → 0.55
       STRENGTH           0.75 → 0.88
  3. fs_retouch теперь возвращает результат напрямую без повторного blend
  4. Composite mask усилена (центр = 1.0, края feather)

BiSeNet labels (правильные):
  _SKIN   = {1, 7, 8}   — кожа лица, ушей
  _NOSKIN = остальные   — глаза, брови, губы, нос (геометрия), украшения
  Кожа НА носу = label 1 → обрабатывается. Форма носа/пирсинг — нет.
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

PARSING_DIR   = os.environ.get("BISENET_WEIGHT_DIR", str(Path.home() / ".cache/facexlib"))
FACE_CONF     = float(os.environ.get("FACE_CONF", "0.5"))
MAX_PROC_SIZE = int(os.environ.get("MAX_PROC_SIZE", "2048"))

# ── Параметры — усилены под эталон ────────────────────────────────────────────
STRENGTH            = float(os.environ.get("STRENGTH",              "0.88"))
REDNESS_REDUCTION   = float(os.environ.get("REDNESS_REDUCTION",    "0.80"))
SKIN_SMOOTHING      = float(os.environ.get("SKIN_SMOOTHING",       "0.55"))
DODGE_BURN_STRENGTH = float(os.environ.get("DODGE_BURN_STRENGTH",  "0.65"))
TEXTURE_PRESERVE    = float(os.environ.get("TEXTURE_PRESERVE",     "1.00"))
MOLE_PROTECTION     = float(os.environ.get("MOLE_PROTECTION_STRENGTH", "0.90"))
BLUR_RADIUS_FACTOR  = float(os.environ.get("BLUR_RADIUS_FACTOR",   "0.09"))

# Только чистая кожа; нос (10), украшения (9) — не трогаем
_SKIN   = {1, 7, 8}
_NOSKIN = {2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}


# ══════════════════════════════════════════════════════════════════════════════
# Face Detector
# ══════════════════════════════════════════════════════════════════════════════

class FaceDetector:
    def __init__(self):
        self._app = self._ret = None
        try:
            from insightface.app import FaceAnalysis
            a = FaceAnalysis(name="buffalo_l",
                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            a.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
            self._app = a
            logger.info("FaceDetector: InsightFace")
        except Exception as e:
            logger.warning("InsightFace: %s", e)
            try:
                from facexlib.detection import init_detection_model
                self._ret = init_detection_model("retinaface_resnet50",
                                                  half=False, device=str(DEVICE))
                logger.info("FaceDetector: RetinaFace")
            except Exception as e2:
                logger.error("No detector: %s", e2)

    def detect(self, img: np.ndarray) -> list[dict]:
        t0 = time.time()
        if self._app:
            faces = self._app.get(img)
            out = [{"bbox": f.bbox.astype(int).tolist(), "score": float(f.det_score)}
                   for f in faces if f.det_score >= FACE_CONF]
        elif self._ret:
            with torch.no_grad():
                boxes = self._ret.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                               conf_threshold=FACE_CONF)
            out = [{"bbox": [int(b[0]),int(b[1]),int(b[2]),int(b[3])], "score": float(b[4])}
                   for b in boxes]
        else:
            return []
        out.sort(key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]), reverse=True)
        logger.info("Faces: %d in %.2fs", len(out), time.time()-t0)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Face Parser
# ══════════════════════════════════════════════════════════════════════════════

class FaceParser:
    def __init__(self):
        self._net = None
        try:
            from facexlib.parsing import init_parsing_model
            self._net = init_parsing_model(model_name="bisenet", device=str(DEVICE),
                                           model_rootpath=PARSING_DIR)
            self._net.eval()
            logger.info("FaceParser: BiSeNet")
        except Exception as e:
            logger.warning("BiSeNet: %s", e)

    def get_skin_mask(self, face: np.ndarray) -> np.ndarray:
        return self._bisenet(face) if self._net else self._hsv(face)

    def _bisenet(self, face: np.ndarray) -> np.ndarray:
        H, W = face.shape[:2]
        x = cv2.resize(face, (512, 512))
        x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
             - [.485,.456,.406]) / [.229,.224,.225]
        t = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            seg = self._net(t)[0].squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
        m = np.zeros((512, 512), np.uint8)
        for l in _SKIN:   m[seg == l] = 255
        for l in _NOSKIN: m[seg == l] = 0
        return cv2.resize(m, (W, H), cv2.INTER_NEAREST).astype(np.float32) / 255.

    def _hsv(self, face: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array([0,15,60]), np.array([25,170,255]))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8)).astype(np.float32)/255.


# ══════════════════════════════════════════════════════════════════════════════
# Mole Protection Mask
# ══════════════════════════════════════════════════════════════════════════════

def build_mole_mask(face: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """
    Детектирует родинки/веснушки: тёмные + чёткие края + не красные.
    Возвращает float32 [0..1]: 1 = родинка → не обрабатывать.
    """
    H, W = face.shape[:2]
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)
    L, a = lab[:,:,0], lab[:,:,1]

    r = max(21, int(min(H,W)*0.04))
    if r % 2 == 0: r += 1

    L_local = cv2.GaussianBlur(L, (r,r), 0)
    a_local = cv2.GaussianBlur(a, (r,r), 0)
    dark_diff = L_local - L       # темнее окружения
    redness   = a - a_local       # краснее окружения

    L_u8    = np.clip(L, 0, 255).astype(np.uint8)
    edges   = np.abs(cv2.Laplacian(L_u8, cv2.CV_32F))
    er      = max(5, int(min(H,W)*0.01))
    if er % 2 == 0: er += 1
    local_e = cv2.GaussianBlur(edges, (er*2+1, er*2+1), 0)

    sp = skin > 0.5
    if sp.sum() < 100:
        return np.zeros((H,W), np.float32)

    dark_thr = max(float(np.percentile(dark_diff[sp], 92)), 8.)
    edge_thr = max(float(np.percentile(local_e[sp],   85)), 3.)

    candidates = ((dark_diff > dark_thr) & (local_e > edge_thr) &
                  (redness < 6.) & sp).astype(np.uint8) * 255
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))

    n_lab, lbl, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    mask = np.zeros((H,W), np.uint8)
    max_area = int((min(H,W)*0.06)**2)
    for i in range(1, n_lab):
        ar = stats[i, cv2.CC_STAT_AREA]
        mw = stats[i, cv2.CC_STAT_WIDTH]
        mh = stats[i, cv2.CC_STAT_HEIGHT]
        if ar < 4 or ar > max_area: continue
        if max(mw,mh) / max(1,min(mw,mh)) > 3.5: continue
        mask[lbl == i] = 255

    ex = max(3, int(min(H,W)*0.008))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ex*2+1,ex*2+1)))
    result = feather(mask.astype(np.float32)/255., r=max(2,ex))
    logger.info("Mole mask: ~%d elements", int((mask>127).sum() / max(1, np.pi*ex**2)))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Blemish Removal
# ══════════════════════════════════════════════════════════════════════════════

def remove_blemishes(face: np.ndarray, skin: np.ndarray, moles: np.ndarray) -> np.ndarray:
    """Удаляет красно-тёмные дефекты (прыщи). Родинки защищены."""
    H, W = face.shape[:2]
    r = max(21, int(min(H,W)*0.04))
    if r % 2 == 0: r += 1

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)
    L, a = lab[:,:,0], lab[:,:,1]
    L_loc = cv2.GaussianBlur(L, (r,r), 0)
    a_loc = cv2.GaussianBlur(a, (r,r), 0)

    # Дефект: темнее + краснее локального окружения
    score = (L_loc - L)*0.5 + (a - a_loc)*1.2
    sp = skin > 0.5
    if sp.sum() < 100: return face

    thr = max(float(np.percentile(score[sp], 88)), 5.)
    bm  = ((score > thr) & sp).astype(np.uint8) * 255
    bm[moles > 0.5] = 0  # защищаем родинки

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, k3)
    bm = cv2.dilate(bm, k5, iterations=1)
    bm[moles > 0.4] = 0  # снова после dilate

    n = int(bm.sum()/255)
    mx = int(H*W*0.04)
    if n == 0 or n > mx:
        if n > mx: logger.info("Blemish %d > %d skip", n, mx)
        return face
    r_ = cv2.inpaint(face, bm, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    logger.info("Blemish inpaint: %d px", n)
    return r_


# ══════════════════════════════════════════════════════════════════════════════
# Frequency Separation — ИСПРАВЛЕН (убран двойной blend)
# ══════════════════════════════════════════════════════════════════════════════

def _fs(ch: np.ndarray, r: int):
    k = r*2+1
    low = cv2.GaussianBlur(ch.astype(np.float32),(k,k),r/2.)
    return low, ch.astype(np.float32)-low


def fs_retouch(face: np.ndarray, skin_soft: np.ndarray, moles: np.ndarray) -> np.ndarray:
    """
    Photoshop Frequency Separation retouch.
    ИСПРАВЛЕНО: результат применяется НАПРЯМУЮ через одну маску strength.
    Нет двойного blend.

    Этапы:
      1. Разделяем на low_freq + high_freq
      2. Корректируем low_freq:
         - убираем красноту (a-канал)
         - добавляем тёплый тон (b-канал)
         - Dodge & Burn (L-канал coarse + mid)
         - мягкое сглаживание тона
      3. HIGH_FREQ возвращаем 100% нетронутым
      4. final = clip(low_corrected + high_freq)
      5. Применяем через skin * strength маску (без повтора)
    """
    H, W = face.shape[:2]
    m = skin_soft > 0.2
    if m.sum() < 200:
        return face

    # Эффективная маска: кожа минус родинки
    eff = skin_soft * (1. - moles * MOLE_PROTECTION)

    # Адаптивные радиусы
    r_tone  = max(45, int(min(H,W) * BLUR_RADIUS_FACTOR));   r_tone  += r_tone%2==0
    r_db_c  = max(55, int(min(H,W) * BLUR_RADIUS_FACTOR*1.25)); r_db_c += r_db_c%2==0
    r_db_m  = max(25, int(min(H,W) * BLUR_RADIUS_FACTOR*0.55)); r_db_m += r_db_m%2==0
    r_micro = max(7,  int(min(H,W) * 0.014));                r_micro += r_micro%2==0
    r_smooth= max(r_tone//2, 15);                            r_smooth+= r_smooth%2==0

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)

    # ── 1. Убираем покраснения (a-канал) ──────────────────────────────────
    a_low, a_high = _fs(lab[:,:,1], r_tone)
    a_mean = float(a_low[m].mean())
    a_low += (a_mean - a_low) * eff * REDNESS_REDUCTION
    lab[:,:,1] = np.clip(a_low + a_high, 0, 255)

    # ── 2. Тёплый персиковый тон (b-канал) ────────────────────────────────
    b_low, b_high = _fs(lab[:,:,2], r_tone)
    b_mean = float(b_low[m].mean())
    # Сдвигаем к тёплому: среднее + небольшой сдвиг вверх
    b_target = b_mean + 3.0
    b_low += (b_target - b_low) * eff * REDNESS_REDUCTION * 0.30
    lab[:,:,2] = np.clip(b_low + b_high, 0, 255)

    # ── 3. Dodge & Burn — L-канал ─────────────────────────────────────────
    L = lab[:,:,0]

    # Coarse D&B (крупные зоны)
    L_low_c, L_high_c = _fs(L, r_db_c)
    L_mean_c = float(L_low_c[m].mean())
    L_low_c += (L_mean_c - L_low_c) * eff * DODGE_BURN_STRENGTH

    # Mid D&B (средние неровности)
    L_low_m, _ = _fs(L, r_db_m)
    L_mean_m = float(L_low_m[m].mean())
    L_low_m_corr = (L_mean_m - L_low_m) * eff * DODGE_BURN_STRENGTH * 0.42

    # ── 4. Skin smoothing — только low_freq ───────────────────────────────
    L_smooth = cv2.GaussianBlur(L_low_c, (r_smooth, r_smooth), 0)
    # Сглаживание с ослаблением у родинок
    sm_mask = eff * SKIN_SMOOTHING
    L_low_c = L_low_c*(1.-sm_mask) + L_smooth*sm_mask

    # ── 5. Micro-contrast ─────────────────────────────────────────────────
    L_result = L_low_c + L_low_m_corr + L_high_c
    L_mc_blur = cv2.GaussianBlur(L_result, (r_micro*2+1,r_micro*2+1), r_micro/2.)
    micro_mask = skin_soft * (1. - moles*0.7) * 0.22
    L_result += (L_result - L_mc_blur) * micro_mask

    lab[:,:,0] = np.clip(L_result, 0, 255)

    # ── 6. Рекомбинируем — HIGH_FREQ нетронут ─────────────────────────────
    retouched = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)

    # ── 7. ОДИН blend через strength маску (БЕЗ повтора) ──────────────────
    # strength * eff = где и насколько применяем коррекцию
    final_alpha = np.clip(eff * STRENGTH, 0., 1.)
    result = blend(retouched, face, final_alpha)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pad(bbox, H, W, pad=0.32):
    x1,y1,x2,y2 = bbox
    bw,bh = x2-x1, y2-y1
    return [max(0,x1-int(bw*pad)), max(0,y1-int(bh*pad)),
            min(W,x2+int(bw*pad)), min(H,y2+int(bh*pad))]


def _scale_down(img: np.ndarray, max_side: int):
    H, W = img.shape[:2]
    s = max_side / max(H, W)
    if s >= 1.0: return img, 1.0
    return cv2.resize(img,(int(W*s),int(H*s)),cv2.INTER_AREA), s


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
            "Pipeline v12 | strength=%.2f redness=%.2f "
            "smooth=%.2f db=%.2f mole=%.2f",
            STRENGTH, REDNESS_REDUCTION, SKIN_SMOOTHING,
            DODGE_BURN_STRENGTH, MOLE_PROTECTION,
        )

    def run(self, img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        origH, origW = img_bgr.shape[:2]
        result = img_bgr.copy()
        stats = {"faces": 0}

        # Детектируем на уменьшенной копии
        det_img, scale = _scale_down(img_bgr, MAX_PROC_SIZE)
        td = time.time()
        faces = self.detector.detect(det_img)
        stats["faces"] = len(faces)
        stats["t_detect"] = round(time.time()-td, 2)

        if not faces:
            logger.info("No faces — skipped")
            return result, stats

        tr = time.time()
        for fi in faces:
            # Масштабируем bbox к оригинальному размеру
            bx = [int(v/scale) for v in fi["bbox"]]
            x1,y1,x2,y2 = _pad(bx, origH, origW)
            face_crop = ucrop(img_bgr, x1, y1, x2, y2)
            if face_crop.size == 0: continue

            fH, fW = face_crop.shape[:2]
            logger.info("Face: %dx%d at [%d,%d,%d,%d]", fW,fH, x1,y1,x2,y2)

            try:
                # 1. Skin mask
                ts = time.time()
                skin_raw    = self.parser.get_skin_mask(face_crop)
                skin_strict = erode(skin_raw, k=3, n=1)
                skin_soft   = feather(dilate(skin_strict, k=5, n=1),
                                      r=max(8, int(min(fH,fW)*0.012)))
                logger.info("Skin mask: %.2fs", time.time()-ts)

                # 2. Mole protection
                tm = time.time()
                moles = build_mole_mask(face_crop, skin_strict)
                logger.info("Mole mask: %.2fs", time.time()-tm)

                # 3. Blemish removal (прыщи, родинки защищены)
                tb = time.time()
                proc = remove_blemishes(face_crop, skin_strict, moles)
                logger.info("Blemish: %.2fs", time.time()-tb)

                # 4. Frequency Separation Retouch (исправлен)
                tf = time.time()
                proc = fs_retouch(proc, skin_soft, moles)
                logger.info("FS retouch: %.2fs", time.time()-tf)

                # 5. Composite — бесшовная вставка
                pad_px = max(20, int(min(fH,fW)*0.06))
                comp = np.zeros((fH,fW), np.float32)
                comp[pad_px:-pad_px, pad_px:-pad_px] = 1.0
                comp = feather(comp, r=pad_px)

                result[y1:y2, x1:x2] = blend(proc, face_crop, comp)

            except Exception as e:
                logger.exception("Face error: %s", e)

        stats["t_retouch"] = round(time.time()-tr, 2)
        stats["t_total"]   = round(time.time()-t0, 2)
        logger.info("Done: faces=%d total=%.2fs", stats["faces"], stats["t_total"])
        return result, stats
