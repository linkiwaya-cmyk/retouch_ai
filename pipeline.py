"""
RetouchPipeline v9 — Photoshop Frequency Separation workflow

Точная реализация того как ретушёр работает в Photoshop:

  LOW_FREQ  = GaussianBlur(original, large_radius)   ← тон, цвет, яркость
  HIGH_FREQ = original - LOW_FREQ                    ← текстура, поры

  Работаем ТОЛЬКО с LOW_FREQ:
    → выравниваем тон (a-канал Lab)
    → убираем покраснения
    → Dodge & Burn по зонам яркости

  HIGH_FREQ возвращаем 100% нетронутым.

  final = clip(LOW_FREQ_corrected + HIGH_FREQ, 0, 255)

Параметры настроены под эталон:
  TONE_STR  = 0.60  (выравнивание тона)
  DB_STR    = 0.50  (Dodge & Burn)
  MICRO_STR = 0.28  (micro-contrast)

Скорость: работаем на downscaled копии (max 2048px), upscale обратно.
"""
from __future__ import annotations

import logging, os, time
from pathlib import Path
from typing import Optional

import cv2, numpy as np, torch

from utils import blend, feather, dilate, erode, crop as ucrop

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARSING_DIR = os.environ.get("BISENET_WEIGHT_DIR", str(Path.home()/".cache/facexlib"))
FACE_CONF   = float(os.environ.get("FACE_CONF", "0.5"))

# ── Параметры ──────────────────────────────────────────────────────────────────
TONE_STR  = float(os.environ.get("TONE_STRENGTH",  "0.60"))  # выравнивание тона/цвета
DB_STR    = float(os.environ.get("DB_STRENGTH",    "0.50"))  # Dodge & Burn яркости
MICRO_STR = float(os.environ.get("MICRO_STRENGTH", "0.28"))  # micro-contrast
BLEMISH_PCT = float(os.environ.get("BLEMISH_MAX_PCT", "0.04"))

# Max сторона для обработки (скорость)
MAX_PROC_SIZE = int(os.environ.get("MAX_PROC_SIZE", "2048"))

_SKIN   = {1,2,3,4,5,6,7,8,9,10,13}
_NOSKIN = {11,12,17,18}


# ══════════════════════════════════════════════════════════════════════════════
# Face Detector
# ══════════════════════════════════════════════════════════════════════════════

class FaceDetector:
    def __init__(self):
        self._app = self._ret = None
        try:
            from insightface.app import FaceAnalysis
            a = FaceAnalysis(name="buffalo_l",
                             providers=["CUDAExecutionProvider","CPUExecutionProvider"])
            a.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640,640))
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
                boxes = self._ret.detect_faces(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),
                                               conf_threshold=FACE_CONF)
            out = [{"bbox":[int(b[0]),int(b[1]),int(b[2]),int(b[3])],"score":float(b[4])} for b in boxes]
        else:
            return []
        out.sort(key=lambda d:(d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]),reverse=True)
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
            self._net = init_parsing_model(model_name="bisenet",
                                           device=str(DEVICE),
                                           model_rootpath=PARSING_DIR)
            self._net.eval()
            logger.info("FaceParser: BiSeNet")
        except Exception as e:
            logger.warning("BiSeNet: %s", e)

    def mask(self, face: np.ndarray) -> np.ndarray:
        return self._bisenet(face) if self._net else self._hsv(face)

    def _bisenet(self, face: np.ndarray) -> np.ndarray:
        H,W = face.shape[:2]
        x = cv2.resize(face,(512,512))
        x = (cv2.cvtColor(x,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
             -[.485,.456,.406])/[.229,.224,.225]
        t = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            seg = self._net(t)[0].squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
        m = np.zeros((512,512),np.uint8)
        for l in _SKIN:   m[seg==l]=255
        for l in _NOSKIN: m[seg==l]=0
        return cv2.resize(m,(W,H),cv2.INTER_NEAREST).astype(np.float32)/255.

    def _hsv(self, face: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(face,cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv,np.array([0,15,60]),np.array([25,170,255]))
        return cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8)).astype(np.float32)/255.


# ══════════════════════════════════════════════════════════════════════════════
# CORE: Frequency Separation
# ══════════════════════════════════════════════════════════════════════════════

def fs(ch: np.ndarray, r: int):
    """Возвращает (low, high). low=blur, high=ch-low."""
    k = r*2+1
    low = cv2.GaussianBlur(ch.astype(np.float32),(k,k),r/2.)
    return low, ch.astype(np.float32)-low


def recombine(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(low+high, 0, 255)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Blemish inpaint
# ══════════════════════════════════════════════════════════════════════════════

def blemish_inpaint(face: np.ndarray, skin: np.ndarray) -> np.ndarray:
    H,W = face.shape[:2]
    mx = int(H*W*BLEMISH_PCT)
    r = max(21, int(min(H,W)*.04))
    if r%2==0: r+=1
    L = cv2.cvtColor(face,cv2.COLOR_BGR2Lab)[:,:,0].astype(np.float32)
    diff = np.abs(L - cv2.GaussianBlur(L,(r,r),0))
    sp = diff[skin>.5]
    if len(sp)<100: return face
    thr = max(np.percentile(sp,90), 10.)
    m = ((diff>thr)&(skin>.5)).astype(np.uint8)*255
    m = cv2.morphologyEx(m,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    m = cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=1)
    n = int(m.sum()/255)
    if n==0 or n>mx:
        if n>mx: logger.info("Blemish %d>%d skip",n,mx)
        return face
    r_ = cv2.inpaint(face,m,inpaintRadius=4,flags=cv2.INPAINT_TELEA)
    logger.info("Blemish: %d px",n)
    return r_


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2-5: Frequency Separation Retouch
# ══════════════════════════════════════════════════════════════════════════════

def fs_retouch(face: np.ndarray, skin_soft: np.ndarray) -> np.ndarray:
    """
    ГЛАВНАЯ ФУНКЦИЯ — Photoshop Frequency Separation:

    1. Разделяем на LOW + HIGH
    2. Корректируем LOW:
       - STEP 2: Tone/Color correction (убираем покраснения, a-channel)
       - STEP 3: Luminosity leveling (Dodge & Burn, L-channel)
       - STEP 4: Micro-contrast boost
    3. HIGH остаётся нетронутым
    4. final = LOW_corrected + HIGH
    """
    H, W = face.shape[:2]
    m = skin_soft > 0.25
    if m.sum() < 200:
        return face

    # Радиусы под разрешение изображения
    r_tone = max(45, int(min(H,W) * 0.09))   # для цвета/тона
    r_db   = max(55, int(min(H,W) * 0.11))   # для D&B яркости
    r_mid  = max(25, int(min(H,W) * 0.05))   # для mid-tone D&B
    r_micro= max(7,  int(min(H,W) * 0.014))  # для micro-contrast

    for r in [r_tone, r_db, r_mid, r_micro]:
        if r % 2 == 0: r += 1

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2Lab).astype(np.float32)

    # ── STEP 2: Tone/Color correction ─────────────────────────────────────
    # a-канал: покраснения
    a_low, a_high = fs(lab[:,:,1], r_tone)
    a_mean = float(a_low[m].mean())
    a_low_new = a_low + (a_mean - a_low) * skin_soft * TONE_STR
    lab[:,:,1] = np.clip(recombine(a_low_new, a_high), 0, 255)

    # b-канал: жёлтый/синий тон (слабее)
    b_low, b_high = fs(lab[:,:,2], r_tone)
    b_mean = float(b_low[m].mean())
    b_low_new = b_low + (b_mean - b_low) * skin_soft * TONE_STR * 0.35
    lab[:,:,2] = np.clip(recombine(b_low_new, b_high), 0, 255)

    # ── STEP 3: Dodge & Burn (L-channel) ──────────────────────────────────
    L = lab[:,:,0]

    # Coarse D&B — крупные зоны
    L_low_c, L_high_c = fs(L, r_db)
    L_mean_c = float(L_low_c[m].mean())
    L_low_c_new = L_low_c + (L_mean_c - L_low_c) * skin_soft * DB_STR

    # Mid D&B — средние неровности
    L_low_m, _ = fs(L, r_mid)
    L_mean_m = float(L_low_m[m].mean())
    L_low_m_new = L_low_m + (L_mean_m - L_low_m) * skin_soft * DB_STR * 0.42

    # HIGH_FREQ нетронут — рекомбинируем
    L_result = L_low_c_new + (L_low_m_new - L_low_m) + L_high_c
    lab[:,:,0] = np.clip(L_result, 0, 255)

    # ── STEP 4: Micro-contrast boost ──────────────────────────────────────
    r_mc = r_micro if r_micro % 2 == 1 else r_micro + 1
    L_cur = lab[:,:,0]
    L_blur = cv2.GaussianBlur(L_cur, (r_mc*2+1, r_mc*2+1), r_mc/2.)
    lab[:,:,0] = np.clip(L_cur + (L_cur - L_blur) * skin_soft * MICRO_STR, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pad(bbox, H, W, pad=0.30):
    x1,y1,x2,y2 = bbox
    bw,bh = x2-x1, y2-y1
    return [max(0,x1-int(bw*pad)), max(0,y1-int(bh*pad)),
            min(W,x2+int(bw*pad)), min(H,y2+int(bh*pad))]


def _scale_down(img: np.ndarray, max_side: int):
    """Масштабирует вниз если нужно. Возвращает (scaled, scale_factor)."""
    H, W = img.shape[:2]
    s = max_side / max(H, W)
    if s >= 1.0:
        return img, 1.0
    new_W, new_H = int(W*s), int(H*s)
    return cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA), s


def _scale_up(img: np.ndarray, orig_H: int, orig_W: int):
    if img.shape[:2] == (orig_H, orig_W):
        return img
    return cv2.resize(img, (orig_W, orig_H), interpolation=cv2.INTER_LANCZOS4)


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
        logger.info("Pipeline v9 | tone=%.2f db=%.2f micro=%.2f blemish=%.0f%% max_px=%d",
                    TONE_STR, DB_STR, MICRO_STR, BLEMISH_PCT*100, MAX_PROC_SIZE)

    def run(self, img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        FLIP SAFE:
        img_bgr уже в правильной ориентации (EXIF применён в decode_image).
        Кроп вставляется по тем же координатам → никакого flip.

        SPEED:
        Детектируем на downscaled копии (max MAX_PROC_SIZE),
        но ретушируем оригинальный кроп в полном разрешении.
        """
        t0 = time.time()
        origH, origW = img_bgr.shape[:2]
        result = img_bgr.copy()
        stats = {"faces": 0, "tone": TONE_STR, "db": DB_STR}

        # Детектируем на уменьшенной копии (быстрее)
        det_img, scale = _scale_down(img_bgr, MAX_PROC_SIZE)
        td = time.time()
        faces_det = self.detector.detect(det_img)
        stats["faces"] = len(faces_det)
        stats["t_detect"] = round(time.time()-td, 2)

        if not faces_det:
            logger.info("No faces detected")
            return result, stats

        tr = time.time()
        for fi in faces_det:
            # Масштабируем bbox обратно к оригинальному размеру
            bx = [int(v / scale) for v in fi["bbox"]]
            x1,y1,x2,y2 = _pad(bx, origH, origW, pad=0.32)

            face_crop = ucrop(img_bgr, x1, y1, x2, y2)
            if face_crop.size == 0:
                continue

            fH, fW = face_crop.shape[:2]
            logger.info("Face crop: %dx%d at [%d,%d,%d,%d]", fW,fH, x1,y1,x2,y2)

            try:
                # Skin mask
                skin_raw    = self.parser.mask(face_crop)
                skin_strict = erode(skin_raw, k=3, n=1)
                skin_soft   = feather(
                    dilate(skin_strict, k=5, n=1),
                    r=max(8, int(min(fH,fW)*0.012))
                )

                # Blemish inpaint (точечно)
                proc = blemish_inpaint(face_crop, skin_strict)

                # Frequency Separation Retouch (главное)
                proc = fs_retouch(proc, skin_soft)

                # Composite — бесшовная вставка в оригинал
                pad_px = max(20, int(min(fH,fW)*0.06))
                comp = np.zeros((fH,fW), np.float32)
                comp[pad_px:-pad_px, pad_px:-pad_px] = 1.0
                comp = feather(comp, r=pad_px)

                # Вставляем по ТЕМ ЖЕ координатам — нет flip
                result[y1:y2, x1:x2] = blend(proc, face_crop, comp)

            except Exception as e:
                logger.exception("Face error: %s", e)

        stats["t_retouch"] = round(time.time()-tr, 2)
        stats["t_total"]   = round(time.time()-t0, 2)
        logger.info("Done: faces=%d total=%.2fs (det=%.2fs ret=%.2fs)",
                    stats["faces"], stats["t_total"],
                    stats.get("t_detect",0), stats.get("t_retouch",0))
        return result, stats
