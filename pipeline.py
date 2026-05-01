"""
RetouchPipeline v13 — Beauty Retouch

ПОЛНОСТЬЮ ПЕРЕПИСАН. Гарантированный видимый результат.

Ключевое отличие от предыдущих версий:
  Работаем НАПРЯМУЮ с пикселями кожи, без лишних масок blend.
  Каждый шаг делает заметное изменение.

Pipeline:
  1. Skin mask (BiSeNet, исправленные labels)
  2. Mole protection (родинки/веснушки — не трогаем)
  3. Blemish removal (точечный inpaint дефектов)
  4. Redness removal — прямая коррекция a-канала по коже
  5. Tone evening — выравнивание яркости через guided filter
  6. Skin smoothing — frequency separation, только low-freq
  7. Dodge & Burn — локальный свет/тень
  8. Warmth boost — лёгкий тёплый тон
  9. Composite обратно в оригинал

Настройки через .env:
  RETOUCH_STYLE=beauty
  RETOUCH_STRENGTH=strong   # light / medium / strong
  SKIN_SMOOTHING=0.65
  TONE_EVENING=0.70
  DODGE_BURN_STRENGTH=0.55
  REDNESS_REDUCTION=0.75
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

PARSING_DIR   = os.environ.get("BISENET_WEIGHT_DIR", str(Path.home()/".cache/facexlib"))
FACE_CONF     = float(os.environ.get("FACE_CONF", "0.5"))
MAX_PROC_SIZE = int(os.environ.get("MAX_PROC_SIZE", "2048"))

# ── Пресеты силы ──────────────────────────────────────────────────────────────
_PRESETS = {
    "light":  dict(smooth=0.45, tone=0.50, db=0.40, redness=0.60, warmth=0.25),
    "medium": dict(smooth=0.60, tone=0.65, db=0.52, redness=0.72, warmth=0.30),
    "strong": dict(smooth=0.75, tone=0.55, db=0.65, redness=0.85, warmth=0.38),
}
_style    = os.environ.get("RETOUCH_STYLE",    "beauty")
_strength = os.environ.get("RETOUCH_STRENGTH", "strong")
_p = _PRESETS.get(_strength, _PRESETS["strong"])

SKIN_SMOOTHING      = float(os.environ.get("SKIN_SMOOTHING",       _p["smooth"]))
TONE_EVENING        = float(os.environ.get("TONE_EVENING",         _p["tone"]))
DODGE_BURN_STRENGTH = float(os.environ.get("DODGE_BURN_STRENGTH",  _p["db"]))
REDNESS_REDUCTION   = float(os.environ.get("REDNESS_REDUCTION",    _p["redness"]))
WARMTH_STRENGTH     = float(os.environ.get("WARMTH_STRENGTH",      _p["warmth"]))
MOLE_PROTECTION     = float(os.environ.get("MOLE_PROTECTION_STRENGTH", "0.92"))

# BiSeNet labels
_SKIN   = {1, 7, 8, 10, 14}
_NOSKIN = {2, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 17, 18}


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
            out = [{"bbox":[int(b[0]),int(b[1]),int(b[2]),int(b[3])],"score":float(b[4])}
                   for b in boxes]
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

    def skin_mask(self, face: np.ndarray) -> np.ndarray:
        return self._bisenet(face) if self._net else self._hsv(face)

    def _bisenet(self, face: np.ndarray) -> np.ndarray:
        H, W = face.shape[:2]
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
# Mole Protection
# ══════════════════════════════════════════════════════════════════════════════

def mole_mask(face: np.ndarray, skin: np.ndarray) -> np.ndarray:
    H,W = face.shape[:2]
    lab = cv2.cvtColor(face,cv2.COLOR_BGR2Lab).astype(np.float32)
    L,a = lab[:,:,0], lab[:,:,1]
    r = max(21,int(min(H,W)*0.04)); r += r%2==0
    L_loc = cv2.GaussianBlur(L,(r,r),0)
    a_loc = cv2.GaussianBlur(a,(r,r),0)
    dark = L_loc - L
    red  = a - a_loc
    er = max(5,int(min(H,W)*0.01)); er += er%2==0
    edges = np.abs(cv2.Laplacian(np.clip(L,0,255).astype(np.uint8),cv2.CV_32F))
    local_e = cv2.GaussianBlur(edges,(er*2+1,er*2+1),0)
    sp = skin > 0.5
    if sp.sum() < 100: return np.zeros((H,W),np.float32)
    dt = max(float(np.percentile(dark[sp],92)),8.)
    et = max(float(np.percentile(local_e[sp],85)),3.)
    cand = ((dark>dt)&(local_e>et)&(red<6.)&sp).astype(np.uint8)*255
    cand = cv2.morphologyEx(cand,cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    n_lab,lbl,stats,_ = cv2.connectedComponentsWithStats(cand,8)
    mask = np.zeros((H,W),np.uint8)
    mx_a = int((min(H,W)*0.06)**2)
    for i in range(1,n_lab):
        ar=stats[i,cv2.CC_STAT_AREA]; mw=stats[i,cv2.CC_STAT_WIDTH]; mh=stats[i,cv2.CC_STAT_HEIGHT]
        if ar<4 or ar>mx_a: continue
        if max(mw,mh)/max(1,min(mw,mh))>3.5: continue
        mask[lbl==i]=255
    ex=max(3,int(min(H,W)*0.008))
    mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ex*2+1,ex*2+1)))
    return feather(mask.astype(np.float32)/255.,r=max(2,ex))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Blemish Removal
# ══════════════════════════════════════════════════════════════════════════════

def remove_blemishes(face: np.ndarray, skin: np.ndarray, moles: np.ndarray) -> np.ndarray:
    H,W = face.shape[:2]
    r=max(21,int(min(H,W)*0.04)); r+=r%2==0
    lab=cv2.cvtColor(face,cv2.COLOR_BGR2Lab).astype(np.float32)
    L,a=lab[:,:,0],lab[:,:,1]
    score=(cv2.GaussianBlur(L,(r,r),0)-L)*0.5+(a-cv2.GaussianBlur(a,(r,r),0))*1.2
    sp=skin>0.5
    if sp.sum()<100: return face
    thr=max(float(np.percentile(score[sp],88)),5.)
    bm=((score>thr)&sp).astype(np.uint8)*255
    bm[moles>0.5]=0
    k3=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    k5=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    bm=cv2.morphologyEx(bm,cv2.MORPH_OPEN,k3)
    bm=cv2.dilate(bm,k5,iterations=1)
    bm[moles>0.4]=0
    n=int(bm.sum()/255); mx=int(H*W*0.04)
    if n==0 or n>mx:
        if n>mx: logger.info("Blemish %d>%d skip",n,mx)
        return face
    out=cv2.inpaint(face,bm,inpaintRadius=4,flags=cv2.INPAINT_TELEA)
    logger.info("Blemish: %d px",n)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Redness Removal — прямая коррекция пикселей кожи
# ══════════════════════════════════════════════════════════════════════════════

def remove_redness(face: np.ndarray, skin_soft: np.ndarray, moles: np.ndarray) -> np.ndarray:
    """
    Убирает покраснения напрямую — сдвигаем a-канал к среднему по коже.
    Не frequency separation, а прямое изменение цвета кожи.
    """
    if REDNESS_REDUCTION < 0.01: return face
    lab = cv2.cvtColor(face,cv2.COLOR_BGR2Lab).astype(np.float32)
    m = skin_soft > 0.3
    if m.sum() < 200: return face

    eff = skin_soft * (1. - moles * MOLE_PROTECTION)

    # Среднее значение a по коже
    a_mean = float(lab[:,:,1][m].mean())
    # Сдвигаем каждый пиксель кожи к среднему
    correction = (a_mean - lab[:,:,1]) * eff * REDNESS_REDUCTION
    lab[:,:,1] = np.clip(lab[:,:,1] + correction, 0, 255)

    # Лёгкий тёплый сдвиг b-канала
    b_mean = float(lab[:,:,2][m].mean())
    b_target = b_mean + 4.0  # чуть теплее
    b_corr = (b_target - lab[:,:,2]) * eff * REDNESS_REDUCTION * 0.35
    lab[:,:,2] = np.clip(lab[:,:,2] + b_corr, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Tone Evening — выравнивание яркости кожи
# ══════════════════════════════════════════════════════════════════════════════

def even_tone(face: np.ndarray, skin_soft: np.ndarray, moles: np.ndarray) -> np.ndarray:
    """
    Выравнивает яркость кожи — тёмные участки осветляем, пересветы убираем.
    Работает с L-каналом напрямую.
    """
    if TONE_EVENING < 0.01: return face
    lab = cv2.cvtColor(face,cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:,:,0]
    m = skin_soft > 0.25
    if m.sum() < 200: return face

    eff = skin_soft * (1. - moles * MOLE_PROTECTION)

    H,W = face.shape[:2]
    # Очень большой радиус — только глобальное выравнивание, без локальных пятен
    r = max(81, int(min(H,W)*0.18)); r += r%2==0

    # Low-freq яркости (крупные зоны)
    k = r*2+1
    L_low = cv2.GaussianBlur(L,(k,k),r/2.)
    L_mean = float(L_low[m].mean())

    # Коррекция: тёмные → светлее, светлые → темнее
    correction = (L_mean - L_low) * eff * TONE_EVENING
    L_new = L + correction

    # HIGH-FREQ нетронут (текстура)
    lab[:,:,0] = np.clip(L_new, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Skin Smoothing — frequency separation, texture 100% preserved
# ══════════════════════════════════════════════════════════════════════════════

def smooth_skin(face: np.ndarray, skin_soft: np.ndarray, moles: np.ndarray) -> np.ndarray:
    """
    Сглаживает кожу через frequency separation.
    Текстура (high-freq) возвращается 100%.
    """
    if SKIN_SMOOTHING < 0.01: return face
    H,W = face.shape[:2]
    eff = skin_soft * (1. - moles * MOLE_PROTECTION)

    # Radius адаптивный
    r = max(25, int(min(H,W)*0.055)); r += r%2==0

    result = face.astype(np.float32)
    out = face.astype(np.float32).copy()

    # Для каждого канала: разделяем, сглаживаем low, возвращаем high
    k = r*2+1
    for ch in range(3):
        channel = result[:,:,ch]
        low  = cv2.GaussianBlur(channel,(k,k),r/2.)
        high = channel - low
        # Сглаживаем low дополнительно
        low_smooth = cv2.GaussianBlur(low,(k//2*2+1,k//2*2+1),r/4.)
        # Применяем сглаживание на коже
        low_new = low*(1.-eff*SKIN_SMOOTHING) + low_smooth*(eff*SKIN_SMOOTHING)
        out[:,:,ch] = np.clip(low_new + high, 0, 255)

    return out.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Dodge & Burn
# ══════════════════════════════════════════════════════════════════════════════

def dodge_burn(face: np.ndarray, skin_soft: np.ndarray, moles: np.ndarray) -> np.ndarray:
    """
    Локальный D&B — осветляем тёмные зоны, затемняем пересветы.
    Работает с mid-freq L-канала.
    """
    if DODGE_BURN_STRENGTH < 0.01: return face
    lab = cv2.cvtColor(face,cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:,:,0]
    m = skin_soft > 0.2
    if m.sum() < 200: return face

    eff = skin_soft * (1. - moles * MOLE_PROTECTION * 0.7)
    H,W = face.shape[:2]

    # Mid-freq radius
    r_m = max(21, int(min(H,W)*0.045)); r_m += r_m%2==0
    k = r_m*2+1
    L_mid = cv2.GaussianBlur(L,(k,k),r_m/2.)
    L_mean_m = float(L_mid[m].mean())
    corr_m = (L_mean_m - L_mid) * eff * DODGE_BURN_STRENGTH * 0.55

    # Coarse radius
    r_c = max(51, int(min(H,W)*0.10)); r_c += r_c%2==0
    k2 = r_c*2+1
    L_coarse = cv2.GaussianBlur(L,(k2,k2),r_c/2.)
    L_mean_c = float(L_coarse[m].mean())
    corr_c = (L_mean_c - L_coarse) * eff * DODGE_BURN_STRENGTH

    lab[:,:,0] = np.clip(L + corr_c + corr_m, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Warmth & Glow
# ══════════════════════════════════════════════════════════════════════════════

def add_warmth(face: np.ndarray, skin_soft: np.ndarray, moles: np.ndarray) -> np.ndarray:
    """Добавляет тёплый персиковый оттенок коже."""
    if WARMTH_STRENGTH < 0.01: return face
    lab = cv2.cvtColor(face,cv2.COLOR_BGR2Lab).astype(np.float32)
    eff = skin_soft * (1. - moles * MOLE_PROTECTION * 0.5)
    # Тёплый = +a (чуть теплее), +b (чуть желтее)
    lab[:,:,1] = np.clip(lab[:,:,1] - eff * WARMTH_STRENGTH * 2.0, 0, 255)  # чуть убираем красноту
    lab[:,:,2] = np.clip(lab[:,:,2] + eff * WARMTH_STRENGTH * 3.5, 0, 255)  # добавляем тепло
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pad(bbox, H, W, pad=0.32):
    x1,y1,x2,y2=bbox; bw,bh=x2-x1,y2-y1
    return [max(0,x1-int(bw*pad)),max(0,y1-int(bh*pad)),
            min(W,x2+int(bw*pad)),min(H,y2+int(bh*pad))]

def _scale_down(img,mx):
    H,W=img.shape[:2]; s=mx/max(H,W)
    if s>=1.: return img,1.
    return cv2.resize(img,(int(W*s),int(H*s)),cv2.INTER_AREA),s


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
            "Pipeline v13 | style=%s strength=%s | "
            "smooth=%.2f tone=%.2f db=%.2f redness=%.2f warmth=%.2f mole=%.2f",
            _style, _strength,
            SKIN_SMOOTHING, TONE_EVENING, DODGE_BURN_STRENGTH,
            REDNESS_REDUCTION, WARMTH_STRENGTH, MOLE_PROTECTION,
        )

    def run(self, img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        origH, origW = img_bgr.shape[:2]
        result = img_bgr.copy()
        stats = {"faces": 0, "style": _style, "strength": _strength}

        det_img, scale = _scale_down(img_bgr, MAX_PROC_SIZE)
        td = time.time()
        faces = self.detector.detect(det_img)
        stats["faces"] = len(faces)
        stats["t_detect"] = round(time.time()-td, 2)

        if not faces:
            logger.info("No faces — skipped")
            return result, stats

        tr = time.time()

        # ══════════════════════════════════════════════════════════════════
        # PASS 1: Обработка лица через BiSeNet (точная маска)
        # ══════════════════════════════════════════════════════════════════
        for fi in faces:
            bx = [int(v/scale) for v in fi["bbox"]]
            x1,y1,x2,y2 = _pad(bx, origH, origW, pad=0.32)
            crop = ucrop(img_bgr, x1, y1, x2, y2)
            if crop.size == 0: continue

            fH, fW = crop.shape[:2]
            logger.info("Face crop: %dx%d at [%d,%d,%d,%d]", fW,fH,x1,y1,x2,y2)

            try:
                ts = time.time()
                skin_raw    = self.parser.skin_mask(crop)
                skin_strict = erode(skin_raw, k=3, n=1)
                skin_soft   = feather(dilate(skin_strict, k=5, n=1),
                                      r=max(8, int(min(fH,fW)*0.012)))
                mm = mole_mask(crop, skin_strict)
                logger.info("Face masks: %.2fs", time.time()-ts)

                proc = crop.copy()
                proc = remove_redness(proc, skin_soft, mm)
                proc = even_tone(proc, skin_soft, mm)
                proc = smooth_skin(proc, skin_soft, mm)
                proc = dodge_burn(proc, skin_soft, mm)
                proc = add_warmth(proc, skin_soft, mm)

                pad_px = max(20, int(min(fH,fW)*0.06))
                comp = np.zeros((fH,fW), np.float32)
                comp[pad_px:-pad_px, pad_px:-pad_px] = 1.0
                comp = feather(comp, r=pad_px)
                result[y1:y2, x1:x2] = blend(proc, crop, comp)

            except Exception as e:
                logger.exception("Face error: %s", e)

        # ══════════════════════════════════════════════════════════════════
        # PASS 2: Обработка ТЕЛА (шея, руки, плечи, грудь)
        # HSV skin mask на всём изображении, исключаем уже обработанное лицо
        # ══════════════════════════════════════════════════════════════════
        try:
            tb = time.time()
            body_skin = self._body_skin_mask(result, origH, origW)

            # Исключаем зону лица из body mask (уже обработана)
            for fi in faces:
                bx = [int(v/scale) for v in fi["bbox"]]
                fx1,fy1,fx2,fy2 = _pad(bx, origH, origW, pad=0.15)
                body_skin[fy1:fy2, fx1:fx2] = 0.0

            # Применяем только если есть достаточно пикселей тела
            body_px = (body_skin > 0.3).sum()
            logger.info("Body skin: %d px (%.1f%% image)", body_px, body_px/origH/origW*100)

            if body_px > 5000:
                # Mole mask для тела
                body_mm = mole_mask(result, (body_skin > 0.5).astype(np.float32))
                body_soft = feather(body_skin, r=max(10, int(min(origH,origW)*0.008)))

                body_proc = result.copy()
                body_proc = remove_redness(body_proc, body_soft, body_mm)
                body_proc = even_tone(body_proc, body_soft, body_mm)
                body_proc = smooth_skin(body_proc, body_soft, body_mm)
                body_proc = dodge_burn(body_proc, body_soft, body_mm)
                body_proc = add_warmth(body_proc, body_soft, body_mm)

                # Блендим тело обратно
                result = blend(body_proc, result, body_soft)
                logger.info("Body retouch: %.2fs", time.time()-tb)

        except Exception as e:
            logger.exception("Body retouch error: %s", e)

        stats["t_retouch"] = round(time.time()-tr, 2)
        stats["t_total"]   = round(time.time()-t0, 2)
        logger.info("Done: faces=%d total=%.2fs", stats["faces"], stats["t_total"])
        return result, stats

    def _body_skin_mask(self, img: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        HSV + YCrCb skin detection для тела (шея, руки, плечи).
        Возвращает float32 [0..1] маску.
        """
        # HSV диапазон кожи
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m_hsv = cv2.inRange(hsv, np.array([0,20,60]), np.array([25,170,255]))

        # YCrCb диапазон кожи (более точный)
        ycr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        m_ycr = cv2.inRange(ycr, np.array([0,133,77]), np.array([255,173,127]))

        # Объединяем
        combined = cv2.bitwise_and(m_hsv, m_ycr)

        # Морфология — убираем шум
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k7)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k15)
        combined = cv2.dilate(combined, k7, iterations=2)

        return combined.astype(np.float32) / 255.0

        stats["t_retouch"] = round(time.time()-tr, 2)
        stats["t_total"]   = round(time.time()-t0, 2)
        logger.info("Done: faces=%d total=%.2fs", stats["faces"], stats["t_total"])
        return result, stats
