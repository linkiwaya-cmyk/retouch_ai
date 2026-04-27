"""
RetouchPipeline v3 — Professional Beauty Retouch

Принцип: ручная ретушь по логике фотографа.
НЕ AI-перерисовывание. НЕ blur. НЕ пластик.

Этапы:
  1. Face detection
  2. Face parsing → skin mask (только кожа, без глаз/губ/волос/фона)
  3. Blemish removal — удаление дефектов через inpainting
  4. Skin tone evening — выравнивание пятен и покраснений (low-freq)
  5. Dodge & Burn — профессиональный D&B по skin mask
  6. Мягкое сглаживание (frequency separation, texture preserved)
  7. CodeFormer — МИНИМАЛЬНО (fidelity=0.95, blend=0.10), только дефекты
  8. Финальный composite

Настройки через .env:
  RETOUCH_STRENGTH=medium   # light / medium / strong
  CF_FIDELITY=0.95
  CF_BLEND=0.10
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
from utils import blend_with_mask, feather_mask, dilate_mask, safe_crop

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CF_WEIGHT = os.environ.get(
    "CODEFORMER_WEIGHT",
    str(Path.home() / ".cache/codeformer/codeformer.pth"),
)
PARSING_WEIGHT_DIR = os.environ.get(
    "BISENET_WEIGHT_DIR",
    str(Path.home() / ".cache/facexlib"),
)

# ── Сила ретуши (RETOUCH_STRENGTH=light/medium/strong) ────────────────────────
_STRENGTH_PRESETS = {
    "light":  {"smooth": 0.30, "db": 0.12, "tone": 0.25, "cf_blend": 0.08},
    "medium": {"smooth": 0.50, "db": 0.22, "tone": 0.45, "cf_blend": 0.12},
    "strong": {"smooth": 0.70, "db": 0.35, "tone": 0.65, "cf_blend": 0.15},
}

_preset_name = os.environ.get("RETOUCH_STRENGTH", "medium")
_preset = _STRENGTH_PRESETS.get(_preset_name, _STRENGTH_PRESETS["medium"])

SMOOTH_STRENGTH = float(os.environ.get("SKIN_SMOOTH_STRENGTH", _preset["smooth"]))
DB_STRENGTH     = float(os.environ.get("DB_STRENGTH",          _preset["db"]))
TONE_STRENGTH   = float(os.environ.get("TONE_STRENGTH",        _preset["tone"]))
CF_FIDELITY     = float(os.environ.get("CF_FIDELITY",          "0.95"))
CF_BLEND        = float(os.environ.get("CODEFORMER_BLEND",     _preset["cf_blend"]))
FACE_CONF       = float(os.environ.get("FACE_CONF",            "0.5"))
_CF_SIZE        = 512

# BiSeNet labels
_SKIN_LABELS    = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13}
_NON_SKIN       = {11, 12, 17, 18}  # глаза, губы


# ══════════════════════════════════════════════════════════════════════════════
# Face Detector
# ══════════════════════════════════════════════════════════════════════════════

class FaceDetector:
    def __init__(self):
        self._app = None
        self._retina = None
        self._load()

    def _load(self):
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
            self._app = app
            logger.info("FaceDetector: InsightFace buffalo_l")
        except Exception as exc:
            logger.warning("InsightFace unavailable (%s), trying RetinaFace", exc)
            try:
                from facexlib.detection import init_detection_model
                self._retina = init_detection_model(
                    "retinaface_resnet50", half=False, device=str(DEVICE)
                )
                logger.info("FaceDetector: RetinaFace fallback")
            except Exception as exc2:
                logger.error("No face detector: %s", exc2)

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        if self._app:
            return self._detect_insight(img_bgr)
        if self._retina:
            return self._detect_retina(img_bgr)
        return []

    def _detect_insight(self, img_bgr):
        faces = self._app.get(img_bgr)
        out = [
            {"bbox": f.bbox.astype(int).tolist(), "score": float(f.det_score)}
            for f in faces if f.det_score >= FACE_CONF
        ]
        out.sort(key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]), reverse=True)
        return out

    def _detect_retina(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            boxes = self._retina.detect_faces(rgb, conf_threshold=FACE_CONF)
        out = [{"bbox": [int(b[0]),int(b[1]),int(b[2]),int(b[3])], "score": float(b[4])} for b in boxes]
        out.sort(key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]), reverse=True)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Face Parser → skin mask
# ══════════════════════════════════════════════════════════════════════════════

class FaceParser:
    def __init__(self):
        self._net = None
        self._load()

    def _load(self):
        try:
            from facexlib.parsing import init_parsing_model
            self._net = init_parsing_model(
                model_name="bisenet", device=str(DEVICE),
                model_rootpath=PARSING_WEIGHT_DIR,
            )
            self._net.eval()
            logger.info("FaceParser: BiSeNet loaded")
        except Exception as exc:
            logger.warning("BiSeNet unavailable (%s). HSV fallback.", exc)

    def get_skin_mask(self, face_bgr: np.ndarray) -> np.ndarray:
        if self._net is not None:
            return self._bisenet(face_bgr)
        return self._hsv(face_bgr)

    def _bisenet(self, face_bgr: np.ndarray) -> np.ndarray:
        h, w = face_bgr.shape[:2]
        inp = cv2.resize(face_bgr, (512, 512))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = (inp - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
        t = torch.from_numpy(inp.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            out = self._net(t)[0]
        seg = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
        mask = np.zeros((512,512), np.uint8)
        for lbl in _SKIN_LABELS:
            mask[seg == lbl] = 255
        for lbl in _NON_SKIN:
            mask[seg == lbl] = 0
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.float32) / 255.0

    def _hsv(self, face_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,20,70]), np.array([20,170,255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        return mask.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Blemish Removal (inpainting дефектов)
# ══════════════════════════════════════════════════════════════════════════════

def remove_blemishes(face_bgr: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    """
    Детектирует локальные дефекты (прыщи, пятна) и убирает их через inpainting.
    Работает только в пределах skin_mask.
    """
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2Lab)
    L = lab[:,:,0].astype(np.float32)

    h, w = face_bgr.shape[:2]
    # Адаптивный радиус под разрешение
    k = max(21, int(min(h, w) * 0.04))
    if k % 2 == 0: k += 1

    local_mean = cv2.GaussianBlur(L, (k, k), 0)
    diff = np.abs(L - local_mean)

    # Порог для детекции дефектов
    thresh = np.percentile(diff[skin_mask > 0.3], 82) if skin_mask.sum() > 100 else 18.0
    blemish_mask = ((diff > thresh) & (skin_mask > 0.3)).astype(np.uint8) * 255

    # Морфология — убираем шум, расширяем дефекты
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    blemish_mask = cv2.morphologyEx(blemish_mask, cv2.MORPH_OPEN, kernel)
    blemish_mask = cv2.dilate(blemish_mask, kernel, iterations=2)

    if blemish_mask.sum() == 0:
        return face_bgr

    # Inpainting — заполняем дефекты соседними пикселями
    result = cv2.inpaint(face_bgr, blemish_mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    logger.info("Blemish removal: %d pixels inpainted", int(blemish_mask.sum()/255))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Skin Tone Evening (выравнивание пятен/покраснений)
# ══════════════════════════════════════════════════════════════════════════════

def even_skin_tone(
    face_bgr: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = TONE_STRENGTH,
) -> np.ndarray:
    """
    Выравнивает неровный тон кожи (покраснения, пятна).
    Работает в пространстве Lab — не меняет цвет, только выравнивает L и a.
    Текстура (high-freq) сохраняется полностью.
    """
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    h, w = face_bgr.shape[:2]

    k = max(31, int(min(h, w) * 0.07))
    if k % 2 == 0: k += 1

    result_lab = lab.copy()

    for ch in [0, 1]:  # L (яркость) и a (красный-зелёный)
        channel = lab[:,:,ch]
        # Low-freq = локальный средний тон
        low_freq = cv2.GaussianBlur(channel, (k, k), 0)
        # Global mean на коже
        skin_pixels = channel[skin_mask > 0.5]
        if len(skin_pixels) == 0:
            continue
        global_mean = float(skin_pixels.mean())
        # Целевой low-freq = сглаженный в сторону глобального среднего
        target_low = low_freq * (1 - strength) + global_mean * strength
        # Применяем только на коже
        corrected = channel.copy()
        m = skin_mask
        corrected = channel + (target_low - low_freq) * m * strength
        result_lab[:,:,ch] = np.clip(corrected, 0, 255)

    result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Dodge & Burn (профессиональный D&B)
# ══════════════════════════════════════════════════════════════════════════════

def dodge_and_burn(
    face_bgr: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = DB_STRENGTH,
) -> np.ndarray:
    """
    Профессиональный Dodge & Burn:
    - Работает в L канале (Lab) — не меняет цвет
    - Low-frequency D&B: выравнивает крупные зоны света/тени
    - High-frequency D&B: убирает мелкие неровности тона
    - Применяется только на skin_mask
    - Текстура кожи (поры) сохраняется полностью
    """
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:,:,0]
    h, w = face_bgr.shape[:2]

    # ── Low-frequency D&B (крупные зоны) ──────────────────────────────────
    k_low = max(51, int(min(h, w) * 0.12))
    if k_low % 2 == 0: k_low += 1
    L_low = cv2.GaussianBlur(L, (k_low, k_low), 0)

    # Разница между локальной яркостью и средней по коже
    skin_L = L[skin_mask > 0.5]
    if len(skin_L) == 0:
        return face_bgr
    mean_L = float(skin_L.mean())

    # Корректируем: тёмные участки -> dodge, светлые -> burn
    low_correction = (mean_L - L_low) * skin_mask * strength * 0.6
    L_corrected = L + low_correction

    # ── High-frequency D&B (мелкие неровности тона) ───────────────────────
    k_high = max(11, int(min(h, w) * 0.025))
    if k_high % 2 == 0: k_high += 1
    L_high_blur = cv2.GaussianBlur(L_corrected, (k_high, k_high), 0)
    L_high_freq = L_corrected - L_high_blur  # текстура

    # Unsharp mask на mid-frequencies (не на мелкой текстуре)
    k_mid = max(21, int(min(h, w) * 0.05))
    if k_mid % 2 == 0: k_mid += 1
    L_mid = cv2.GaussianBlur(L_corrected, (k_mid, k_mid), 0)
    L_mid_detail = L_corrected - L_mid
    L_corrected = L_corrected + L_mid_detail * skin_mask * strength * 0.4

    # ── Рекомбинируем ──────────────────────────────────────────────────────
    # Возвращаем high-freq текстуру нетронутой
    L_final = L_corrected - cv2.GaussianBlur(L_corrected, (k_high, k_high), 0) * 0
    L_final = np.clip(L_corrected, 0, 255)

    lab[:,:,0] = L_final
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Frequency Separation Smooth (текстура сохранена)
# ══════════════════════════════════════════════════════════════════════════════

def frequency_separation_smooth(
    face_bgr: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = SMOOTH_STRENGTH,
) -> np.ndarray:
    """
    Frequency separation:
    - Bilateral filter → low-freq (тон, цвет)
    - оригинал - low = high-freq (текстура, поры)
    - Сглаживаем ТОЛЬКО low-freq внутри skin_mask
    - High-freq добавляем обратно 100% → текстура кожи сохранена
    """
    img = face_bgr.astype(np.float32)
    h, w = face_bgr.shape[:2]

    # Адаптивный радиус
    d = max(9, int(min(h, w) * 0.012))
    if d % 2 == 0: d += 1

    # Bilateral — сохраняет края, убирает мелкие неровности цвета
    low_freq = cv2.bilateralFilter(
        face_bgr, d=d*2+1, sigmaColor=45, sigmaSpace=45
    ).astype(np.float32)

    # Текстура = оригинал минус low-freq
    high_freq = img - low_freq

    # Дополнительно сглаживаем low-freq Гауссом внутри маски
    k = max(5, d)
    if k % 2 == 0: k += 1
    smoother = cv2.GaussianBlur(low_freq, (k, k), 0)

    m = skin_mask[..., np.newaxis]
    low_out = low_freq * (1.0 - m * strength) + smoother * (m * strength)

    # Рекомбинируем: возвращаем текстуру полностью
    result = low_out + high_freq
    return np.clip(result, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: CodeFormer — минимально, только дефекты
# ══════════════════════════════════════════════════════════════════════════════

def run_codeformer_minimal(
    face_bgr: np.ndarray,
    net: torch.nn.Module,
    skin_mask: np.ndarray,
    fidelity: float = CF_FIDELITY,
    blend: float = CF_BLEND,
) -> tuple[np.ndarray, bool]:
    """
    CodeFormer с fidelity=0.95 и blend=0.10.
    Результат смешивается ТОЛЬКО на участках кожи.
    Возвращает (result, applied: bool).
    """
    if blend < 0.01:
        return face_bgr, False

    h, w = face_bgr.shape[:2]
    face_512 = cv2.resize(face_bgr, (_CF_SIZE, _CF_SIZE), interpolation=cv2.INTER_LANCZOS4)
    inp = cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = (inp - 0.5) / 0.5
    tensor = torch.from_numpy(inp.transpose(2,0,1)).unsqueeze(0).to(DEVICE)

    try:
        with torch.no_grad():
            output = net(tensor, w=fidelity, adain=True)
            if isinstance(output, (list, tuple)):
                output = output[0]
    except Exception as exc:
        logger.warning("CodeFormer inference failed: %s", exc)
        return face_bgr, False

    out_np = output.squeeze(0).permute(1,2,0).cpu().numpy()
    out_np = np.clip(out_np * 0.5 + 0.5, 0, 1)
    cf_bgr = cv2.cvtColor((out_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cf_bgr = cv2.resize(cf_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Смешиваем только на коже, очень мягко
    skin_feathered = feather_mask(skin_mask, radius=10)
    final_mask = skin_feathered * blend
    result = blend_with_mask(cf_bgr, face_bgr, final_mask)
    return result, True


# ══════════════════════════════════════════════════════════════════════════════
# Face helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pad_bbox(bbox: list, img_h: int, img_w: int, pad: float = 0.35) -> list:
    x1, y1, x2, y2 = bbox
    bw, bh = x2-x1, y2-y1
    px, py = int(bw*pad), int(bh*pad)
    return [max(0,x1-px), max(0,y1-py), min(img_w,x2+px), min(img_h,y2+py)]


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class RetouchPipeline:
    def __init__(self):
        self.detector: Optional[FaceDetector] = None
        self.parser: Optional[FaceParser] = None
        self.codeformer: Optional[torch.nn.Module] = None

    def load_models(self):
        logger.info("Loading FaceDetector…")
        self.detector = FaceDetector()
        logger.info("Loading FaceParser…")
        self.parser = FaceParser()
        logger.info("Loading CodeFormer…")
        self.codeformer = load_codeformer(weight_path=CF_WEIGHT, device=DEVICE)
        logger.info("All models loaded. RETOUCH_STRENGTH=%s smooth=%.2f db=%.2f tone=%.2f cf_blend=%.2f",
                    _preset_name, SMOOTH_STRENGTH, DB_STRENGTH, TONE_STRENGTH, CF_BLEND)

    def run(self, img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Полный pipeline ретуши.
        Returns: (result_bgr, stats_dict)
        """
        if not all([self.detector, self.parser, self.codeformer]):
            raise RuntimeError("Models not loaded.")

        h, w = img_bgr.shape[:2]
        result = img_bgr.copy()
        stats = {
            "faces_found": 0,
            "codeformer_applied": False,
            "db_applied": False,
            "smooth_applied": False,
            "retouch_strength": _preset_name,
            "smooth_strength": SMOOTH_STRENGTH,
            "db_strength": DB_STRENGTH,
            "cf_fidelity": CF_FIDELITY,
            "cf_blend": CF_BLEND,
        }

        # ── Detect faces ───────────────────────────────────────────────────
        faces = self.detector.detect(img_bgr)
        stats["faces_found"] = len(faces)
        logger.info("Faces found: %d", len(faces))

        if not faces:
            logger.info("No faces — applying mild global polish")
            try:
                skin_mask = self.parser.get_skin_mask(img_bgr)
                result = frequency_separation_smooth(result, skin_mask, strength=0.25)
                result = dodge_and_burn(result, skin_mask, strength=0.08)
                stats["smooth_applied"] = True
            except Exception as exc:
                logger.warning("Global polish failed: %s", exc)
            return result, stats

        for face in faces:
            bbox = _pad_bbox(face["bbox"], h, w, pad=0.35)
            x1, y1, x2, y2 = bbox
            face_crop = safe_crop(img_bgr, x1, y1, x2, y2)
            if face_crop.size == 0:
                continue

            fh, fw = face_crop.shape[:2]
            logger.info("Processing face %dx%d at [%d,%d,%d,%d]", fw, fh, x1, y1, x2, y2)

            try:
                # ── 1. Skin mask ───────────────────────────────────────────
                skin_mask = self.parser.get_skin_mask(face_crop)
                skin_smooth = feather_mask(
                    dilate_mask((skin_mask * 255).astype(np.uint8), ksize=7, iterations=1) / 255.0,
                    radius=12,
                )

                # ── 2. Blemish removal (inpainting) ───────────────────────
                processed = remove_blemishes(face_crop, skin_mask)

                # ── 3. Skin tone evening ───────────────────────────────────
                processed = even_skin_tone(processed, skin_smooth, strength=TONE_STRENGTH)

                # ── 4. Dodge & Burn ────────────────────────────────────────
                processed = dodge_and_burn(processed, skin_smooth, strength=DB_STRENGTH)
                stats["db_applied"] = True

                # ── 5. Frequency separation smooth ─────────────────────────
                processed = frequency_separation_smooth(processed, skin_smooth, strength=SMOOTH_STRENGTH)
                stats["smooth_applied"] = True

                # ── 6. CodeFormer — минимально ─────────────────────────────
                processed, cf_applied = run_codeformer_minimal(
                    processed, self.codeformer, skin_smooth,
                    fidelity=CF_FIDELITY, blend=CF_BLEND,
                )
                if cf_applied:
                    stats["codeformer_applied"] = True

                # ── 7. Composite face → original ───────────────────────────
                pad_px = max(15, int(min(fh, fw) * 0.08))
                comp_mask = np.zeros((fh, fw), np.float32)
                comp_mask[pad_px:-pad_px, pad_px:-pad_px] = 1.0
                comp_mask = feather_mask(comp_mask, radius=pad_px)

                blended = blend_with_mask(processed, face_crop, comp_mask)
                result[y1:y2, x1:x2] = blended

            except Exception as exc:
                logger.exception("Face processing failed: %s", exc)
                continue

        logger.info(
            "Pipeline done: faces=%d cf=%s db=%s smooth=%s",
            stats["faces_found"],
            stats["codeformer_applied"],
            stats["db_applied"],
            stats["smooth_applied"],
        )
        return result, stats
