"""
RetouchPipeline v2

Принцип: НЕ МЕНЯЕМ ИЗОБРАЖЕНИЕ — только улучшаем кожу как ретушёр.

Pipeline:
  1. Face detection (координаты лица)
  2. Face parsing → маска кожи (BiSeNet)
  3. Blemish detection → маска дефектов
  4. CodeFormer с высоким fidelity (0.92) — минимум изменений лица
     + blend только 15% поверх оригинала
  5. Frequency separation smoothing только на коже
  6. Лёгкий dodge & burn (сила 0.08)
  7. Финальный composite с оригиналом

Всё работает в оригинальном разрешении — без resize.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

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

# ── Настройки силы обработки ────────────────────────────────────────────────
# fidelity=0.92: CodeFormer максимально сохраняет оригинал
CF_FIDELITY = float(os.environ.get("CF_FIDELITY", "0.92"))
# blend=0.15: смешиваем только 15% результата CodeFormer с оригиналом
CF_BLEND = float(os.environ.get("CF_BLEND", "0.15"))
# Сглаживание кожи — мягкое, текстура сохранена
SMOOTH_STRENGTH = float(os.environ.get("SMOOTH_STRENGTH", "0.40"))
# Dodge & burn — едва заметный
DB_STRENGTH = float(os.environ.get("DB_STRENGTH", "0.08"))

FACE_CONF = float(os.environ.get("FACE_CONF", "0.5"))
_CF_SIZE = 512

# Метки кожи BiSeNet
_SKIN_LABELS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13}
_NON_SKIN = {11, 12, 17, 18}  # глаза, губы — не трогаем


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
                logger.error("No face detector available: %s", exc2)

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        """Returns [{'bbox': [x1,y1,x2,y2], 'score': float}] sorted by area."""
        if self._app:
            return self._detect_insightface(img_bgr)
        if self._retina:
            return self._detect_retina(img_bgr)
        return []

    def _detect_insightface(self, img_bgr):
        faces = self._app.get(img_bgr)
        results = [
            {"bbox": f.bbox.astype(int).tolist(), "score": float(f.det_score)}
            for f in faces if f.det_score >= FACE_CONF
        ]
        results.sort(
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True,
        )
        return results

    def _detect_retina(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            boxes = self._retina.detect_faces(rgb, conf_threshold=FACE_CONF)
        results = [
            {"bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])], "score": float(b[4])}
            for b in boxes
        ]
        results.sort(
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True,
        )
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Face Parser (BiSeNet → маска кожи)
# ══════════════════════════════════════════════════════════════════════════════

class FaceParser:
    def __init__(self):
        self._net = None
        self._load()

    def _load(self):
        try:
            from facexlib.parsing import init_parsing_model
            self._net = init_parsing_model(
                model_name="bisenet",
                device=str(DEVICE),
                model_rootpath=PARSING_WEIGHT_DIR,
            )
            self._net.eval()
            logger.info("FaceParser: BiSeNet loaded")
        except Exception as exc:
            logger.warning("BiSeNet unavailable (%s). Using HSV fallback.", exc)

    def get_skin_mask(self, face_bgr: np.ndarray) -> np.ndarray:
        """Returns float32 mask [0..1], same HxW as face_bgr."""
        if self._net is not None:
            return self._bisenet_mask(face_bgr)
        return self._hsv_mask(face_bgr)

    def _bisenet_mask(self, face_bgr: np.ndarray) -> np.ndarray:
        h, w = face_bgr.shape[:2]
        inp = cv2.resize(face_bgr, (512, 512))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std  = np.array([0.229, 0.224, 0.225], np.float32)
        inp = (inp - mean) / std
        t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = self._net(t)[0]
        seg = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
        mask = np.zeros((512, 512), np.uint8)
        for lbl in _SKIN_LABELS:
            mask[seg == lbl] = 255
        for lbl in _NON_SKIN:
            mask[seg == lbl] = 0
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.float32) / 255.0

    def _hsv_mask(self, face_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 170, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return mask.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# Blemish detector — находит дефекты кожи
# ══════════════════════════════════════════════════════════════════════════════

def detect_blemishes(face_bgr: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    """
    Детектирует прыщи и дефекты через анализ локальных аномалий яркости.
    Возвращает float32 маску [0..1] — только участки с дефектами.
    """
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype(np.float32)

    # Разница между пикселем и его локальным окружением
    ksize = max(15, int(min(face_bgr.shape[:2]) * 0.05))
    if ksize % 2 == 0:
        ksize += 1
    local_mean = cv2.GaussianBlur(L, (ksize, ksize), 0)
    diff = np.abs(L - local_mean)

    # Нормализуем и порогуем
    diff_norm = diff / (diff.max() + 1e-6)
    blemish = (diff_norm > 0.15).astype(np.float32)

    # Только на коже
    blemish = blemish * skin_mask

    # Дилятируем чтобы захватить края дефектов
    blemish_u8 = (blemish * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blemish_u8 = cv2.dilate(blemish_u8, kernel, iterations=2)

    return blemish_u8.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# CodeFormer — только на участках дефектов
# ══════════════════════════════════════════════════════════════════════════════

def run_codeformer_subtle(
    face_bgr: np.ndarray,
    net: torch.nn.Module,
    blemish_mask: np.ndarray,
    fidelity: float = CF_FIDELITY,
    blend: float = CF_BLEND,
) -> np.ndarray:
    """
    Запускает CodeFormer и смешивает результат ТОЛЬКО на участках дефектов.
    blend=0.15 → 15% CodeFormer + 85% оригинал.
    fidelity=0.92 → CodeFormer максимально сохраняет лицо.
    """
    h, w = face_bgr.shape[:2]

    # Resize до 512 для CodeFormer
    face_512 = cv2.resize(face_bgr, (_CF_SIZE, _CF_SIZE), interpolation=cv2.INTER_LANCZOS4)
    inp = cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = (inp - 0.5) / 0.5
    tensor = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = net(tensor, w=fidelity, adain=True)
        if isinstance(output, (list, tuple)):
            output = output[0]

    out_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = np.clip(out_np * 0.5 + 0.5, 0, 1)
    cf_bgr = cv2.cvtColor((out_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Resize обратно
    cf_bgr = cv2.resize(cf_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Resize blemish mask
    blemish_resized = cv2.resize(
        blemish_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
    )
    blemish_feathered = feather_mask(blemish_resized, radius=8)

    # Финальная маска = дефекты * blend_strength
    final_mask = blemish_feathered * blend

    # Смешиваем: только на дефектах, очень мягко
    result = blend_with_mask(cf_bgr, face_bgr, final_mask)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Frequency separation smoothing (текстура сохранена)
# ══════════════════════════════════════════════════════════════════════════════

def frequency_separation_smooth(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = SMOOTH_STRENGTH,
) -> np.ndarray:
    """
    Frequency separation:
    - low freq  = bilateral filter (тон, цвет)
    - high freq = оригинал - low (текстура)
    Сглаживаем ТОЛЬКО low freq внутри маски кожи.
    High freq (текстура) сохраняется полностью.
    """
    img = img_bgr.astype(np.float32)
    h, w = img.shape[:2]

    # Размер ядра зависит от разрешения
    radius = max(5, int(min(h, w) * 0.008))
    if radius % 2 == 0:
        radius += 1

    # Bilateral filter — сохраняет края
    low_freq = cv2.bilateralFilter(
        img_bgr, d=radius * 2 + 1, sigmaColor=35, sigmaSpace=35
    ).astype(np.float32)

    # Текстура = оригинал минус низкие частоты
    high_freq = img - low_freq

    # Сглаживаем low freq
    smoother = cv2.GaussianBlur(low_freq, (radius, radius), 0)

    # Применяем сглаживание только на коже
    m = skin_mask[..., np.newaxis]
    low_out = low_freq * (1.0 - m * strength) + smoother * (m * strength)

    # Рекомбинируем с текстурой
    result = low_out + high_freq
    return np.clip(result, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Dodge & Burn — локальный микроконтраст
# ══════════════════════════════════════════════════════════════════════════════

def dodge_burn(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    strength: float = DB_STRENGTH,
) -> np.ndarray:
    """
    Очень лёгкий dodge & burn только на коже.
    Работает в L*a*b* — не меняет цвет.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:, :, 0]

    ksize = max(5, int(min(img_bgr.shape[:2]) * 0.015))
    if ksize % 2 == 0:
        ksize += 1

    blurred_L = cv2.GaussianBlur(L, (ksize, ksize), 0)
    local_contrast = L - blurred_L

    # Усиливаем локальный контраст только на коже
    L_enhanced = L + local_contrast * skin_mask * strength
    lab[:, :, 0] = np.clip(L_enhanced, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# Face region helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pad_bbox(bbox: list, img_h: int, img_w: int, pad: float = 0.30) -> list:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    return [
        max(0, x1 - px), max(0, y1 - py),
        min(img_w, x2 + px), min(img_h, y2 + py),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class RetouchPipeline:
    def __init__(self):
        self.detector: Optional[FaceDetector] = None
        self.parser: Optional[FaceParser] = None
        self.codeformer: Optional[torch.nn.Module] = None

    def load_models(self):
        logger.info("Loading face detector…")
        self.detector = FaceDetector()
        logger.info("Loading face parser…")
        self.parser = FaceParser()
        logger.info("Loading CodeFormer…")
        self.codeformer = load_codeformer(weight_path=CF_WEIGHT, device=DEVICE)
        logger.info("All models loaded.")

    def run(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Полный pipeline ретуши.
        Входное и выходное изображение одинакового размера.
        Не меняет композицию, лицо, фон.
        """
        if not all([self.detector, self.parser, self.codeformer]):
            raise RuntimeError("Models not loaded.")

        original = img_bgr.copy()
        result = img_bgr.copy()
        h, w = img_bgr.shape[:2]
        logger.info("Input image: %dx%d", w, h)

        # ── 1. Detect faces ────────────────────────────────────────────────
        faces = self.detector.detect(img_bgr)

        if not faces:
            logger.info("No faces detected — applying mild global skin polish")
            skin_mask = self.parser.get_skin_mask(img_bgr)
            result = frequency_separation_smooth(result, skin_mask, strength=0.25)
            result = dodge_burn(result, skin_mask, strength=0.05)
            return result

        logger.info("%d face(s) detected", len(faces))

        for face in faces:
            bbox_padded = _pad_bbox(face["bbox"], h, w, pad=0.30)
            x1, y1, x2, y2 = bbox_padded
            face_crop = safe_crop(img_bgr, x1, y1, x2, y2)

            if face_crop.size == 0:
                continue

            fh, fw = face_crop.shape[:2]

            # ── 2. Skin mask ───────────────────────────────────────────────
            skin_mask = self.parser.get_skin_mask(face_crop)
            skin_feathered = feather_mask(
                dilate_mask(
                    (skin_mask * 255).astype(np.uint8), ksize=5, iterations=1
                ) / 255.0,
                radius=10,
            )

            # ── 3. Blemish detection ───────────────────────────────────────
            blemish_mask = detect_blemishes(face_crop, skin_mask)

            # ── 4. CodeFormer ТОЛЬКО на дефектах (blend=15%) ──────────────
            cf_result = run_codeformer_subtle(
                face_crop,
                self.codeformer,
                blemish_mask,
                fidelity=CF_FIDELITY,
                blend=CF_BLEND,
            )

            # ── 5. Frequency separation smooth ────────────────────────────
            smoothed = frequency_separation_smooth(
                cf_result, skin_feathered, strength=SMOOTH_STRENGTH
            )

            # ── 6. Dodge & burn ────────────────────────────────────────────
            retouched_face = dodge_burn(smoothed, skin_feathered, strength=DB_STRENGTH)

            # ── 7. Composite face back — мягкая вставка с feather ─────────
            # Маска для вставки лица в оригинал
            composite_mask = np.zeros((fh, fw), np.float32)
            pad_px = max(12, int(min(fh, fw) * 0.07))
            composite_mask[pad_px:-pad_px, pad_px:-pad_px] = 1.0
            composite_mask = feather_mask(composite_mask, radius=pad_px)

            blended = blend_with_mask(retouched_face, face_crop, composite_mask)

            # Вставляем обратно
            result[y1:y2, x1:x2] = blended

        logger.info("Output image: %dx%d (unchanged resolution)", w, h)
        return result
