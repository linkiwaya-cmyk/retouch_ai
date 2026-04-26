"""
RetouchPipeline

Pipeline stages:
  1. Face detection (InsightFace → RetinaFace fallback)
  2. Face region crop with padding
  3. CodeFormer face restoration (remove blemishes, restore details)
  4. Face parsing → skin mask (BiSeNet)
  5. Selective skin smoothing (frequency separation — no blur on edges)
  6. Dodge & burn (local luminosity micro-contrast)
  7. Seamless compositing back to original image
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
from PIL import Image

from codeformer_loader import load_codeformer
from utils import (
    bgr_to_rgb,
    rgb_to_bgr,
    blend_with_mask,
    feather_mask,
    dilate_mask,
    safe_crop,
)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Env-configurable paths
CF_WEIGHT = os.environ.get("CODEFORMER_WEIGHT", str(Path.home() / ".cache/codeformer/codeformer.pth"))
PARSING_WEIGHT = os.environ.get(
    "BISENET_WEIGHT",
    str(Path.home() / ".cache/facexlib/parsing_bisenet.pth"),
)

# CodeFormer fidelity weight: 0.0 = max enhancement, 1.0 = max fidelity
CF_FIDELITY = float(os.environ.get("CF_FIDELITY", "0.7"))

# Face detection confidence threshold
FACE_CONF = float(os.environ.get("FACE_CONF", "0.5"))


# ══════════════════════════════════════════════════════════════════════════════
# Face detector wrapper
# ══════════════════════════════════════════════════════════════════════════════

class FaceDetector:
    """Wraps InsightFace with RetinaFace as fallback."""

    def __init__(self):
        self._insight = None
        self._retina = None
        self._load()

    def _load(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
            self._insight = app
            logger.info("FaceDetector: using InsightFace buffalo_l")
        except Exception as exc:
            logger.warning("InsightFace unavailable (%s), falling back to RetinaFace.", exc)
            try:
                from facexlib.detection import init_detection_model
                self._retina = init_detection_model("retinaface_resnet50", half=False, device=str(DEVICE))
                logger.info("FaceDetector: using RetinaFace (facexlib)")
            except Exception as exc2:
                logger.error("RetinaFace also unavailable: %s. No face detection.", exc2)

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        """
        Returns list of dicts: {'bbox': [x1,y1,x2,y2], 'score': float}
        Sorted by area descending.
        """
        if self._insight is not None:
            return self._detect_insightface(img_bgr)
        if self._retina is not None:
            return self._detect_retina(img_bgr)
        return []

    def _detect_insightface(self, img_bgr: np.ndarray) -> list[dict]:
        faces = self._insight.get(img_bgr)
        results = []
        for f in faces:
            if f.det_score < FACE_CONF:
                continue
            x1, y1, x2, y2 = f.bbox.astype(int)
            results.append({"bbox": [x1, y1, x2, y2], "score": float(f.det_score)})
        results.sort(key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]), reverse=True)
        return results

    def _detect_retina(self, img_bgr: np.ndarray) -> list[dict]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            bboxes = self._retina.detect_faces(img_rgb, conf_threshold=FACE_CONF)
        results = []
        for b in bboxes:
            x1, y1, x2, y2, score = int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4])
            results.append({"bbox": [x1, y1, x2, y2], "score": score})
        results.sort(key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]), reverse=True)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Face parser wrapper (BiSeNet → skin mask)
# ══════════════════════════════════════════════════════════════════════════════

# BiSeNet face-parsing label indices that correspond to skin
_SKIN_LABELS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13}


class FaceParser:
    """Runs BiSeNet face parsing to produce a skin mask."""

    def __init__(self):
        self._net = None
        self._load()

    def _load(self):
        try:
            from facexlib.parsing import init_parsing_model
            self._net = init_parsing_model(
                model_name="bisenet",
                device=str(DEVICE),
                model_rootpath=str(Path(PARSING_WEIGHT).parent),
            )
            self._net.eval()
            logger.info("FaceParser: BiSeNet loaded.")
        except Exception as exc:
            logger.warning("FaceParser: BiSeNet unavailable (%s). Skin mask will cover full face.", exc)

    def get_skin_mask(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Returns float32 mask [0..1], same HxW as face_bgr.
        1 = skin, 0 = non-skin.
        """
        h, w = face_bgr.shape[:2]

        if self._net is None:
            return _fallback_skin_mask(face_bgr)

        # BiSeNet expects 512×512 RGB float tensor normalised
        inp = cv2.resize(face_bgr, (512, 512))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = (inp - mean) / std
        tensor = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = self._net(tensor)[0]  # (1, num_classes, 512, 512)
        seg = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)  # (512, 512)

        # Build skin mask from selected labels
        skin = np.zeros((512, 512), dtype=np.uint8)
        for lbl in _SKIN_LABELS:
            skin[seg == lbl] = 255

        # Remove eyes, mouth from skin mask
        for non_skin in [11, 12, 17, 18]:  # l_eye, r_eye, u_lip, l_lip
            skin[seg == non_skin] = 0

        skin = cv2.resize(skin, (w, h), interpolation=cv2.INTER_NEAREST)
        return skin.astype(np.float32) / 255.0


def _fallback_skin_mask(face_bgr: np.ndarray) -> np.ndarray:
    """HSV-based rough skin mask when BiSeNet is unavailable."""
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 170, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    return mask.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# Skin smoothing — frequency separation (no blur artefacts)
# ══════════════════════════════════════════════════════════════════════════════

def frequency_separation_smooth(img_bgr: np.ndarray, skin_mask: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """
    Frequency separation skin smoothing.
    - Low freq  = Gaussian-blurred version (colour/tone)
    - High freq = original - low (texture detail)
    Only the low-frequency layer is touched inside the skin mask.
    High-frequency (texture) is preserved 100%.
    """
    img = img_bgr.astype(np.float32)
    radius = max(3, int(min(img.shape[:2]) * 0.015))
    if radius % 2 == 0:
        radius += 1

    # Surface-blur (bilateral) preserves edges better than Gaussian
    low_freq = cv2.bilateralFilter(img_bgr, d=radius*2+1, sigmaColor=40, sigmaSpace=40).astype(np.float32)
    high_freq = img - low_freq  # texture layer (can be negative)

    # Smooth only the low-frequency layer inside the skin region
    smoother = cv2.GaussianBlur(low_freq, (radius, radius), 0)
    # Blend: strength controls how much smoothing is applied inside skin
    low_freq_out = img.copy()
    m = skin_mask[..., np.newaxis]
    low_freq_out = low_freq * (1 - m * strength) + smoother * (m * strength)

    result = low_freq_out + high_freq  # recombine with texture
    return np.clip(result, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Dodge & Burn — local luminosity micro-contrast
# ══════════════════════════════════════════════════════════════════════════════

def dodge_burn(img_bgr: np.ndarray, skin_mask: np.ndarray, strength: float = 0.12) -> np.ndarray:
    """
    Subtle dodge & burn applied only on skin.
    Uses Laplacian-of-Gaussian to find highlights/shadows locally,
    then nudges them slightly (not a filter, not a blur).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:, :, 0]  # [0..255]

    # Local contrast via unsharp-mask on L channel
    ksize = max(3, int(min(img_bgr.shape[:2]) * 0.02))
    if ksize % 2 == 0:
        ksize += 1
    blurred_L = cv2.GaussianBlur(L, (ksize, ksize), 0)
    local_contrast = L - blurred_L  # positive = highlight, negative = shadow

    # Amplify subtle details only inside skin region
    L_enhanced = L + local_contrast * skin_mask * strength
    L_enhanced = np.clip(L_enhanced, 0, 255)

    lab[:, :, 0] = L_enhanced
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CodeFormer face restoration
# ══════════════════════════════════════════════════════════════════════════════

_CF_SIZE = 512  # CodeFormer expects 512×512


def run_codeformer(face_bgr: np.ndarray, net: torch.nn.Module, fidelity: float = CF_FIDELITY) -> np.ndarray:
    """
    Run CodeFormer on a face crop.
    Input/output: BGR uint8, any size → resized internally → resized back.
    """
    h, w = face_bgr.shape[:2]
    face_512 = cv2.resize(face_bgr, (_CF_SIZE, _CF_SIZE), interpolation=cv2.INTER_LANCZOS4)

    # BGR → RGB → [0,1] → tensor
    inp = cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = (inp - 0.5) / 0.5  # normalize to [-1, 1]
    tensor = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = net(tensor, w=fidelity, adain=True)
        if isinstance(output, (list, tuple)):
            output = output[0]  # some versions return (restored, codebook_loss)

    # Tensor → BGR uint8
    out_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = np.clip(out_np * 0.5 + 0.5, 0, 1)  # denormalize
    out_bgr = cv2.cvtColor((out_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Resize back to original face dimensions
    return cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)


# ══════════════════════════════════════════════════════════════════════════════
# Face region helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pad_bbox(bbox: list[int], img_h: int, img_w: int, pad_ratio: float = 0.35) -> list[int]:
    """Expand bounding box by pad_ratio to include chin, forehead, cheeks."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    return [
        max(0, x1 - px),
        max(0, y1 - py),
        min(img_w, x2 + px),
        min(img_h, y2 + py),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

class RetouchPipeline:
    def __init__(self):
        self.detector: Optional[FaceDetector] = None
        self.parser: Optional[FaceParser] = None
        self.codeformer: Optional[torch.nn.Module] = None

    def load_models(self):
        logger.info("Loading face detector …")
        self.detector = FaceDetector()
        logger.info("Loading face parser …")
        self.parser = FaceParser()
        logger.info("Loading CodeFormer …")
        self.codeformer = load_codeformer(device=DEVICE)
        logger.info("All models loaded.")

    def run(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Full retouching pipeline.
        Returns BGR uint8 image of same size as input.
        """
        if self.detector is None or self.parser is None or self.codeformer is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        h, w = img_bgr.shape[:2]
        result = img_bgr.copy()

        # ── 1. Face detection ──────────────────────────────────────────────
        faces = self.detector.detect(img_bgr)
        if not faces:
            logger.warning("No faces detected — applying global skin smoothing only.")
            # Apply mild whole-image polish without face-specific processing
            skin_mask = _fallback_skin_mask(img_bgr)
            result = frequency_separation_smooth(result, skin_mask, strength=0.3)
            result = dodge_burn(result, skin_mask, strength=0.08)
            return result

        logger.info("%d face(s) detected.", len(faces))

        for face in faces:
            bbox_padded = _pad_bbox(face["bbox"], h, w, pad_ratio=0.35)
            x1, y1, x2, y2 = bbox_padded
            face_crop = safe_crop(img_bgr, x1, y1, x2, y2)

            if face_crop.size == 0:
                continue

            fh, fw = face_crop.shape[:2]

            # ── 2. CodeFormer restoration ──────────────────────────────────
            restored = run_codeformer(face_crop, self.codeformer, fidelity=CF_FIDELITY)

            # ── 3. Face parsing → skin mask ────────────────────────────────
            skin_mask = self.parser.get_skin_mask(restored)
            skin_mask_smooth = feather_mask(dilate_mask(skin_mask.astype(np.uint8) * 255, ksize=7, iterations=1) / 255.0, radius=12)

            # ── 4. Frequency-separation smoothing (texture-safe) ───────────
            smoothed = frequency_separation_smooth(restored, skin_mask_smooth, strength=0.55)

            # ── 5. Dodge & burn ────────────────────────────────────────────
            db = dodge_burn(smoothed, skin_mask_smooth, strength=0.12)

            # ── 6. Blend processed face back into original with feather mask ─
            # Build a feathered face mask for seamless compositing
            composite_mask = np.zeros((fh, fw), dtype=np.float32)
            pad = max(10, int(min(fh, fw) * 0.08))
            composite_mask[pad:-pad, pad:-pad] = 1.0
            composite_mask = feather_mask(composite_mask, radius=pad)

            blended_face = blend_with_mask(db, face_crop, composite_mask)

            # ── 7. Paste back ──────────────────────────────────────────────
            result[y1:y2, x1:x2] = blended_face

        return result
