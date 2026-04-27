"""
codeformer_loader.py — загружает CodeFormer.

Приоритет:
1. Официальный репозиторий /workspace/CodeFormer (если есть PYTHONPATH)
2. basicsr напрямую
3. Возвращает None — pipeline работает без CodeFormer
"""
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_WEIGHT = os.environ.get(
    "CODEFORMER_WEIGHT",
    str(Path.home() / ".cache/codeformer/codeformer.pth"),
)


def load_codeformer(
    weight_path: str = DEFAULT_WEIGHT,
    device: torch.device | None = None,
) -> nn.Module | None:
    """
    Возвращает nn.Module или None если загрузить не удалось.
    None = pipeline продолжает без CodeFormer (не падает).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_path = Path(weight_path)
    if not weight_path.exists():
        _download(weight_path)

    # ── попытка 1: официальный репо ────────────────────────────────────────
    cf_repo = "/workspace/CodeFormer"
    if os.path.isdir(cf_repo) and cf_repo not in sys.path:
        sys.path.insert(0, cf_repo)

    try:
        from basicsr.archs.codeformer_arch import CodeFormer  # type: ignore
        net = CodeFormer(
            dim_embd=512, n_head=8, n_layers=9,
            codebook_size=1024, latent_size=256,
            connect_list=["32","64","128","256"],
            fix_modules=["quantize","generator"],
        )
        ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        sd = ckpt.get("params_ema") or ckpt.get("params") or ckpt
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        net.load_state_dict(sd, strict=False)
        net.eval().to(device)
        logger.info("CodeFormer loaded ✓ device=%s", device)
        return net
    except Exception as exc:
        logger.warning("CodeFormer load failed: %s — будет работать без него", exc)
        return None


def _download(dest: Path):
    import urllib.request
    URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading CodeFormer → %s", dest)
    urllib.request.urlretrieve(URL, str(dest))
