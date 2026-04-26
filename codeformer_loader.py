from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_WEIGHT_PATH = os.environ.get(
    "CODEFORMER_WEIGHT",
    str(Path.home() / ".cache/codeformer/codeformer.pth"),
)

# Добавляем CodeFormer в путь
CF_REPO = os.environ.get("CODEFORMER_REPO", "/workspace/CodeFormer")
if CF_REPO not in sys.path:
    sys.path.insert(0, CF_REPO)


def load_codeformer(weight_path: str = DEFAULT_WEIGHT_PATH, device=None) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_path = Path(weight_path)
    if not weight_path.exists():
        _download_weights(weight_path)

    # Импорт из официального репозитория
    from basicsr.archs.codeformer_arch import CodeFormer

    net = CodeFormer(
        dim_embd=512,
        n_head=8,
        n_layers=9,
        codebook_size=1024,
        latent_size=256,
        connect_list=["32", "64", "128", "256"],
        fix_modules=["quantize", "generator"],
    )

    checkpoint = torch.load(str(weight_path), map_location="cpu")
    state_dict = (
        checkpoint.get("params_ema")
        or checkpoint.get("params")
        or checkpoint
    )
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    net.load_state_dict(state_dict, strict=True)
    net.eval().to(device)
    logger.info("CodeFormer loaded ✓  device=%s", device)
    return net


def _download_weights(dest: Path) -> None:
    import urllib.request

    URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading CodeFormer weights → %s", dest)
    urllib.request.urlretrieve(URL, str(dest))
