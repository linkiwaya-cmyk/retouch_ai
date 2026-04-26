"""
CodeFormer loader — полностью автономный, без basicsr.archs.

Архитектура CodeFormer определена инлайн по оригинальному коду:
https://github.com/sczhou/CodeFormer/blob/master/basicsr/archs/codeformer_arch.py

Не требует basicsr.archs, ComfyUI, diffusion моделей.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_WEIGHT_PATH = os.environ.get(
    "CODEFORMER_WEIGHT",
    str(Path.home() / ".cache" / "codeformer" / "codeformer.pth"),
)


# ══════════════════════════════════════════════════════════════════════════════
# Inline CodeFormer Architecture
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        w = torch.bmm(q, k) * (C ** -0.5)
        w = F.softmax(w, dim=2)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)
        h = torch.bmm(w, v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class VQEncoder(nn.Module):
    def __init__(self, in_channels, nf, ch_mult, num_res_blocks, resolution, attn_resolutions, z_channels):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, nf, 3, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i, mult in enumerate(ch_mult):
            block_in = nf * in_ch_mult[i]
            block_out = nf * mult
            layers = [ResBlock(block_in if j == 0 else block_out, block_out) for j in range(num_res_blocks)]
            if curr_res in attn_resolutions:
                layers.append(AttnBlock(block_out))
            down = nn.Module()
            down.block = nn.ModuleList(layers)
            down.downsample = Downsample(block_out) if i < len(ch_mult) - 1 else nn.Identity()
            self.down.append(down)
            if i < len(ch_mult) - 1:
                curr_res //= 2
        block_out_final = nf * ch_mult[-1]
        self.mid_block_1 = ResBlock(block_out_final)
        self.mid_attn = AttnBlock(block_out_final)
        self.mid_block_2 = ResBlock(block_out_final)
        self.norm_out = nn.GroupNorm(32, block_out_final, eps=1e-6)
        self.conv_out = nn.Conv2d(block_out_final, z_channels, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for down in self.down:
            for layer in down.block:
                h = layer(h)
            h = down.downsample(h)
        h = self.mid_block_2(self.mid_attn(self.mid_block_1(h)))
        return self.conv_out(F.silu(self.norm_out(h)))


class VQDecoder(nn.Module):
    def __init__(self, z_channels, nf, ch_mult, num_res_blocks, resolution, attn_resolutions, out_channels):
        super().__init__()
        block_in = nf * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, padding=1)
        self.mid_block_1 = ResBlock(block_in)
        self.mid_attn = AttnBlock(block_in)
        self.mid_block_2 = ResBlock(block_in)
        curr_res = resolution // (2 ** (len(ch_mult) - 1))
        self.up = nn.ModuleList()
        for i in reversed(range(len(ch_mult))):
            block_out = nf * ch_mult[i]
            layers = [ResBlock(block_in if j == 0 else block_out, block_out) for j in range(num_res_blocks + 1)]
            if curr_res in attn_resolutions:
                layers.append(AttnBlock(block_out))
            up = nn.Module()
            up.block = nn.ModuleList(layers)
            up.upsample = Upsample(block_out) if i > 0 else nn.Identity()
            self.up.insert(0, up)
            block_in = block_out
            if i > 0:
                curr_res *= 2
        self.norm_out = nn.GroupNorm(32, block_out, eps=1e-6)
        self.conv_out = nn.Conv2d(block_out, out_channels, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid_block_2(self.mid_attn(self.mid_block_1(h)))
        for up in reversed(self.up):
            for layer in up.block:
                h = layer(h)
            h = up.upsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.beta = beta
        self.embedding = nn.Embedding(codebook_size, emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z):
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z_perm.view(-1, self.emb_dim)
        d = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = d.argmin(1)
        z_q = self.embedding(indices).view(z_perm.shape)
        loss = self.beta * (z_q.detach() - z_perm).pow(2).mean() + (z_q - z_perm.detach()).pow(2).mean()
        z_q = z_perm + (z_q - z_perm).detach()
        return z_q.permute(0, 3, 1, 2).contiguous(), loss, indices.view(z_perm.shape[:-1])


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = tgt2 if query_pos is None else tgt2 + query_pos
        tgt2 = self.self_attn(q, k, tgt2)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt2))))
        return tgt + self.dropout(tgt2)


class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2 * in_ch, out_ch)
        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.LeakyReLU(0.2, True), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.LeakyReLU(0.2, True), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        return dec_feat + w * (dec_feat * scale + shift)


class CodeFormer(nn.Module):
    """
    CodeFormer — inline implementation matching official v0.1.0 checkpoint.
    No basicsr registry dependency.
    """

    def __init__(
        self,
        dim_embd=512,
        n_head=8,
        n_layers=9,
        codebook_size=1024,
        latent_size=256,
        connect_list=("32", "64", "128", "256"),
        fix_modules=("quantize", "decoder"),
    ):
        super().__init__()
        self.connect_list = list(connect_list)
        self.n_layers = n_layers
        self.dim_embd = dim_embd

        self.position_emb = nn.Parameter(torch.zeros(latent_size, dim_embd))
        self.feat_emb = nn.Linear(256, dim_embd)

        self.ft_layers = nn.ModuleList([
            TransformerSALayer(dim_embd, nhead=n_head, dim_mlp=dim_embd * 2)
            for _ in range(n_layers)
        ])
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False),
        )

        self.channels = {
            '16': 512, '32': 512, '64': 256, '128': 128, '256': 128, '512': 64,
        }

        self.fuse_encoder_block = nn.ModuleDict({
            res: Fuse_sft_block(ch, ch) for res, ch in self.channels.items()
        })
        self.fuse_generator_block = nn.ModuleDict({
            res: Fuse_sft_block(ch, ch) for res, ch in self.channels.items()
        })

        self.quantize = VectorQuantizer(codebook_size, 256, beta=0.25)

        self.encoder = VQEncoder(
            in_channels=3, nf=64, ch_mult=[1, 2, 2, 4, 4, 8],
            num_res_blocks=2, resolution=512,
            attn_resolutions=[16], z_channels=256,
        )
        self.decoder = VQDecoder(
            z_channels=256, nf=64, ch_mult=[1, 2, 2, 4, 4, 8],
            num_res_blocks=2, resolution=512,
            attn_resolutions=[16], out_channels=3,
        )

        if fix_modules:
            for mod_name in fix_modules:
                mod = getattr(self, mod_name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x, w=0.5, adain=True):
        # 1. Encode to latent
        enc_feat = self.encoder(x)

        # 2. VQ quantize
        z_q, _, token_idx = self.quantize(enc_feat)

        # 3. Transformer refinement
        token_idx_flat = token_idx.view(token_idx.shape[0], -1)
        z_flat = self.quantize.embedding(token_idx_flat)          # (B, L, 256)
        query = self.feat_emb(z_flat)                              # (B, L, dim_embd)
        pos = self.position_emb[:query.shape[1]]                   # (L, dim_embd)
        query = (query + pos).permute(1, 0, 2)                     # (L, B, dim_embd)

        for layer in self.ft_layers:
            query = layer(query)

        query = query.permute(1, 0, 2)                             # (B, L, dim_embd)
        logits = self.idx_pred_layer(query)                        # (B, L, codebook_size)
        soft_one_hot = F.softmax(logits, dim=2)

        # (B, L, 256) via weighted sum over codebook
        pred_emb = soft_one_hot @ self.quantize.embedding.weight   # (B, L, 256)

        # 4. Reshape to spatial
        h_size = int(math.sqrt(pred_emb.shape[1]))
        pred_emb = pred_emb.permute(0, 2, 1).reshape(
            pred_emb.shape[0], 256, h_size, h_size
        )

        # 5. Fidelity blend: w=0 → full enhancement, w=1 → original reconstruction
        pred_emb = (1 - w) * pred_emb + w * z_q

        # 6. Decode
        out = self.decoder(pred_emb)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def load_codeformer(
    weight_path: str = DEFAULT_WEIGHT_PATH,
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_path = Path(weight_path)
    if not weight_path.exists():
        _download_weights(weight_path)

    net = CodeFormer(
        dim_embd=512,
        n_head=8,
        n_layers=9,
        codebook_size=1024,
        latent_size=256,
        connect_list=["32", "64", "128", "256"],
        fix_modules=["quantize", "decoder"],
    )

    checkpoint = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint.get("params_ema")
        or checkpoint.get("params")
        or checkpoint
    )
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:5])

    net.eval().to(device)
    logger.info("CodeFormer loaded ✓  device=%s", device)
    return net


def _download_weights(dest: Path) -> None:
    import urllib.request
    URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading CodeFormer weights → %s", dest)
    urllib.request.urlretrieve(URL, str(dest))
    logger.info("Download complete.")
