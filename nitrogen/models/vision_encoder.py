"""Vision encoder — a compact SigLIP-2-style Vision Transformer.

NitroGen conditions entirely on a **single** 256x256 RGB frame (no frame stack,
no history). That single frame is turned into a set of *patch tokens* by a
Vision Transformer, and those tokens are what the action head cross-attends to.
The real model uses a pretrained SigLIP-2 ViT; here we implement a small,
self-contained ViT with the same interface so the whole pipeline is runnable
without downloading multi-hundred-MB weights.

Why a ViT (and not a CNN)?
    * The action head is a Transformer, so exposing the image as a *sequence of
      tokens* lets it attend to specific regions ("where is the enemy / the
      platform edge / the health bar") via cross-attention.
    * ViTs scale cleanly and are what the SigLIP family uses.

Output: ``(B, num_patches, width)`` patch tokens. With a 256px image and a
16px patch that is ``(256/16)^2 = 256`` tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class VisionConfig:
    image_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    width: int = 384        # token / embedding dimension
    depth: int = 6          # number of transformer blocks
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @property
    def num_patches(self) -> int:
        g = self.image_size // self.patch_size
        return g * g


class Attention(nn.Module):
    """Standard multi-head self-attention (pre-norm block uses it)."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # PyTorch fused scaled-dot-product attention (flash when available).
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block (attention + MLP with residuals)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """Turn an RGB frame into a sequence of patch tokens.

    Args mirror a SigLIP/ViT encoder. ``forward`` accepts pixel values in
    ``[0, 1]`` (float) of shape ``(B, 3, H, W)`` and returns patch tokens of
    shape ``(B, num_patches, width)``.
    """

    def __init__(self, cfg: VisionConfig | None = None):
        super().__init__()
        self.cfg = cfg or VisionConfig()
        c = self.cfg

        # Patchify with a strided conv: each patch -> one token vector. We prepend
        # two **CoordConv** channels (normalized x and y coordinate maps) to the
        # RGB input. Plain convolutions/ViTs are translation-equivariant and famously
        # poor at reporting the *absolute position* of a feature — but our control
        # tasks are exactly "where is the bright object → push toward/away from it",
        # so we hand the network the coordinates explicitly. (Liu et al., 2018.)
        self.patch_embed = nn.Conv2d(
            c.in_channels + 2, c.width, kernel_size=c.patch_size, stride=c.patch_size
        )
        # Learned positional embedding (SigLIP uses learned, no [CLS] token).
        self.pos_embed = nn.Parameter(torch.zeros(1, c.num_patches, c.width))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(c.width, c.num_heads, c.mlp_ratio, c.dropout)
                for _ in range(c.depth)
            ]
        )
        self.norm = nn.LayerNorm(c.width)

        # ImageNet normalization constants (SigLIP-style preprocessing).
        self.register_buffer("pixel_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    @property
    def width(self) -> int:
        return self.cfg.width

    @property
    def num_patches(self) -> int:
        return self.cfg.num_patches

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """``pixels``: (B, 3, H, W) in [0, 1]. Returns (B, num_patches, width).

        Frames whose resolution differs from ``cfg.image_size`` are resized with
        bilinear interpolation. This decouples *environment* rendering resolution
        (always 256 for nice-looking demos) from the *model* input resolution,
        so you can train at a smaller, CPU-friendly size without re-rendering.
        """
        if pixels.shape[-1] != self.cfg.image_size or pixels.shape[-2] != self.cfg.image_size:
            pixels = torch.nn.functional.interpolate(
                pixels, size=(self.cfg.image_size, self.cfg.image_size),
                mode="bilinear", align_corners=False,
            )
        x = (pixels - self.pixel_mean) / self.pixel_std
        # Append normalized coordinate channels (CoordConv).
        b, _, h, w = x.shape
        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        x = torch.cat([x, xs, ys], dim=1)    # (B, 5, H, W)
        x = self.patch_embed(x)              # (B, width, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)     # (B, num_patches, width)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
