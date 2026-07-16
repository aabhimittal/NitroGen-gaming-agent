"""Action head — a Diffusion Transformer ("action expert") that predicts velocity.

This is the heart of NitroGen's generative policy. Given

    * a *noised* action chunk ``x_t``            shape (B, T, action_dim)
    * the flow time ``t``                        shape (B, 1, 1)
    * image tokens from the vision encoder       shape (B, N_img, width)

it predicts the flow-matching **velocity** for every action in the chunk,
shape ``(B, T, action_dim)``. Integrating that velocity (see
:class:`~nitrogen.models.flow_matching.FlowMatching`) turns noise into actions.

Design (a DiT with cross-attention, mirroring GR00T's "action expert"):

    x_t --linear--> action tokens (+ positional emb over the T chunk steps)
        │
        ├─ self-attention   : action steps talk to each other (temporal coherence)
        ├─ cross-attention  : action steps read the image tokens (what's on screen)
        └─ MLP
        every sublayer is modulated by the flow time ``t`` via **AdaLN-Zero**.

AdaLN-Zero (from the DiT paper) injects the scalar conditioning ``t`` by
predicting per-layer LayerNorm shift/scale plus a residual "gate" that starts at
zero, so at init every block is the identity and training is stable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ActionHeadConfig:
    action_dim: int = 18
    chunk_size: int = 16        # T: number of future actions generated per frame
    width: int = 384            # hidden dim (matches vision width for simplicity)
    context_width: int = 384    # width of the incoming image tokens
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    time_embed_dim: int = 384


# ---------------------------------------------------------------------------
# Flow-time embedding
# ---------------------------------------------------------------------------
class TimestepEmbedding(nn.Module):
    """Embed a scalar flow-time ``t in [0, 1]`` with a sinusoidal code + MLP."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) -> (B, dim). Standard transformer sinusoidal embedding.
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:  # pad if odd
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t.view(-1)))


# ---------------------------------------------------------------------------
# Attention modules (self + cross)
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class CrossAttention(nn.Module):
    """Queries from the action tokens, keys/values from the image context."""

    def __init__(self, dim: int, context_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(context_dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        M = context.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN affine modulation: ``x * (1 + scale) + shift`` (broadcast over T)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """One DiT block: AdaLN-Zero self-attn, cross-attn, and MLP, all gated by ``t``."""

    def __init__(self, cfg: ActionHeadConfig):
        super().__init__()
        d = cfg.width
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.self_attn = SelfAttention(d, cfg.num_heads)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(d, cfg.context_width, cfg.num_heads)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        hidden = int(d * cfg.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

        # AdaLN modulation from the flow time. Self-attn and MLP use AdaLN-**Zero**
        # (a zero-init residual gate) for stable training. Cross-attention to the
        # image is deliberately **always-on** (no zero gate): it is the only path
        # carrying *spatial* image information ("where is the object"), and if it
        # were zero-gated the policy would sit at the image-independent marginal
        # and never learn to look at the frame. So we emit shift/scale/gate for
        # self-attn (3) + shift/scale for cross-attn (2) + shift/scale/gate for
        # MLP (3) = 8 * d values.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cfg.time_embed_dim, 8 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        (sa_shift, sa_scale, sa_gate,
         ca_shift, ca_scale,
         mlp_shift, mlp_scale, mlp_gate) = self.ada(c).chunk(8, dim=-1)

        x = x + sa_gate.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), sa_shift, sa_scale))
        # Always-on cross-attention (no gate) so the image is read from step 0.
        x = x + self.cross_attn(modulate(self.norm2(x), ca_shift, ca_scale), context)
        x = x + mlp_gate.unsqueeze(1) * self.mlp(modulate(self.norm3(x), mlp_shift, mlp_scale))
        return x


class ActionHead(nn.Module):
    """Predicts the flow-matching velocity for a noised action chunk.

    ``forward(x_t, t, image_tokens)`` -> velocity of shape ``(B, T, action_dim)``.
    """

    def __init__(self, cfg: ActionHeadConfig | None = None):
        super().__init__()
        self.cfg = cfg or ActionHeadConfig()
        c = self.cfg

        self.in_proj = nn.Linear(c.action_dim, c.width)
        # Learned positional embedding over the T chunk timesteps.
        self.pos_embed = nn.Parameter(torch.zeros(1, c.chunk_size, c.width))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_embed = TimestepEmbedding(c.time_embed_dim)
        self.blocks = nn.ModuleList([DiTBlock(c) for _ in range(c.depth)])

        # Final AdaLN-Zero + projection back to action space.
        self.final_norm = nn.LayerNorm(c.width, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(c.time_embed_dim, 2 * c.width))
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        self.out_proj = nn.Linear(c.width, c.action_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, image_tokens: torch.Tensor
    ) -> torch.Tensor:
        # x_t: (B, T, action_dim); t: (B, 1, 1) or (B,); image_tokens: (B, N, ctx_w)
        c = self.time_embed(t)                    # (B, time_embed_dim)
        x = self.in_proj(x_t) + self.pos_embed    # (B, T, width)
        for block in self.blocks:
            x = block(x, image_tokens, c)
        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        return self.out_proj(x)                   # (B, T, action_dim)
