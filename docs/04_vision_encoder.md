# 4 · The vision encoder (SigLIP-style ViT)

**Code:** [`nitrogen/models/vision_encoder.py`](../nitrogen/models/vision_encoder.py)

## Job

Turn one **256×256 RGB frame** into a sequence of **image tokens** the action
head can attend to. NitroGen uses a pretrained **SigLIP-2 Vision Transformer**;
this repo ships a small self-contained ViT with the same interface so the
pipeline runs without downloading large weights.

## Why a Vision Transformer (not a CNN)?

The action head is a Transformer that reads the image via **cross-attention**
([ch 6](06_action_head.md)). For that to work, the image has to be a *sequence of
tokens*, each corresponding to a region of the frame. Then the action head can
learn to attend to whatever is relevant — "where's the enemy", "where's the
platform edge", "what does the health bar say" — one region at a time. A ViT
produces exactly that token sequence; it is also what the SigLIP family uses and
scales cleanly.

## How it works

```
frame (B, 3, 256, 256) in [0, 1]
        │  normalize (mean/std)
        │  Conv2d(stride=patch)         ← "patchify": each 16×16 patch → 1 token
        ▼
patches (B, 256, width)                 ← (256/16)² = 256 tokens
        │  + learned positional embedding
        │  N × Transformer encoder blocks (pre-norm: self-attn + MLP)
        ▼
image tokens (B, 256, width)            ← what the action head cross-attends to
```

Key points, each mirrored in the code:

- **Patchify with a strided convolution.** `nn.Conv2d(3, width, kernel=16,
  stride=16)` slices the image into non-overlapping 16×16 patches and linearly
  projects each into a `width`-dim token — the standard ViT patch embedding.
- **Learned positional embeddings**, no `[CLS]` token. SigLIP encoders keep *all*
  patch tokens (there's no single pooled vector); the action head wants the full
  spatial set so it can attend region-by-region.
- **Pre-norm Transformer blocks** with fused scaled-dot-product attention
  (`F.scaled_dot_product_attention`, which uses FlashAttention when available).
- **CoordConv input.** Before patchifying we append two channels — normalized `x`
  and `y` coordinate maps — to the RGB frame. Convolutions and ViTs are
  translation-equivariant and notoriously bad at reporting the *absolute
  position* of a feature, yet our control tasks are exactly "the bright object is
  *over there* → push toward/away from it." Handing the network explicit
  coordinates (Liu et al., 2018) makes that localization easy. At the real
  NitroGen scale a pretrained SigLIP-2 encoder localizes fine without this; at our
  tiny from-scratch scale it is the difference between a policy that plays and one
  that idles.

```python
from nitrogen.models.vision_encoder import VisionEncoder, VisionConfig
enc = VisionEncoder(VisionConfig(width=384, depth=6))
tokens = enc(frames)          # (B, num_patches, width)
```

## Frame-only conditioning

NitroGen conditions on a **single** frame — no stack of past frames, no recurrent
state. That is a deliberate, slightly surprising choice: many control policies use
history to infer velocity/intent. NitroGen offloads temporal reasoning to the
**action head**, which predicts a whole *chunk* of future actions from that one
frame ([ch 6](06_action_head.md), [ch 8](08_inference.md)). Fewer moving parts,
and it forces the encoder to extract everything decision-relevant from the current
image.

Continue to [**5 · Flow matching →**](05_flow_matching.md)
