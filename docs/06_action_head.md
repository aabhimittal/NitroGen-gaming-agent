# 6 · The action head — a Diffusion Transformer ("action expert")

**Code:** [`nitrogen/models/action_head.py`](../nitrogen/models/action_head.py)

This is the network that predicts the flow-matching velocity `v_θ`. NitroGen
builds on **GR00T N1**'s "action expert": a **Diffusion Transformer (DiT)** that
generates the action chunk while cross-attending to the image tokens. It is the
piece that ties the vision encoder ([ch 4](04_vision_encoder.md)) and flow
matching ([ch 5](05_flow_matching.md)) together.

## Inputs and output

```
x_t          : (B, T, 18)          noised action chunk (T = 16 steps)
t            : (B, 1, 1)           flow time in [0, 1]
image_tokens : (B, N_img, width)   from the vision encoder
        │
        ▼   ActionHead (DiT)
velocity     : (B, T, 18)          predicted dx_t/dt for every action in the chunk
```

## Anatomy of the DiT

```
x_t ──Linear──► action tokens (B, T, width)  + learned positional emb over T steps
      │
      │   for each of D blocks:
      │     ├─ self-attention   : the 16 action steps attend to EACH OTHER
      │     │                     → temporal coherence within the chunk
      │     ├─ cross-attention  : action steps attend to the IMAGE tokens
      │     │                     → "what's on screen right now"
      │     └─ MLP
      │     (every sublayer modulated by the flow time t via AdaLN-Zero)
      ▼
final AdaLN + Linear ──► velocity (B, T, 18)
```

Three ideas make this work:

### 1. Self-attention over the chunk → temporal coherence
The 16 action steps are tokens that attend to one another. Generating them
*jointly* (rather than 16 independent predictions) is what keeps the chunk
self-consistent: a commitment to "dodge left" at step 0 is visible to steps 1–15,
so they continue the same maneuver instead of each independently sampling.

### 2. Cross-attention to image tokens → grounding in the frame
Each action query reads the image token sequence, so different parts of the chunk
can look at different parts of the screen. This is why the encoder emits a full
token grid ([ch 4](04_vision_encoder.md)) rather than a single pooled vector — the
head wants to attend region-by-region.

### 3. AdaLN-Zero → clean injection of the flow time `t`
The scalar flow time `t` conditions the whole network through **Adaptive
LayerNorm - Zero** (from the DiT paper). A small MLP embeds `t` (via a sinusoidal
code) and predicts, per block, LayerNorm **shift** and **scale** plus a residual
**gate**:

```
h = h + gate · sublayer( LN(h) · (1 + scale) + shift )
```

The projection producing (shift, scale, gate) is **initialized to zero**, so at
the start of training `gate = 0` and every block is the identity — the network
starts as a no-op and learns to deviate. This "-Zero" init is what makes deep
DiTs train stably. The final output projection is also zero-initialized, so the
model's initial velocity prediction is exactly zero (a well-behaved starting
point for the ODE).

## Putting it together

```python
from nitrogen.models.action_head import ActionHead, ActionHeadConfig
head = ActionHead(ActionHeadConfig(action_dim=18, chunk_size=16, width=384))
v = head(x_t, t, image_tokens)     # (B, 16, 18) velocity
```

The `velocity_fn` that flow matching calls is just a closure over `image_tokens`:

```python
def velocity_fn(x_t, t):
    return head(x_t, t, image_tokens)
```

Continue to [**7 · Training: behavior cloning →**](07_training.md)
