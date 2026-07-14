# 7 · Training — large-scale behavior cloning

**Code:** [`nitrogen/training/trainer.py`](../nitrogen/training/trainer.py),
[`nitrogen/training/config.py`](../nitrogen/training/config.py)

## Pure imitation, no RL

NitroGen is trained with **behavior cloning only**. There is no reward function,
no environment interaction during training, no policy-gradient or value learning
— nothing that would have to be re-designed per game. The entire objective is:

> match the expert's action chunk, using the flow-matching loss.

This is the crucial design decision that lets one recipe span 1,000+ games.
Reward functions are game-specific and don't generalize; *demonstrations* do, and
the overlay pipeline ([ch 3](03_data_pipeline.md)) supplies them at internet
scale.

## The training step

Every step is four lines of concept:

```python
image_tokens = encoder(frames)                       # (B, N, width)   ch 4
def velocity_fn(x_t, t):                              #                 ch 6
    return action_head(x_t, t, image_tokens)
loss = flow.loss(velocity_fn, expert_action_chunks)  # FM MSE          ch 5
loss.backward()
```

`flow.loss` internally: samples noise `x₀` and a time `t`, forms the noised chunk
`x_t = (1−t)x₀ + t·x₁`, and asks the head to predict the velocity `x₁ − x₀`. See
[`NitroGen.compute_loss`](../nitrogen/models/nitrogen.py).

## The recipe (small-scale mirror of the paper)

[`trainer.py`](../nitrogen/training/trainer.py) reproduces NitroGen's training
ingredients:

| Ingredient | Why | Where |
|-----------|-----|-------|
| **AdamW** (decoupled weight decay) | stable default for Transformers | `torch.optim.AdamW` |
| **Warmup-Stable-Decay (WSD)** LR schedule | linear warmup → long constant plateau → cosine decay; the plateau is resumable/extendable, which matters at foundation-model scale | `wsd_lr()` |
| **EMA** of weights | evaluate a smoothed copy of the parameters; reduces variance and usually beats the raw weights | `EMA` |
| **Image augmentation** | brightness/contrast jitter → robustness to the visual diversity of 1,000 games | `augment()` |
| **Gradient clipping** | guards against occasional large updates | `clip_grad_norm_` |

### Warmup-Stable-Decay, visually

```
lr
 │        ┌───────────────────────────┐
 │       /                             \
 │      /                               \
 │     /                                 \____
 └────┴─────────────────────────────────┴──────► step
   warmup        stable (plateau)         decay
```

The long flat middle is the point: you can train for as long as your compute
budget allows on the plateau and only *then* schedule the decay, without having
committed to a total step count up front (unlike a pure cosine schedule).

## Running it

```bash
# tiny end-to-end run (a few minutes on CPU at the default 128px)
python -m scripts.train --games reacher dodger --steps 1500 --out checkpoints/nitrogen.pt
```

The trainer saves the **EMA** weights — those are what you evaluate and deploy.
Watch the flow-matching loss fall (a healthy run drops from ~0.8 to well under
0.1 as the head learns to reconstruct expert action chunks).

Continue to [**8 · Inference: action chunking →**](08_inference.md)
