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

## Data design: why the chunk must be predictable from the frame

Getting this tiny model to actually *play* (not just drive the loss down) came
down to the **training data**, not the architecture. Three lessons, learned the
hard way, that are worth internalizing because they generalize to real BC:

1. **The whole chunk must be a predictable function of the conditioning frame.**
   We predict `T` future actions from *one* frame. If anything in that window is
   *unknowable* from the frame — e.g. an object that randomly respawns in a new
   place mid-chunk — those steps become irreducible noise in the target. Flow
   matching then hedges toward the mean and the policy collapses into mush. Our
   toy games therefore use only *deterministic* object dynamics (no random jumps),
   so the chunk is a clean function of what's on screen.

2. **Watch the class balance of your actions.** Early versions let each episode
   keep running after the goal was reached; the agent then sat still and emitted a
   flood of "do nothing" frames that *dominated* the dataset. The policy dutifully
   learned to do nothing. The fix: end reacher/chaser episodes on contact, so the
   data is pure "steer toward the objective." (Avoider never idles — it is always
   pushing the hazard away.)

3. **EMA needs enough steps to catch up.** With decay `0.999`, after only ~1k
   steps the EMA weights are still ~40% their random initialization — so the
   "smoothed" model you evaluate is half-untrained and looks broken. For short
   runs use a faster decay (this repo defaults to `0.99`).

These are exactly the un-glamorous data/optimization details that separate a BC
policy that works from one that silently doesn't — the model was capable the
whole time (it memorizes a clean image→action mapping to ~0.99 correlation).

## Running it

```bash
# tiny end-to-end run (a few minutes on CPU at the default 128px)
python -m scripts.train --games reacher avoider --steps 1500 --out checkpoints/nitrogen.pt
```

The trainer saves the **EMA** weights — those are what you evaluate and deploy.
Watch the flow-matching loss fall (a healthy run drops from ~0.8 to well under
0.1 as the head learns to reconstruct expert action chunks); on the toy suite the
resulting policy reaches ~100% closed-loop success on the games it trained on.

Continue to [**8 · Inference: action chunking →**](08_inference.md)
