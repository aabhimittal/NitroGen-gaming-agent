# 10 · Cross-game transfer — the generalist payoff

**Code:** [`nitrogen/benchmark/evaluate.py`](../nitrogen/benchmark/evaluate.py)
(`transfer_report`, `few_shot_transfer`)

## The claim

The whole point of a *foundation* model is that pretraining on broad data buys
you competence on things you didn't train for. NitroGen's headline result:
pretraining on internet gameplay transfers to **unseen games**, giving up to a
**52% relative improvement** in task-completion rate over policies trained from
scratch on the target game — the improvement is largest exactly where target-game
data is **scarce**.

## Two flavors of transfer

1. **Zero-shot transfer.** Run the pretrained policy on a game it never trained
   on. At the paper's scale (1,000+ games) the learned representation is broad
   enough to do non-trivial things zero-shot.
2. **Few-shot / low-data transfer.** Fine-tune the pretrained policy on a *small*
   amount of target-game data. The pretrained features mean it needs far fewer
   demonstrations than a from-scratch model to reach the same success — this is
   the setting the "52% relative improvement" describes.

## What this repo actually shows (and an honest caveat about scale)

Our miniature trains on just **two** games (**Reacher + Avoider**) and holds out
**Chaser**. At that scale a policy inevitably *overfits the exact appearance* of
the two training games, so **pure zero-shot** success on a genuinely novel game is
near zero — `transfer_report` will show it, and that is the honest result:

```python
from nitrogen.benchmark.evaluate import transfer_report
report = transfer_report(model, episodes=30)   # unseen chaser ≈ 0% zero-shot
```

The **few-shot** signal, on the other hand, reproduces cleanly and *is* the
paper's actual claim. `few_shot_transfer` fine-tunes the pretrained policy on a
handful of Chaser episodes and compares it to a model trained from scratch on the
exact same few episodes:

```python
from nitrogen.benchmark.evaluate import few_shot_transfer
r = few_shot_transfer(pretrained_model, game="chaser", n_episodes=12, finetune_steps=400)
# {"from_scratch": ~low, "pretrained": ~high, "relative_improvement": large +%}
```

Chaser shares Reacher's skill — *steer toward the warm salient object* — so the
pretrained policy adapts from a few demos far better than a fresh model can learn
from the same tiny dataset. That gap is the transfer payoff, measured on a laptop.

### Reproduce it

```bash
python -m scripts.train --games reacher avoider --steps 1500 --out checkpoints/nitrogen.pt
python -m scripts.evaluate --ckpt checkpoints/nitrogen.pt --transfer        # zero-shot view
python -m scripts.evaluate --ckpt checkpoints/nitrogen.pt --few-shot chaser  # low-data view
```

Or run the whole pipeline (data → train → benchmark → transfer) in one file:

```bash
python examples/quickstart.py
```

## Caveats — what this miniature does and doesn't show

This repo demonstrates the *mechanisms* faithfully at a scale you can run on a
laptop. It does **not** reproduce the paper's magnitudes: two toy games are not
1,000 real titles, and the emergent zero-shot breadth NitroGen shows comes
precisely from that scale and diversity. Treat the few-shot number here as a
working illustration of the transfer *setup*, not a restatement of the paper's
result.

← Back to [**1 · Overview**](01_overview.md) · See the [repository README](../README.md)
