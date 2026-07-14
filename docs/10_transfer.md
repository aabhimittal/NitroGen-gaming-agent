# 10 · Cross-game transfer — the generalist payoff

**Code:** [`nitrogen/benchmark/evaluate.py`](../nitrogen/benchmark/evaluate.py)
(`transfer_report`)

## The claim

The whole point of a *foundation* model is that pretraining on broad data buys
you competence on things you didn't train for. NitroGen's headline result:
pretraining on internet gameplay transfers to **unseen games**, giving up to a
**52% relative improvement** in task-completion rate over policies trained from
scratch on the target game, and it can perform non-trivial tasks on some unseen
games **zero-shot**.

## Two flavors of transfer

1. **Zero-shot transfer.** Evaluate the pretrained policy on a game that was
   *never* in its training set. Because the action space is standardized
   ([ch 2](02_action_space.md)) and the required skill ("move toward the salient
   object") recurs across games, the policy retains non-trivial success.
2. **Few-shot / low-data transfer.** Fine-tune the pretrained policy on a small
   amount of target-game data. The pretrained features mean it needs far fewer
   demonstrations than a from-scratch model to reach the same success — this is
   where the "52% relative improvement" comes from.

## The experiment in this repo

We train on **Reacher + Dodger** and hold out **Chaser** as the unseen game.
`transfer_report` scores both splits:

```python
from nitrogen.benchmark.evaluate import transfer_report
report = transfer_report(model, episodes=30)
# {
#   "seen":   {"reacher": ..., "dodger": ..., "mean": ...},
#   "unseen": {"chaser": ...,               "mean": ...},
# }
```

Chaser shares Reacher's underlying skill (drive the avatar onto a salient object)
but has different colours, a different avatar, and a *moving* target. A policy
that merely memorized Reacher pixels would score ~0 on Chaser; a policy that
learned the transferable skill "push the stick toward the bright objective"
scores well above chance zero-shot. That gap **is** generalization, measured.

### Reproduce it

```bash
python -m scripts.train --games reacher dodger --steps 1500 --out checkpoints/nitrogen.pt
python -m scripts.evaluate --ckpt checkpoints/nitrogen.pt --transfer
```

Or run the whole pipeline (data → train → benchmark → transfer) in one file:

```bash
python examples/quickstart.py
```

## Caveats — what this miniature does and doesn't show

This repo demonstrates the *mechanisms* faithfully at a scale you can run on a
laptop. It does **not** reproduce the paper's magnitudes: three toy games are not
1,000 real titles, and the emergent breadth NitroGen shows comes precisely from
that scale and diversity. Treat the numbers here as a working illustration of the
transfer *setup*, not a restatement of the paper's results.

← Back to [**1 · Overview**](01_overview.md) · See the [repository README](../README.md)
