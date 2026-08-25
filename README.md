# NitroGen — an open vision-action foundation model for generalist gaming agents

> An **educational, end-to-end, runnable** re-implementation of
> [**NitroGen**](https://nitrogen.minedojo.org/) (NVIDIA, 2025): a single RGB
> game frame → standardized gamepad actions, generated with **flow matching** and
> trained by **pure behavior cloning**. Every concept from the paper is
> implemented in miniature, explained step by step in [`docs/`](docs/), and
> actually trains and plays on a laptop.

<p align="center"><em>frame&nbsp;→&nbsp;vision encoder&nbsp;→&nbsp;flow-matching action head&nbsp;→&nbsp;action chunk&nbsp;→&nbsp;play</em></p>

This repo is **not** the official model or weights. It is a from-scratch teaching
implementation whose goal is that you can read one Python package + ten short
chapters and understand *exactly* how a vision-action gaming foundation model
works — then run the whole pipeline yourself.

- 📄 Paper: *NitroGen: An Open Foundation Model for Generalist Gaming Agents* ([project page](https://nitrogen.minedojo.org/) · [arXiv](https://arxiv.org/abs/2601.02427))
- 🧠 Original model builds on **SigLIP-2** (vision) + **GR00T N1** (flow-matching action expert)

---

## What NitroGen actually is (30-second version)

| | NitroGen |
|---|---|
| **Input** | one 256×256 RGB game frame (no history, no game internals) |
| **Output** | a chunk of 16 future **standardized gamepad** actions (sticks + triggers + buttons) |
| **How actions are generated** | **flow matching** — a diffusion-style ODE from noise → actions |
| **Training** | **pure behavior cloning** (imitation). No RL, no rewards. |
| **Data** | ~40,000 h of internet gameplay across 1,000+ games, action-labeled by **reading on-screen controller overlays** |
| **Evaluation** | a **multi-game benchmark**; transfers to unseen games (up to **52%** relative gain vs. from-scratch) |

The design rests on **three pillars**, each mapped to code and a doc chapter:

1. **Internet-scale video-action data** from controller overlays → [`nitrogen/data/`](nitrogen/data/) · [ch 3](docs/03_data_pipeline.md)
2. **A multi-game benchmark** for cross-game generalization → [`nitrogen/envs/`](nitrogen/envs/), [`nitrogen/benchmark/`](nitrogen/benchmark/) · [ch 9](docs/09_benchmark.md)
3. **A unified vision-action model** (SigLIP ViT + flow-matching DiT) trained with BC → [`nitrogen/models/`](nitrogen/models/) · [ch 4–8](docs/04_vision_encoder.md)

---

## Architecture at a glance

```
                        ┌───────────────────── training (behavior cloning) ─────────────────────┐
 gameplay video         │                                                                        │
   with overlay ──► [overlay extraction] ──► (frame, action-chunk) pairs ──┐                      │
   (ch 3)                                                                   │                      │
                                        standardized gamepad action space  │  (ch 2)              │
                                                                           ▼                      │
   frame (256×256) ──► [SigLIP-style ViT] ──► image tokens ──┐                                    │
   (ch 4)                                                     │                                    │
                                                             ▼                                     │
   noise x₀ + flow-time t ──► [flow-matching DiT action head] ──► velocity v_θ  (ch 5, 6)          │
                                                             │                                     │
                                          flow-matching MSE: ‖v_θ − (x₁−x₀)‖²  ◄── expert chunk x₁ │
                        └────────────────────────────────────────────────────────────────────────┘

 inference (ch 8):  frame ──► encode ──► integrate ODE from noise ──► action chunk ──► execute k steps, re-plan
```

Full walkthrough: [`docs/01_overview.md`](docs/01_overview.md).

---

## Quickstart

```bash
pip install -e .          # or: pip install -r requirements.txt

# 1) See action-extraction-from-overlays reproduce the paper's metric framing
python -m scripts.generate_data --overlay-demo

# 2) Run the ENTIRE pipeline (data → BC train → benchmark → transfer) in one file
python examples/quickstart.py
```

Or drive the pieces individually:

```bash
python -m scripts.generate_data --games reacher avoider --episodes 80 --out data/train.npz
python -m scripts.train        --games reacher avoider --steps 1500 --out checkpoints/nitrogen.pt
python -m scripts.evaluate     --ckpt checkpoints/nitrogen.pt --transfer
python -m scripts.evaluate     --ckpt checkpoints/nitrogen.pt --robustness
python -m scripts.demo         --ckpt checkpoints/nitrogen.pt --game reacher --out assets/reacher.gif
```

Use it as a library:

```python
import numpy as np
from nitrogen import NitroGen, NitroGenConfig

model = NitroGen(NitroGenConfig())          # ~10M-param policy
frame = np.random.rand(256, 256, 3)         # your game frame (H, W, 3)
action_chunk = model.act(frame)             # (16, 18) raw gamepad vectors
```

Ship it against a real game — sanitized pixels, a supervised sampler, and
hardware-legal actions ([ch 11](docs/11_deployment.md)):

```python
from nitrogen.deploy import PolicyRuntime

runtime = PolicyRuntime.from_model(model, seed=0)   # seed => reproducible rollouts
report = runtime.step(capture.grab())               # never raises, never stalls
report.action                                       # always a legal gamepad action
```

```bash
python -m examples.deployment    # one episode against a deliberately broken capture
```

> **Resolution note.** The paper encodes 256×256 frames (256 patch tokens). The
> default config here downsamples to **128×128** so training finishes in minutes
> on CPU; the environments still render at 256 for crisp demos. Pass
> `VisionConfig(image_size=256)` to match the paper.

---

## Repository layout

```
nitrogen/
├── action_space.py            # (ch 2) standardized gamepad; encode/decode to [-1,1]
├── models/
│   ├── vision_encoder.py      # (ch 4) SigLIP-style ViT → image tokens
│   ├── flow_matching.py       # (ch 5) rectified-flow paths, loss, ODE sampler
│   ├── action_head.py         # (ch 6) DiT "action expert" w/ AdaLN-Zero + cross-attn
│   └── nitrogen.py            # full policy: compute_loss (BC) + sample_actions
├── data/
│   ├── overlay_extraction.py  # (ch 3) render + read back a controller overlay
│   ├── synthetic.py           # expert rollouts → (frame, action) corpus
│   └── dataset.py             # single-frame → 16-step action-chunk pairs
├── envs/
│   ├── rendering.py           # tiny numpy software renderer
│   └── toy_game.py            # (ch 9) Reacher / Avoider / Chaser + experts
├── training/
│   ├── config.py
│   └── trainer.py             # (ch 7) AdamW · WSD schedule · EMA · augmentation
├── deploy/
│   ├── frames.py              # (ch 11) capture hardening: dtype/layout/NaN/black/frozen
│   ├── safety.py              # (ch 11) hardware-legal, human-plausible actions
│   ├── controller.py          # (ch 11) chunk → per-tick: ensembling, latency, starvation
│   └── runtime.py             # (ch 11) supervised loop, circuit breaker, metrics
└── benchmark/
    ├── evaluate.py            # (ch 8/9/10) closed-loop success, transfer report
    └── robustness.py          # (ch 11) success under capture faults, naive vs. guarded

docs/     # 11 step-by-step chapters (start at 01_overview.md)
scripts/  # generate_data · train · evaluate · demo
examples/ # quickstart.py — the whole thing in one file; deployment.py — the hardened loop
tests/    # pytest: action space, flow matching, model, envs & data, deployment, robustness
```

---

## The step-by-step guide

Read these in order — each is short and links to the exact code it explains.

1. [Overview: what NitroGen is and why it matters](docs/01_overview.md)
2. [The standardized gamepad action space](docs/02_action_space.md)
3. [Building a video-action dataset from controller overlays](docs/03_data_pipeline.md)
4. [The vision encoder (SigLIP-style ViT)](docs/04_vision_encoder.md)
5. [Flow matching — the generative engine](docs/05_flow_matching.md)
6. [The action head — a Diffusion Transformer](docs/06_action_head.md)
7. [Training — large-scale behavior cloning](docs/07_training.md)
8. [Inference — action chunking and closed-loop play](docs/08_inference.md)
9. [The multi-game benchmark](docs/09_benchmark.md)
10. [Cross-game transfer — the generalist payoff](docs/10_transfer.md)
11. [Deployment — running the policy against a real game](docs/11_deployment.md)

---

## Faithful vs. simplified

**Faithful to the paper (mechanisms):**
- pixels-only input; a *single* frame conditions a *chunk* of future actions,
- one **standardized gamepad** action interface shared across all games,
- action **generation via flow matching** (noise → actions ODE), not regression,
- a **DiT action expert** with self-attention over the chunk + cross-attention to
  image tokens + AdaLN-Zero time conditioning,
- **pure behavior-cloning** objective with AdamW · WSD schedule · EMA · augmentation,
- **action chunking** at inference; a **multi-game benchmark** with a held-out game,
- **overlay-based action labeling** as the data-construction principle.

**Simplified so it runs anywhere:**
- the vision encoder is a small from-scratch ViT (plus two CoordConv coordinate
  channels to make object localization tractable at tiny scale), not pretrained
  SigLIP-2 weights,
- cross-attention to the image is left **always-on** rather than AdaLN-zero-gated,
  so the tiny model actually learns to look at the frame (see [ch 6](docs/06_action_head.md)),
- data is scripted-expert rollouts in toy *agent-centric* games, not 40,000 h of
  real video, and overlay extraction reads a clean synthetic HUD instead of a
  learned SegFormer,
- the toy chunk horizon is 8 (the paper uses 16); the model is ~10M params at
  128px, not 500M at 256px,
- three toy games stand in for 1,000+ real titles, so transfer here is shown
  **few-shot** (the paper's low-data claim), not the emergent zero-shot breadth
  that only appears at scale.

On the toy suite the trained policy reaches **~100% closed-loop success** on the
games it was trained on (Reacher, Avoider), and few-shot fine-tuning on a dozen
episodes of the held-out Chaser game clearly beats training from scratch on the
same data — the transferable-skill payoff, in miniature.

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers the action-space round-trip, flow-matching invariants (path endpoints,
velocity target, exact ODE recovery), model shapes + single-batch overfitting,
environment/expert competence, and overlay-extraction quality.

Roughly half the suite is **industrial edge cases** — the things that only show
up once a real capture pipeline and a real gamepad are on the other end:

| Area | What is pinned down |
|---|---|
| `test_deploy_frames.py` | channel-first/BGR/RGBA/grayscale inputs, uint16 and float-in-`[0,255]` captures, NaN and HDR pixels, ultrawide letterboxing, black and frozen-capture detection |
| `test_deploy_safety.py` | unit-circle clamping, radial (not axis-wise) deadzones, slew limiting, press/release hysteresis and debounce, impossible d-pad combinations, blocked menu buttons |
| `test_deploy_controller.py` | chunk seams and ensembling, latency compensation, starvation policies, bounded memory under rapid replanning |
| `test_deploy_runtime.py` | a throwing policy, NaN chunks, the circuit breaker opening and recovering, buttons never latching while idle, an end-to-end episode against a hostile capture stream |
| `test_model_edge_cases.py` | empty batches, chunk size 1, a single ODE step, odd resolutions, seeded determinism, saturated targets |
| `test_data_edge_cases.py` | chunk padding at episode boundaries, ragged save/load round trips, overlay footage with no visible widget |

---

## Citation

If you use the *ideas*, cite the original work:

```bibtex
@article{nitrogen2025,
  title   = {NitroGen: An Open Foundation Model for Generalist Gaming Agents},
  author  = {NVIDIA},
  year    = {2025},
  url     = {https://nitrogen.minedojo.org/}
}
```

This repository is an independent educational re-implementation and is not
affiliated with or endorsed by the NitroGen authors or NVIDIA.

## License

[MIT](LICENSE).
