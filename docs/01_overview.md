# 1 · Overview — what NitroGen is and why it matters

> **NitroGen** (NVIDIA, 2025) is an *open vision-action foundation model for
> generalist gaming agents*. It maps a single **256×256 RGB game frame** directly
> to **standardized gamepad actions**, and it is trained by **pure behavior
> cloning** on ~40,000 hours of internet gameplay across 1,000+ games — no
> reinforcement learning, no reward functions, no per-game engineering.

This repository is a compact, **runnable** re-implementation that teaches every
concept end to end. It won't reach the paper's scale, but every mechanism —
overlay-based action labeling, the standardized action interface, the SigLIP
vision encoder, the flow-matching action head, behavior-cloning training, action
chunking, and the multi-game benchmark — is here in miniature and actually trains
and plays.

## The problem: generalist control from pixels

A "generalist gaming agent" should be able to pick up a controller, look at *any*
game, and do something sensible — the way a human who has played many games can
sit down at a new one and immediately make progress. Concretely that means a
single policy that:

- takes **only pixels** as input (no access to game internals, memory, or APIs),
- outputs **controller actions** that work across wildly different games,
- was trained **once**, and **transfers** to games it never saw.

Classic game-playing agents (AlphaStar, OpenAI Five, most RL agents) are the
opposite: one agent, one game, trained with reward signals specific to that game
and often privileged state. They are *specialists*. NitroGen asks whether the
"foundation model" recipe — huge, diverse, self-supervised-ish data + one big
model + simple imitation objective — produces a *generalist* controller the same
way it produced generalist language and vision models.

## The three ingredients

NitroGen's contribution is really three things that fit together:

| # | Ingredient | This repo's chapter |
|---|------------|---------------------|
| 1 | An **internet-scale video-action dataset**, built by *reading player actions off on-screen controller overlays* in ordinary gameplay videos. | [03 · Data pipeline](03_data_pipeline.md) |
| 2 | A **multi-game benchmark** that measures cross-game generalization, so "generalist" is a number, not a vibe. | [09 · Benchmark](09_benchmark.md) |
| 3 | A **unified vision-action model** — one SigLIP encoder + one flow-matching action head — trained with large-scale behavior cloning. | [04](04_vision_encoder.md)–[08](08_inference.md) |

The rest of the guide walks through each, in the order the data flows:

```
gameplay video ─► [overlay extraction] ─► (frame, action) pairs      (ch 3)
                          │
    standardized gamepad action space ◄──┘                            (ch 2)
                          │
    frame ─► [SigLIP ViT] ─► image tokens                             (ch 4)
                          │
    noise + t ─► [flow-matching DiT action head] ─► velocity          (ch 5, 6)
                          │
    behavior cloning loss  ◄──┘                                       (ch 7)
                          │
    at inference: integrate ODE ─► action chunk ─► play the game      (ch 8)
                          │
    multi-game benchmark + transfer                                   (ch 9, 10)
```

## Why this design (the one-line intuitions)

- **Pixels → gamepad**, nothing else: the only interface every game shares is
  "a screen" and "a controller", so that's the only interface the model uses.
- **Behavior cloning, not RL**: reward functions don't generalize across 1,000
  games, but *imitation* does — and internet gameplay is a near-limitless supply
  of demonstrations if you can label the actions ([ch 3](03_data_pipeline.md)).
- **Flow matching, not regression**: at any moment several actions may be valid;
  a generative head can commit to one coherent choice instead of averaging them
  into mush ([ch 5](05_flow_matching.md)).
- **Action chunks, not single actions**: predicting 16 steps at once gives
  temporal coherence and lets the agent run several steps per model call
  ([ch 8](08_inference.md)).

Continue to [**2 · The standardized gamepad action space →**](02_action_space.md)
