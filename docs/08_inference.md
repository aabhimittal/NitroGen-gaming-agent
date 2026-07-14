# 8 · Inference — action chunking and closed-loop play

**Code:** [`nitrogen/models/nitrogen.py`](../nitrogen/models/nitrogen.py)
(`sample_actions`, `act`),
[`nitrogen/benchmark/evaluate.py`](../nitrogen/benchmark/evaluate.py)

## From one frame to a chunk of actions

At inference the policy sees a single frame and **generates** an action chunk by
integrating the flow-matching ODE ([ch 5](05_flow_matching.md)) from noise to
data:

```python
frame ─► encoder ─► image_tokens
x ← noise ~ N(0, I)                     # (1, T, 18)
for i in range(num_steps):             # ~10 steps is plenty
    t ← i / num_steps
    x ← x + (1/num_steps) · action_head(x, t, image_tokens)
chunk ← x                              # (1, 16, 18) in [-1, 1]
```

`NitroGen.sample_actions` returns the chunk in normalized space;
`NitroGen.act` additionally `decode_action`s it into raw gamepad vectors (sticks
in `[-1,1]`, triggers `[0,1]`, buttons rounded to `{0,1}`) ready to feed to a
game.

## Action chunking: execute several steps per model call

The model predicts **16** future actions from one frame, but we don't have to
re-plan every step. Standard practice (and what the benchmark uses) is:

```
observe frame
generate 16-action chunk
execute the first `replan_every` actions open-loop   (e.g. 4)
observe a new frame, generate a fresh chunk, repeat
```

This **action chunking** buys two things:

1. **Cheaper control.** The generative forward pass is the expensive part;
   amortizing it over several environment steps cuts the per-step cost.
2. **Temporal coherence.** Because the chunk was generated jointly
   ([ch 6](06_action_head.md)), executing a run of its actions produces a smooth,
   committed maneuver rather than per-step jitter.

There's a tradeoff in `replan_every`: larger = cheaper and smoother but less
reactive to surprises; smaller = more reactive but more compute. The default of 4
works well for the toy games.

## Closed-loop evaluation

`run_episode` in [`evaluate.py`](../nitrogen/benchmark/evaluate.py) implements the
loop above and returns whether the task succeeded:

```python
from nitrogen.benchmark.evaluate import evaluate_game
success_rate = evaluate_game(model, "reacher", episodes=30, replan_every=4)
```

This closed-loop success rate — *can the policy actually accomplish the task when
it controls the game* — is the metric that matters, and it's what the multi-game
benchmark aggregates.

Continue to [**9 · The multi-game benchmark →**](09_benchmark.md)
