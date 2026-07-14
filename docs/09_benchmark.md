# 9 · The multi-game benchmark

**Code:** [`nitrogen/envs/toy_game.py`](../nitrogen/envs/toy_game.py),
[`nitrogen/benchmark/evaluate.py`](../nitrogen/benchmark/evaluate.py)

## Why a benchmark is a *contribution*

"Generalist" only means something if you can measure it. NitroGen's second pillar
is a **multi-game benchmark environment** spanning 2D and 3D genres (the paper
reports **10 games, 30 tasks**), scored by closed-loop task success. Without it,
cross-game generalization is an anecdote; with it, it's a number you can compare
models on.

## This repo's miniature suite

We ship a small family of visually distinct games in
[`toy_game.py`](../nitrogen/envs/toy_game.py). They look different but all exercise
the same generalist skill — *look at the frame and push the left stick toward what
matters* — which is exactly what makes cross-game transfer meaningful.

| Game | Genre analog | Skill | Success |
|------|--------------|-------|---------|
| **Reacher** | 3D "go to the objective" | drive the avatar onto a target | reach the target |
| **Dodger** | 2D platformer reflex | slide a paddle to avoid a falling hazard | survive the episode |
| **Chaser** | pursuit / exploration | catch a fleeing orb (held out for transfer) | catch the prey |

Every game implements the same tiny interface and is controlled through the
shared gamepad ([ch 2](02_action_space.md)):

```python
from nitrogen.envs.toy_game import make_game
env = make_game("reacher")
frame = env.reset(seed=0)                 # (256, 256, 3) uint8
frame, done, info = env.step(action)      # action: raw gamepad vector
# info["success"] reports task completion
```

Each game also provides a scripted `expert_action()` — used to generate the
behavior-cloning data ([ch 3](03_data_pipeline.md)) and, in the tests, to confirm
the tasks are actually solvable (the experts win ~100%).

## Scoring

`evaluate_suite` runs the policy closed-loop over fresh seeds for each game and
reports per-game success plus the mean:

```python
from nitrogen.benchmark.evaluate import evaluate_suite
scores = evaluate_suite(model, ["reacher", "dodger"], episodes=30)
# {"reacher": 0.9, "dodger": 0.8, "mean": 0.85}
```

Fresh seeds (offset well past the training seeds) mean evaluation states were
never in the training data — so even the "seen games" number measures
generalization to new episodes, not memorization.

Continue to [**10 · Cross-game transfer →**](10_transfer.md)
