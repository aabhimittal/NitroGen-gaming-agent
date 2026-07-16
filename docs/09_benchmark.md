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
the same generalist skill — *find the one salient object and push the left stick
toward or away from it* — which is exactly what makes cross-game transfer
meaningful. Each is rendered **agent-centric**: the controllable avatar is a fixed
reticle at screen center and the world is drawn around it (as most 1st/3rd-person
games do), so the task reduces to single-object visuomotor control the model can
actually learn from pixels.

| Game | Genre analog | Skill | Success |
|------|--------------|-------|---------|
| **Reacher** | "go to the objective" | push the reticle onto a stationary beacon | reach the beacon |
| **Avoider** | reflex dodging | strafe *away* so an incoming asteroid misses the reticle | survive the episode |
| **Chaser** | pursuit / exploration | push *toward* a fleeing orb until you catch it (held out for transfer) | catch the orb |

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
scores = evaluate_suite(model, ["reacher", "avoider"], episodes=30)
# {"reacher": 0.9, "avoider": 0.8, "mean": 0.85}
```

Fresh seeds (offset well past the training seeds) mean evaluation states were
never in the training data — so even the "seen games" number measures
generalization to new episodes, not memorization.

Continue to [**10 · Cross-game transfer →**](10_transfer.md)
