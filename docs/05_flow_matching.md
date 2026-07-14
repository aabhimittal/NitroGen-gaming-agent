# 5 · Flow matching — the generative engine

**Code:** [`nitrogen/models/flow_matching.py`](../nitrogen/models/flow_matching.py)

## Why not just regress the action?

The obvious way to imitate an expert is regression: a network that takes the
frame and outputs the action, trained with MSE. This fails for control in two
related ways:

1. **Multi-modality.** At a given frame, several actions may be equally good —
   dodge *left* or dodge *right*, both valid. MSE regression minimizes average
   squared error, so it predicts the **mean** of the valid actions: "dodge
   straight into the hazard." Averaging distinct modes yields an action that is
   *nowhere near* any expert demonstration.
2. **No coherent commitment.** We want the agent to *pick* a mode and follow
   through, not blend them.

A **generative** model of `p(action | frame)` fixes this: it can represent the
multi-modal distribution and *sample* one coherent action instead of averaging.
NitroGen's generative model of choice is **(rectified) flow matching**.

## Flow matching in one page

Think of two distributions: pure Gaussian **noise** `x₀ ~ N(0, I)` and real
**data** `x₁` (an expert action chunk). Draw a straight line between a noise
sample and a data sample:

```
x_t = (1 − t)·x₀ + t·x₁ ,        t ∈ [0, 1]
```

At `t = 0` you're at noise, at `t = 1` you're at data. Differentiate w.r.t. `t`:
the velocity that carries a point along this line is **constant**:

```
dx_t/dt = x₁ − x₀            ← the flow-matching target
```

We train a network `v_θ(x_t, t, condition)` to predict that velocity from a
*noised* chunk `x_t`, the time `t`, and the image conditioning. The loss is a
plain MSE between predicted and true velocity:

```
L = E_{x₁, x₀, t}  ‖ v_θ(x_t, t, cond) − (x₁ − x₀) ‖²
```

That's the entire training objective. No score functions, no noise schedules, no
variational bounds — just "predict the straight-line velocity."

### Sampling: integrate the ODE

To *generate* an action chunk at inference, start from noise `x₀ ~ N(0, I)` and
follow the learned velocity field from `t=0` to `t=1` with a simple ODE
integrator (forward Euler):

```
x ← x₀
for i in range(num_steps):
    t ← i / num_steps
    x ← x + (1/num_steps) · v_θ(x, t, cond)
return x        # ≈ a sample from p(action chunk | frame)
```

Because the ideal path is a straight line, flow matching needs **very few
integration steps** (this repo defaults to 10; the endpoint is stable long before
that). That is a major practical win over classic diffusion, which often needs
tens to hundreds of denoising steps.

## The API in this repo

`FlowMatching` is deliberately model-agnostic — it knows nothing about images or
gamepads, only paths and velocities:

```python
from nitrogen.models.flow_matching import FlowMatching
fm = FlowMatching()

# training: velocity_fn closes over the image conditioning
loss = fm.loss(velocity_fn, action_chunk_x1)

# sampling: integrate noise -> data
chunk = fm.sample(velocity_fn, shape=(B, T, 18), device=..., num_steps=10)
```

The unit tests in [`tests/test_flow_matching.py`](../tests/test_flow_matching.py)
verify the path endpoints, that the target equals the displacement, and that
integrating the *true* field lands exactly on the data point.

Next we build the network that predicts `v_θ`: the action head.

Continue to [**6 · The diffusion-transformer action head →**](06_action_head.md)
