# 11 · Deployment — running the policy against a real game

**Code:** [`nitrogen/deploy/`](../nitrogen/deploy/),
[`nitrogen/benchmark/robustness.py`](../nitrogen/benchmark/robustness.py)

Chapters 1–10 build a policy and measure it in the lab. This chapter is about
the gap between that and an agent you can leave running, because the lab loop
quietly assumes three things that stop being true the moment a real game is on
the other end:

```python
frame = env.render()          # 1. always clean, always the right shape
chunk = model.act(frame)      # 2. always answers, instantly
env.step(chunk[i])            # 3. any vector in [-1, 1]^18 is a legal action
```

## 1 · The pixels are not what you trained on

A capture stack is a pipeline of other people's code. In practice a frame
arrives as any of: channel-first, BGR (OpenCV hands you BGR), RGBA from a
compositor, a single grayscale plane, uint16 from a 10-bit capture, float still
in `[0, 255]` because the producer forgot to divide, or with NaN pixels from an
HDR path. The resolution is whatever the window happens to be, and the aspect
ratio may not be the model's. Sometimes the frame is entirely black (loading,
occluded window) — and sometimes it is the *previous* frame, forever, because
capture silently stalled.

[`FrameSanitizer`](../nitrogen/deploy/frames.py) converts what it can and reports
what it can't:

```python
pixels, health = FrameSanitizer(target_size=(256, 256), keep_aspect=True)(raw)
if pixels is None:
    ...                       # health.reason says why: bad_shape, all_nonfinite, ...
health.repaired               # ('chw_to_hwc', 'dropped_alpha', 'rescaled_from_255')
health.black, health.frozen   # advisory: structurally fine, but don't trust it
```

Two judgement calls worth naming. **Freeze detection needs patience**: a single
repeated frame is a paused menu or a static cutscene, not a fault, so the
default fires only after three consecutive identical frames — detection latency
traded for a false-positive rate you can live with. And an **ultrawide capture is
letterboxed, not squashed**: stretching 21:9 into a square changes every
on-screen distance the policy learned from, which is exactly the cue these tasks
are built on.

## 2 · The sampler is not a pure function

Flow matching integrates an ODE from fresh Gaussian noise, on a GPU that is also
running a game. It can be slow, it can OOM, and a diverged checkpoint can hand
back NaN. Meanwhile the game does not wait.

[`PolicyRuntime`](../nitrogen/deploy/runtime.py) enforces the one invariant a
control loop cannot violate — **every tick returns a legal action, on time** — by
degrading in a fixed order: the fresh chunk, else the buffered chunk, else the
executor's fallback. Repeated failures trip a **circuit breaker**: a model
throwing once per tick would otherwise burn the whole tick budget on retries, so
the runtime stops calling it, emits neutral, and retries after a cooldown so a
transient fault still recovers by itself.

It also idles deliberately. On a frozen capture, acting is *worse* than not
acting: the policy is confidently controlling a world state that has already
moved on. And idling means neutral, not "stop updating" — a loop that stops
updating leaves the last action latched on the virtual pad, stick held, trigger
down, indefinitely.

## 3 · Chunks meet a clock

The policy emits `chunk_size` actions from one frame; the game wants exactly one
action per tick, forever. [`ChunkExecutor`](../nitrogen/deploy/controller.py)
bridges the two rates and handles the three things that go wrong there:

| Problem | Handling |
|---|---|
| Chunk seams stutter — chunk *k+1* was generated from a frame the game has passed | **temporal ensembling**: overlapping chunks are averaged with exponentially decaying weights (buttons are *voted*, never half-pressed) |
| Inference latency: a chunk from tick *k* arrives at tick *k+L*, so its first *L* actions replay the past | **latency compensation**: skip the stale head; drop the chunk entirely if it's shorter than *L* |
| Sampling stalls and the buffer runs dry | an explicit **starvation policy** — `neutral`, `hold`, or `decay` — instead of accidentally repeating the last action forever |

## 4 · The action space is wider than the hardware

The action head samples from a learned distribution over `[-1, 1]^18`. Nothing
in that formulation knows about the pad on the other end, so straight from the
sampler you routinely get a stick at magnitude 1.38 (outside the circular gate a
thumbstick is physically confined to — the driver clamps it silently and the game
sees a *different direction* than the policy intended), `dpad_up` and
`dpad_down` simultaneously, a one-tick button flicker from a channel that landed
near the decision boundary, or `start`, which pauses the game and ends your
unattended benchmark run.

[`ActionGuard`](../nitrogen/deploy/safety.py) is the last stage before the
virtual pad: radial deadzone (not axis-wise — that's the classic bug that snaps a
slow diagonal to a cardinal direction), unit-circle clamp, slew-rate limiting,
press/release hysteresis with a minimum hold time, exclusive d-pad pairs, and a
blocklist for the menu buttons.

## 5 · Measuring whether any of it helped

Hardening that doesn't move a number is complexity you should delete, so
[`robustness.py`](../nitrogen/benchmark/robustness.py) measures it. Each fault is
a mechanical failure of the *capture and control path* — dropped frames, a
stalled capture, sensor noise, wrong channel layout, float HDR frames, an
ultrawide resolution, a shifted brightness curve — applied identically to two
arms: the naive `model.act` loop, and the same policy behind `PolicyRuntime`.
Same task, same seeds, same weights; the delta is attributable to the fault
handling alone.

```bash
python -m scripts.evaluate --ckpt checkpoints/nitrogen.pt --robustness
python -m scripts.evaluate --ckpt checkpoints/nitrogen.pt --robustness --latency-steps 2
```

A fault where `guarded` is no better than `raw` is worth knowing about *before*
it happens in production — that's a limit of the layer, and the report is
designed to show it rather than hide it.

## Putting it together

```python
from nitrogen.deploy import PolicyRuntime

runtime = PolicyRuntime.from_model(model, seed=0)   # seed => reproducible rollouts
while not done:
    report = runtime.step(capture.grab())           # never raises, never stalls
    frame, done, info = env.step(report.action)
print(runtime.metrics.summary())                    # p50/p95 latency, degraded rate
```

`seed` matters more than it looks. Sampling starts from fresh noise every call,
so without it two runs over identical frames produce different actions and a
regression run can't tell a real regression from resampling noise.

Runnable end to end: [`examples/deployment.py`](../examples/deployment.py).

Back to [**the guide index →**](README.md)
