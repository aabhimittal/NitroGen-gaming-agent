"""The supervised loop that actually drives a game: sanitize, sample, guard, emit.

:class:`PolicyRuntime` is the piece that turns a research checkpoint into
something you can leave running unattended. It owns the three failure domains
the rest of this package addresses — bad pixels, an unreliable sampler, illegal
actions — and enforces the one invariant a control loop must never violate:

    **every tick returns an action, on time, and that action is always legal.**

A research loop is allowed to raise. A control loop is not: the game keeps
running whether or not the policy answered, and an exception that propagates
leaves the last action latched on the virtual pad — stick held, trigger down,
indefinitely. So the runtime degrades instead, in a fixed order of preference:

    1. the freshly sampled chunk
    2. the previously buffered chunk, if this frame was unusable
    3. the executor's fallback (neutral by default), if nothing is buffered

Repeated policy failures trip a **circuit breaker**. A model that throws once
per tick — CUDA OOM, a shape bug, a corrupted checkpoint — would otherwise burn
the entire tick budget on retries and make the agent miss its deadline as well
as its action. The breaker stops calling it, emits neutral, and retries
periodically so a transient fault still recovers on its own.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Optional

import numpy as np

from nitrogen.action_space import ACTION_DIM
from nitrogen.deploy.controller import ChunkExecutor
from nitrogen.deploy.frames import FrameHealth, FrameSanitizer
from nitrogen.deploy.safety import ActionGuard

PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class StepReport:
    """What happened on one tick. Cheap enough to log every frame."""

    action: np.ndarray
    tick: int
    reason: str = "ok"
    replanned: bool = False
    degraded: bool = False
    latency_ms: float = 0.0
    health: Optional[FrameHealth] = None

    @property
    def ok(self) -> bool:
        return self.reason == "ok"


class RuntimeMetrics:
    """Rolling latency percentiles plus lifetime counters.

    Latency is kept in a bounded window because what matters operationally is
    "am I meeting my deadline *now*", not the average since process start — one
    slow minute during startup shouldn't hide in an hour of good samples.
    """

    def __init__(self, window: int = 512):
        self.window = int(window)
        self.reset()

    def reset(self) -> None:
        self._latencies: Deque[float] = deque(maxlen=self.window)
        self.counts: Dict[str, int] = {
            "ticks": 0, "replans": 0, "policy_errors": 0, "bad_frames": 0,
            "frozen_frames": 0, "black_frames": 0, "degraded_ticks": 0,
            "breaker_trips": 0, "budget_misses": 0, "nonfinite_chunks": 0,
        }

    def record(self, report: StepReport) -> None:
        self.counts["ticks"] += 1
        if report.replanned:
            self._latencies.append(report.latency_ms)
        if report.degraded:
            self.counts["degraded_ticks"] += 1

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def summary(self) -> Dict[str, float]:
        lat = np.asarray(self._latencies, dtype=np.float64)
        out: Dict[str, float] = dict(self.counts)
        if lat.size:
            out.update(
                latency_p50_ms=float(np.percentile(lat, 50)),
                latency_p95_ms=float(np.percentile(lat, 95)),
                latency_max_ms=float(lat.max()),
            )
        else:
            out.update(latency_p50_ms=0.0, latency_p95_ms=0.0, latency_max_ms=0.0)
        ticks = max(1, self.counts["ticks"])
        out["degraded_rate"] = self.counts["degraded_ticks"] / ticks
        return out


class PolicyRuntime:
    """Frame in, one legal gamepad action out — with the failure handling.

    Args:
        policy: ``pixels (H, W, 3) float32 in [0, 1] -> (T, ACTION_DIM)`` raw
            actions. Use :meth:`from_model` to wrap a :class:`~nitrogen.NitroGen`.
        hold_on_frozen: emit neutral while the capture is stalled. Acting on a
            stale frame is *worse* than idling: the policy is confidently
            controlling a world state that has already moved on.
        hold_on_black: same, for loading screens and occluded windows.
        max_consecutive_failures: policy errors in a row before the breaker opens.
        recover_after: ticks the breaker stays open before retrying the policy.
        latency_budget_ms: optional per-replan deadline; exceeding it is counted,
            never enforced by cancelling — a late action still beats no action.
    """

    def __init__(
        self,
        policy: PolicyFn,
        sanitizer: Optional[FrameSanitizer] = None,
        guard: Optional[ActionGuard] = None,
        executor: Optional[ChunkExecutor] = None,
        hold_on_frozen: bool = True,
        hold_on_black: bool = True,
        max_consecutive_failures: int = 3,
        recover_after: int = 30,
        latency_budget_ms: Optional[float] = None,
        metrics_window: int = 512,
    ):
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be >= 1")
        if recover_after < 1:
            raise ValueError("recover_after must be >= 1")
        self.policy = policy
        self.sanitizer = sanitizer if sanitizer is not None else FrameSanitizer()
        self.guard = guard if guard is not None else ActionGuard()
        self.executor = executor if executor is not None else ChunkExecutor()
        self.hold_on_frozen = hold_on_frozen
        self.hold_on_black = hold_on_black
        self.max_consecutive_failures = int(max_consecutive_failures)
        self.recover_after = int(recover_after)
        self.latency_budget_ms = latency_budget_ms
        self.metrics = RuntimeMetrics(window=metrics_window)
        self.reset()

    def reset(self) -> None:
        """Full reset — call between episodes, or after reconnecting capture."""
        self.sanitizer.reset()
        self.guard.reset()
        self.executor.reset()
        self.metrics.reset()
        self._failures = 0
        self._breaker_until: Optional[int] = None

    # -- breaker ----------------------------------------------------------
    @property
    def breaker_open(self) -> bool:
        return self._breaker_until is not None and self.executor.tick < self._breaker_until

    def _note_failure(self) -> None:
        self._failures += 1
        self.metrics.bump("policy_errors")
        if self._failures >= self.max_consecutive_failures:
            self._breaker_until = self.executor.tick + self.recover_after
            self._failures = 0
            self.metrics.bump("breaker_trips")

    # -- the tick ---------------------------------------------------------
    def step(self, frame) -> StepReport:
        """Process one captured frame and return the action to send to the game."""
        tick = self.executor.tick
        pixels, health = self.sanitizer(frame)
        reason, degraded, replanned, latency_ms = "ok", False, False, 0.0

        if pixels is None:
            # Unusable pixels are *not* a reason to stop acting: the buffered
            # chunk is still the best available estimate of what to do next.
            reason, degraded = f"bad_frame:{health.reason}", True
            self.metrics.bump("bad_frames")
        elif health.frozen and self.hold_on_frozen:
            reason, degraded = "frozen_capture", True
            self.metrics.bump("frozen_frames")
        elif health.black and self.hold_on_black:
            reason, degraded = "black_frame", True
            self.metrics.bump("black_frames")
        elif self.breaker_open:
            reason, degraded = "breaker_open", True
        elif self.executor.should_replan:
            replanned = True
            t0 = time.perf_counter()
            try:
                chunk = np.asarray(self.policy(pixels), dtype=np.float32)
            except Exception as exc:                      # noqa: BLE001 - by design
                latency_ms = (time.perf_counter() - t0) * 1e3
                self._note_failure()
                reason, degraded = f"policy_error:{type(exc).__name__}", True
            else:
                latency_ms = (time.perf_counter() - t0) * 1e3
                reason, degraded = self._accept_chunk(chunk)
                if self.latency_budget_ms is not None and latency_ms > self.latency_budget_ms:
                    self.metrics.bump("budget_misses")

        if reason in ("frozen_capture", "black_frame", "breaker_open"):
            # A deliberate idle: neutral, and don't let a stale buffered chunk
            # keep driving. The tick still advances so the clock stays honest.
            action = self.guard(np.zeros(ACTION_DIM, dtype=np.float32))
            self.executor.step()
        else:
            action = self.guard(self.executor.step())

        report = StepReport(
            action=action, tick=tick, reason=reason, replanned=replanned,
            degraded=degraded, latency_ms=latency_ms, health=health,
        )
        if replanned and reason == "ok":
            self.metrics.bump("replans")
        self.metrics.record(report)
        return report

    def _accept_chunk(self, chunk: np.ndarray):
        """Validate a sampled chunk before it can reach the game."""
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or chunk.shape[0] == 0:
            self._note_failure()
            return f"bad_chunk_shape:{tuple(chunk.shape)}", True
        if not np.isfinite(chunk).all():
            # A diverged sampler emits NaN chunks. Never let one reach the pad.
            self.metrics.bump("nonfinite_chunks")
            self._note_failure()
            return "nonfinite_chunk", True
        self.executor.submit(chunk)
        self._failures = 0
        self._breaker_until = None
        return "ok", False

    # -- construction from a trained model ---------------------------------
    @classmethod
    def from_model(
        cls,
        model,
        sample_steps: int = 10,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "PolicyRuntime":
        """Wrap a :class:`~nitrogen.NitroGen` checkpoint as a runtime.

        ``seed`` makes the rollout **reproducible**: the sampler is re-seeded
        from ``seed + tick`` before every chunk, so the same frames produce the
        same actions on every run. Flow matching starts from fresh Gaussian
        noise each call, so without this a regression run against a fixed set of
        frames is not comparable to the previous one — a nondeterministic
        baseline can't tell a real regression from resampling.
        """
        import torch

        holder: Dict[str, PolicyRuntime] = {}

        def policy(pixels: np.ndarray) -> np.ndarray:
            if seed is not None:
                rt = holder.get("rt")
                torch.manual_seed(seed + (rt.executor.tick if rt is not None else 0))
            return model.act(pixels, num_steps=sample_steps)

        runtime = cls(policy, **kwargs)
        holder["rt"] = runtime
        return runtime
