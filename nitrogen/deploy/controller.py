"""Executing an action *chunk* against a game that runs on its own clock.

The policy emits ``chunk_size`` actions from one frame. The game wants exactly
one action per tick, forever. Bridging those two rates is where a chunked policy
actually lives or dies in deployment:

* **Chunk boundaries stutter.** Replan every T ticks and the agent's intent
  changes discontinuously at every seam — the stick snaps because chunk *k+1*
  was generated from a frame the game has already moved past. Overlapping chunks
  and averaging them (temporal ensembling, as in ACT) smooths the seam without
  needing a slower replan.
* **Inference is not free.** A chunk generated from the frame at tick *k* only
  reaches the game at tick *k+L*, where *L* is the sampling latency in ticks. Its
  first *L* actions are already stale on arrival, so executing them replays the
  past. ``latency_steps`` skips them.
* **Chunks run out.** If sampling stalls — a slow GPU, a hitch, an exception —
  the executor is asked for an action it doesn't have. Repeating the last action
  forever is how an agent walks into a wall for ten seconds; the fallback policy
  makes that choice explicit instead of accidental.

Everything here is in *raw* gamepad space (sticks ``[-1, 1]``, triggers
``[0, 1]``, buttons ``{0, 1}``) — the space :meth:`nitrogen.NitroGen.act`
returns.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np

from nitrogen.action_space import ACTION_DIM, BUTTON_SLICE

FALLBACKS = ("neutral", "hold", "decay")


@dataclass
class _PendingChunk:
    start_tick: int          # first game tick this chunk is valid for
    actions: np.ndarray      # (T, ACTION_DIM) raw


class ChunkExecutor:
    """Turn a stream of overlapping action chunks into one action per tick.

    Args:
        replan_every: ticks between replans. Smaller = more reactive, more
            compute. It is independent of ``chunk_size``; chunks are expected to
            be longer than the replan interval, which is what creates the
            overlap that ensembling averages over.
        latency_steps: ticks of inference latency to compensate for by skipping
            the head of each arriving chunk.
        ensemble: average all chunks covering the current tick instead of using
            only the newest one.
        ensemble_decay: weight of each successively older chunk (``1.0`` = plain
            mean, small values = trust the freshest chunk).
        fallback: what to emit when no chunk covers the current tick —
            ``"neutral"`` (release everything), ``"hold"`` (repeat the last
            action), or ``"decay"`` (relax the last action toward neutral).
        max_pending: cap on retained chunks, so a long run can't grow unbounded.
    """

    def __init__(
        self,
        replan_every: int = 4,
        latency_steps: int = 0,
        ensemble: bool = True,
        ensemble_decay: float = 0.5,
        fallback: str = "neutral",
        decay_rate: float = 0.5,
        max_pending: int = 8,
    ):
        if replan_every < 1:
            raise ValueError("replan_every must be >= 1")
        if latency_steps < 0:
            raise ValueError("latency_steps must be >= 0")
        if not 0.0 < ensemble_decay <= 1.0:
            raise ValueError("ensemble_decay must be in (0, 1]")
        if fallback not in FALLBACKS:
            raise ValueError(f"fallback must be one of {FALLBACKS}, got {fallback!r}")
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        self.replan_every = int(replan_every)
        self.latency_steps = int(latency_steps)
        self.ensemble = bool(ensemble)
        self.ensemble_decay = float(ensemble_decay)
        self.fallback = fallback
        self.decay_rate = float(decay_rate)
        self.max_pending = int(max_pending)
        self.reset()

    def reset(self) -> None:
        """Clear all pending chunks and restart the tick clock."""
        self.tick = 0
        self._pending: Deque[_PendingChunk] = deque(maxlen=self.max_pending)
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._last_submit_tick: Optional[int] = None
        self.stats: Dict[str, int] = {
            "ticks": 0, "submitted": 0, "starved": 0, "ensembled": 0, "dropped_stale": 0,
        }

    # -- producing side ---------------------------------------------------
    def submit(self, chunk: np.ndarray) -> None:
        """Hand the executor a freshly sampled ``(T, ACTION_DIM)`` chunk."""
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
            raise ValueError(f"expected a (T, {ACTION_DIM}) chunk, got shape {arr.shape}")
        if arr.shape[0] == 0:
            raise ValueError("refusing to submit an empty chunk")
        if not np.isfinite(arr).all():
            raise ValueError("chunk contains non-finite values; sanitize before submitting")

        # Drop the head that inference latency already consumed. A chunk shorter
        # than the latency is entirely stale — keeping it would execute actions
        # chosen for a frame that is now several ticks old.
        if self.latency_steps >= arr.shape[0]:
            self.stats["dropped_stale"] += 1
            self._last_submit_tick = self.tick
            return
        arr = arr[self.latency_steps:]

        if len(self._pending) == self._pending.maxlen:
            self.stats["dropped_stale"] += 1  # deque evicts the oldest for us
        self._pending.append(_PendingChunk(start_tick=self.tick, actions=arr))
        self._last_submit_tick = self.tick
        self.stats["submitted"] += 1

    @property
    def should_replan(self) -> bool:
        """True when the caller should sample a new chunk before the next tick."""
        if self._last_submit_tick is None:
            return True
        if self.tick - self._last_submit_tick >= self.replan_every:
            return True
        return not self._covering()      # exhausted early: replan now, not later

    # -- consuming side ---------------------------------------------------
    def _covering(self):
        """Pending chunks that have an action for the current tick, newest first."""
        out = []
        for pc in reversed(self._pending):
            offset = self.tick - pc.start_tick
            if 0 <= offset < pc.actions.shape[0]:
                out.append((pc, offset))
        return out

    def step(self) -> np.ndarray:
        """Emit the action for the current tick and advance the clock."""
        covering = self._covering()
        self.stats["ticks"] += 1

        if not covering:
            self.stats["starved"] += 1
            action = self._fallback_action()
        elif not self.ensemble or len(covering) == 1:
            action = covering[0][0].actions[covering[0][1]].copy()
        else:
            self.stats["ensembled"] += 1
            action = self._blend(covering)

        self._last_action = action.astype(np.float32)
        self.tick += 1
        self._evict()
        return self._last_action.copy()

    def _blend(self, covering) -> np.ndarray:
        """Exponentially weighted average over the chunks covering this tick."""
        weights = np.array(
            [self.ensemble_decay ** i for i in range(len(covering))], dtype=np.float32
        )
        weights /= weights.sum()
        stacked = np.stack([pc.actions[off] for pc, off in covering])
        blended = (stacked * weights[:, None]).sum(axis=0)
        # Analog channels average meaningfully; buttons do not — averaging two
        # disagreeing chunks would emit a half-pressed button. Vote instead.
        blended[BUTTON_SLICE] = (blended[BUTTON_SLICE] >= 0.5).astype(np.float32)
        return blended.astype(np.float32)

    def _fallback_action(self) -> np.ndarray:
        if self.fallback == "hold":
            return self._last_action.copy()
        if self.fallback == "decay":
            out = self._last_action * (1.0 - self.decay_rate)
            # Buttons have no meaningful midpoint — release them immediately
            # rather than letting a scaled 0.4 read as "still pressed".
            out[BUTTON_SLICE] = 0.0
            return out.astype(np.float32)
        return np.zeros(ACTION_DIM, dtype=np.float32)

    def _evict(self) -> None:
        """Forget chunks whose last action is in the past."""
        while self._pending:
            pc = self._pending[0]
            if self.tick - pc.start_tick >= pc.actions.shape[0]:
                self._pending.popleft()
            else:
                break

    # -- diagnostics -------------------------------------------------------
    @property
    def pending_chunks(self) -> int:
        return len(self._pending)

    @property
    def horizon_remaining(self) -> int:
        """Ticks of buffered actions left before the executor starves."""
        covering = self._covering()
        if not covering:
            return 0
        return max(pc.actions.shape[0] - off for pc, off in covering)
