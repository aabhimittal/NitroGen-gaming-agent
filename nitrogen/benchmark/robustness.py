"""Closed-loop robustness: how much success survives a hostile capture pipeline.

:mod:`nitrogen.benchmark.evaluate` measures competence under *ideal* conditions —
pixels straight from ``env.render()``, an action delivered every tick, no
latency. Deployment offers none of that, and a policy that scores 90% in the lab
can score near zero against a capture stack that drops frames, adds a couple of
ticks of inference latency, or hands over slightly wrong colours.

This module measures that gap directly. Each **perturbation** below is a
mechanical fault of the *capture and control path*, not a change to the game:
the task, the seeds, and the policy are identical across conditions, so the
difference in success rate is attributable to the fault alone.

Two numbers come out of a run:

    * ``raw`` — the policy driven the naive way (``model.act`` on whatever the
      capture handed over), which is what the research loop does.
    * ``guarded`` — the same policy behind :class:`~nitrogen.deploy.PolicyRuntime`.

The delta is the measured value of the deployment layer, per fault. Reporting
both matters: hardening that doesn't move the number is complexity you should
delete, and a fault the runtime *can't* recover from is worth knowing about
before it happens in production rather than after.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from nitrogen.action_space import ACTION_DIM
from nitrogen.deploy.controller import ChunkExecutor
from nitrogen.deploy.frames import FrameSanitizer
from nitrogen.deploy.runtime import PolicyRuntime
from nitrogen.deploy.safety import ActionGuard, GamepadLimits
from nitrogen.envs.toy_game import make_game
from nitrogen.models.nitrogen import NitroGen

Perturbation = Callable[[np.ndarray, int, np.random.Generator], Optional[np.ndarray]]


# ---------------------------------------------------------------------------
# Faults. Each takes (frame, tick, rng) and returns the frame the *agent* sees —
# or None, meaning the capture produced nothing this tick.
# ---------------------------------------------------------------------------

def clean(frame: np.ndarray, tick: int, rng: np.random.Generator):
    return frame


def dropped_frames(p: float = 0.2) -> Perturbation:
    """Capture returns nothing on a fraction of ticks (a busy or throttled GPU)."""
    def fault(frame, tick, rng):
        return None if rng.random() < p else frame
    return fault


def frozen_capture(stall_every: int = 12, stall_for: int = 4) -> Perturbation:
    """Capture periodically stalls and repeats its last frame for a few ticks."""
    held = {"frame": None, "left": 0}

    def fault(frame, tick, rng):
        if held["left"] > 0:
            held["left"] -= 1
            return held["frame"]
        if tick > 0 and tick % stall_every == 0:
            held["frame"], held["left"] = frame, stall_for - 1
            return frame
        return frame
    return fault


def sensor_noise(sigma: float = 0.08) -> Perturbation:
    """Compression / encoder noise on an 8-bit capture."""
    def fault(frame, tick, rng):
        noisy = frame.astype(np.float32) + rng.normal(0, sigma * 255, frame.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    return fault


def wrong_layout(frame: np.ndarray, tick: int, rng: np.random.Generator):
    """BGR channel order and a channel-first buffer — the two classic wiring bugs."""
    return np.transpose(frame[:, :, ::-1], (2, 0, 1))


def hdr_float(frame: np.ndarray, tick: int, rng: np.random.Generator):
    """Float frames left in ``[0, 255]``, with occasional non-finite pixels."""
    out = frame.astype(np.float32)
    if tick % 7 == 0:
        out[0, 0] = np.nan
    return out


def rescaled(height: int = 90, width: int = 210) -> Perturbation:
    """An ultrawide capture at a resolution the model never trained on."""
    from nitrogen.deploy.frames import resize_nearest

    def fault(frame, tick, rng):
        return resize_nearest(frame, height, width)
    return fault


def brightness_shift(gain: float = 1.6, bias: float = -30.0) -> Perturbation:
    """A different display profile / HDR tone curve than the training frames."""
    def fault(frame, tick, rng):
        return np.clip(frame.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    return fault


DEFAULT_PERTURBATIONS: Dict[str, Perturbation] = {
    "clean": clean,
    "dropped_frames": dropped_frames(0.2),
    "frozen_capture": frozen_capture(),
    "sensor_noise": sensor_noise(),
    "wrong_layout": wrong_layout,
    "hdr_float": hdr_float,
    "ultrawide": rescaled(),
    "brightness": brightness_shift(),
}


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

def _neutral() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype=np.float32)


@torch.no_grad()
def run_raw_episode(
    model: NitroGen,
    game: str,
    seed: int,
    perturb: Perturbation,
    replan_every: int = 4,
    sample_steps: int = 10,
    latency_steps: int = 0,
) -> bool:
    """The naive loop: hand whatever the capture produced straight to the model.

    This is the *baseline*, and it is deliberately fragile in the way real
    research loops are: a frame the model can't consume costs the agent its
    action for that tick, and any exception is caught only to keep the sweep
    running.
    """
    env = make_game(game)
    frame = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    chunk, idx, tick, done = None, 0, 0, False
    while not done:
        observed = perturb(frame, tick, rng)
        if chunk is None or idx >= replan_every:
            try:
                chunk = model.act(observed, num_steps=sample_steps)
                if latency_steps:
                    chunk = chunk[latency_steps:] if latency_steps < len(chunk) else None
                idx = 0
            except Exception:                       # noqa: BLE001 - baseline is fragile
                chunk = None
        if chunk is None or idx >= len(chunk):
            action = _neutral()
        else:
            action = chunk[idx]
            idx += 1
        if not np.isfinite(action).all():
            action = _neutral()
        frame, done, info = env.step(action)
        tick += 1
        if info["success"]:
            return True
    return bool(info["success"])


@torch.no_grad()
def run_guarded_episode(
    model: NitroGen,
    game: str,
    seed: int,
    perturb: Perturbation,
    replan_every: int = 4,
    sample_steps: int = 10,
    latency_steps: int = 0,
    limits: Optional[GamepadLimits] = None,
) -> bool:
    """The same policy and seeds, driven through the deployment runtime."""
    env = make_game(game)
    frame = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    runtime = PolicyRuntime.from_model(
        model,
        sample_steps=sample_steps,
        executor=ChunkExecutor(replan_every=replan_every, latency_steps=latency_steps),
        guard=ActionGuard(limits),
        sanitizer=FrameSanitizer(),
    )
    tick, done = 0, False
    while not done:
        report = runtime.step(perturb(frame, tick, rng))
        frame, done, info = env.step(report.action)
        tick += 1
        if info["success"]:
            return True
    return bool(info["success"])


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def robustness_report(
    model: NitroGen,
    games: Optional[List[str]] = None,
    episodes: int = 15,
    perturbations: Optional[Dict[str, Perturbation]] = None,
    seed_offset: int = 50_000,
    **rollout_kwargs,
) -> Dict[str, Dict[str, float]]:
    """Success rate per fault, naive vs. guarded, averaged over ``games``.

    Every condition reuses the same seeds, so differences come from the fault
    and the handling — not from an easier draw of episodes.
    """
    games = games or ["reacher", "avoider"]
    perturbations = perturbations or DEFAULT_PERTURBATIONS
    model.eval()

    report: Dict[str, Dict[str, float]] = {}
    for name, perturb in perturbations.items():
        raw_wins = guarded_wins = total = 0
        for game in games:
            for e in range(episodes):
                s = seed_offset + e
                raw_wins += run_raw_episode(model, game, s, perturb, **rollout_kwargs)
                guarded_wins += run_guarded_episode(model, game, s, perturb, **rollout_kwargs)
                total += 1
        raw = raw_wins / max(1, total)
        guarded = guarded_wins / max(1, total)
        report[name] = {"raw": raw, "guarded": guarded, "delta": guarded - raw}
    return report


def format_report(report: Dict[str, Dict[str, float]]) -> str:
    """Render :func:`robustness_report` output as a fixed-width table."""
    lines = [f"{'perturbation':<16}{'raw':>9}{'guarded':>10}{'delta':>9}"]
    for name, row in report.items():
        lines.append(
            f"{name:<16}{row['raw']:>8.1%}{row['guarded']:>10.1%}{row['delta']:>+9.1%}"
        )
    return "\n".join(lines)
