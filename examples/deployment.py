"""Drive a policy through the deployment runtime against a deliberately broken capture.

Runs one episode of a toy game where the "capture pipeline" misbehaves on a
rotation: dropped frames, a channel-first BGR buffer, float frames still in
[0, 255] with NaN pixels, a black loading screen, a stalled capture. The naive
loop loses its action on most of those; the runtime sanitizes, degrades, and
keeps producing a legal action every single tick.

No checkpoint needed — an untrained model demonstrates the *mechanics* (the
actions are random either way). Point it at a real checkpoint to see the
robustness numbers move::

    python -m examples.deployment
    python -m examples.deployment --ckpt checkpoints/nitrogen.pt
"""

from __future__ import annotations

import argparse

import numpy as np

from nitrogen import NitroGen, NitroGenConfig
from nitrogen.deploy import ChunkExecutor, PolicyRuntime
from nitrogen.envs.toy_game import make_game
from nitrogen.models.vision_encoder import VisionConfig

CORRUPTIONS = {
    "clean": lambda f, rng: f,
    "dropped": lambda f, rng: None,
    "channel_first_bgr": lambda f, rng: np.transpose(f[:, :, ::-1], (2, 0, 1)),
    "float_255_with_nan": lambda f, rng: np.where(
        rng.random(f.shape) < 0.001, np.nan, f.astype(np.float32)),
    "grayscale": lambda f, rng: f[:, :, 0],
    "rgba": lambda f, rng: np.concatenate(
        [f, np.full(f.shape[:2] + (1,), 255, np.uint8)], axis=2),
    "black_screen": lambda f, rng: np.zeros_like(f),
    "uint16": lambda f, rng: f.astype(np.uint16) * 257,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--game", type=str, default="reacher")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-steps", type=int, default=4)
    args = p.parse_args()

    if args.ckpt:
        from nitrogen.training.trainer import load_checkpoint
        model = load_checkpoint(args.ckpt)
    else:
        model = NitroGen(NitroGenConfig(
            chunk_size=8,
            vision=VisionConfig(image_size=64, patch_size=16, width=64,
                                depth=2, num_heads=4),
        ))

    runtime = PolicyRuntime.from_model(
        model, sample_steps=args.sample_steps, seed=args.seed,
        executor=ChunkExecutor(replan_every=4, latency_steps=1, fallback="decay"),
    )

    env = make_game(args.game)
    frame = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    names = list(CORRUPTIONS)
    done, tick = False, 0

    print(f"{'tick':>4}  {'capture fault':<20}{'runtime verdict':<28}action[left_x]")
    while not done:
        name = names[tick % len(names)]
        observed = CORRUPTIONS[name](frame, rng)
        report = runtime.step(observed)
        print(f"{tick:>4}  {name:<20}{report.reason:<28}{report.action[0]:+.3f}")
        assert np.isfinite(report.action).all()          # the invariant, every tick
        frame, done, info = env.step(report.action)
        tick += 1

    print(f"\nepisode finished in {tick} ticks, success={info['success']}")
    summary = runtime.metrics.summary()
    print("runtime metrics:")
    for key in ("ticks", "replans", "bad_frames", "frozen_frames", "black_frames",
                "policy_errors", "breaker_trips", "degraded_rate",
                "latency_p50_ms", "latency_p95_ms"):
        value = summary[key]
        print(f"  {key:<18}: {value:.3f}" if isinstance(value, float) else
              f"  {key:<18}: {value}")
    print("\nguard interventions:", dict(runtime.guard.stats))


if __name__ == "__main__":
    main()
