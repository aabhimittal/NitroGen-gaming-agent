"""Render a trained NitroGen agent playing a game to an animated GIF.

Usage::

    python -m scripts.demo --ckpt checkpoints/nitrogen.pt --game reacher --out assets/reacher.gif
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from nitrogen.data.overlay_extraction import render_overlay
from nitrogen.envs.toy_game import make_game
from nitrogen.training.trainer import load_checkpoint


def rollout_frames(model, game_name: str, seed: int, replan_every: int, sample_steps: int,
                   draw_overlay: bool) -> tuple[list[np.ndarray], bool]:
    """Play one episode and collect frames (optionally with the predicted-action HUD)."""
    from nitrogen.action_space import GamepadAction

    env = make_game(game_name)
    frame = env.reset(seed=seed)
    frames, done, info = [], False, {"success": False}
    chunk, idx = None, 0
    while not done:
        if chunk is None or idx >= replan_every:
            chunk = model.act(frame, num_steps=sample_steps)
            idx = 0
        action = chunk[idx]
        idx += 1
        shown = frame.copy()
        if draw_overlay:
            shown = render_overlay(shown, GamepadAction.from_vector(action))
        frames.append(shown)
        frame, done, info = env.step(action)
    return frames, bool(info["success"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--game", type=str, default="reacher")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--replan-every", type=int, default=4)
    p.add_argument("--sample-steps", type=int, default=10)
    p.add_argument("--no-overlay", action="store_true", help="hide the predicted-action HUD")
    p.add_argument("--out", type=str, default="assets/demo.gif")
    args = p.parse_args()

    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("imageio is required for the demo: pip install imageio")

    model = load_checkpoint(args.ckpt)
    all_frames = []
    for e in range(args.episodes):
        frames, success = rollout_frames(
            model, args.game, seed=args.seed + e,
            replan_every=args.replan_every, sample_steps=args.sample_steps,
            draw_overlay=not args.no_overlay,
        )
        print(f"episode {e}: {'SUCCESS' if success else 'fail'} ({len(frames)} frames)")
        all_frames.extend(frames)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, all_frames, duration=0.05)
    print(f"Wrote {len(all_frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
