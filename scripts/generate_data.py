"""Generate a synthetic gameplay dataset and (optionally) demo overlay extraction.

Usage::

    python -m scripts.generate_data --games reacher avoider --episodes 80 --out data/train.npz
    python -m scripts.generate_data --overlay-demo   # reproduce the R^2 / accuracy metrics
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from nitrogen.action_space import BUTTON_NAMES, GamepadAction
from nitrogen.data.overlay_extraction import evaluate_extraction
from nitrogen.data.synthetic import collect_dataset, save_episodes
from nitrogen.envs.toy_game import TRAIN_GAMES


def overlay_demo(n: int = 300) -> None:
    rng = np.random.default_rng(0)
    actions = []
    for _ in range(n):
        buttons = {name: bool(rng.random() < 0.3) for name in BUTTON_NAMES}
        actions.append(
            GamepadAction(
                left_x=float(rng.uniform(-1, 1)), left_y=float(rng.uniform(-1, 1)),
                right_x=float(rng.uniform(-1, 1)), right_y=float(rng.uniform(-1, 1)),
                buttons=buttons,
            )
        )
    metrics = evaluate_extraction(actions)
    print("Overlay action-extraction quality (synthetic):")
    print(f"  joystick R^2     : {metrics['joystick_r2']:.3f}   (paper: 0.84)")
    print(f"  button accuracy  : {metrics['button_accuracy']:.3f}   (paper: 0.96)")
    print(f"  frames evaluated : {metrics['n']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", nargs="+", default=TRAIN_GAMES)
    p.add_argument("--episodes", type=int, default=80, help="episodes per game")
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--out", type=str, default="data/train.npz")
    p.add_argument("--overlay-demo", action="store_true",
                   help="run the overlay-extraction quality demo and exit")
    args = p.parse_args()

    if args.overlay_demo:
        overlay_demo()
        return

    print(f"Collecting {args.episodes} episodes/game for {args.games} ...")
    episodes = collect_dataset(args.games, episodes_per_game=args.episodes, noise=args.noise)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_episodes(episodes, args.out)
    total_frames = sum(len(e.frames) for e in episodes)
    print(f"Saved {len(episodes)} episodes ({total_frames} frames) to {args.out}")


if __name__ == "__main__":
    main()
