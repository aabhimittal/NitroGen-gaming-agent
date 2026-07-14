"""Evaluate a trained NitroGen checkpoint on the multi-game benchmark.

Usage::

    python -m scripts.evaluate --ckpt checkpoints/nitrogen.pt --episodes 30
    python -m scripts.evaluate --ckpt checkpoints/nitrogen.pt --transfer
"""

from __future__ import annotations

import argparse

from nitrogen.benchmark.evaluate import evaluate_suite, transfer_report
from nitrogen.envs.toy_game import GAME_REGISTRY
from nitrogen.training.trainer import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--games", nargs="+", default=list(GAME_REGISTRY))
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--replan-every", type=int, default=4)
    p.add_argument("--sample-steps", type=int, default=10)
    p.add_argument("--transfer", action="store_true",
                   help="report seen vs held-out (unseen) game success")
    args = p.parse_args()

    model = load_checkpoint(args.ckpt)
    kwargs = dict(episodes=args.episodes, replan_every=args.replan_every,
                  sample_steps=args.sample_steps)

    if args.transfer:
        report = transfer_report(model, **kwargs)
        print("Transfer report (success rate):")
        for split, scores in report.items():
            print(f"  [{split}]")
            for g, s in scores.items():
                print(f"      {g:10s}: {s:.2%}")
    else:
        scores = evaluate_suite(model, args.games, **kwargs)
        print("Benchmark success rate:")
        for g, s in scores.items():
            print(f"  {g:10s}: {s:.2%}")


if __name__ == "__main__":
    main()
