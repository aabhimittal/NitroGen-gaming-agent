"""Evaluate a trained NitroGen checkpoint on the multi-game benchmark.

Usage::

    python -m scripts.evaluate --ckpt checkpoints/nitrogen.pt --episodes 30
    python -m scripts.evaluate --ckpt checkpoints/nitrogen.pt --transfer
    python -m scripts.evaluate --ckpt checkpoints/nitrogen.pt --few-shot chaser
    python -m scripts.evaluate --ckpt checkpoints/nitrogen.pt --robustness
"""

from __future__ import annotations

import argparse

from nitrogen.benchmark.evaluate import evaluate_suite, few_shot_transfer, transfer_report
from nitrogen.benchmark.robustness import format_report, robustness_report
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
                   help="report seen vs held-out (unseen) game success (zero-shot)")
    p.add_argument("--few-shot", type=str, metavar="GAME", default=None,
                   help="few-shot transfer: fine-tune on a few episodes of GAME vs from scratch")
    p.add_argument("--robustness", action="store_true",
                   help="success under capture faults, naive vs. deployment runtime")
    p.add_argument("--latency-steps", type=int, default=0,
                   help="ticks of inference latency to simulate in --robustness")
    args = p.parse_args()

    model = load_checkpoint(args.ckpt)
    kwargs = dict(episodes=args.episodes, replan_every=args.replan_every,
                  sample_steps=args.sample_steps)

    if args.robustness:
        report = robustness_report(
            model, games=args.games, episodes=args.episodes,
            replan_every=args.replan_every, sample_steps=args.sample_steps,
            latency_steps=args.latency_steps,
        )
        print("Robustness to capture-pipeline faults (success rate):")
        print(format_report(report))
    elif args.few_shot:
        r = few_shot_transfer(model, game=args.few_shot)
        rel = r["relative_improvement"]
        rel_str = f"{rel:+.0%}" if rel is not None else "n/a (from-scratch ≈ 0%)"
        print(f"Few-shot transfer to '{r['game']}' ({r['n_episodes']} episodes):")
        print(f"  from scratch : {r['from_scratch']:.2%}")
        print(f"  pretrained   : {r['pretrained']:.2%}")
        print(f"  absolute gain: {r['absolute_improvement']:+.2%}")
        print(f"  relative gain: {rel_str}")
    elif args.transfer:
        report = transfer_report(model, **kwargs)
        print("Transfer report (zero-shot success rate):")
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
