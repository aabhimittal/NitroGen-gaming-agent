"""Train a NitroGen policy with behavior cloning.

Usage::

    python -m scripts.train --games reacher dodger --steps 1500 --out checkpoints/nitrogen.pt
"""

from __future__ import annotations

import argparse

import torch

from nitrogen.models.nitrogen import NitroGenConfig
from nitrogen.training.config import TrainConfig
from nitrogen.training.trainer import save_checkpoint, train
from nitrogen.envs.toy_game import TRAIN_GAMES


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", nargs="+", default=TRAIN_GAMES)
    p.add_argument("--episodes", type=int, default=80, help="episodes per game")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="checkpoints/nitrogen.pt")
    args = p.parse_args()

    cfg = TrainConfig(
        games=args.games,
        episodes_per_game=args.episodes,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        chunk_size=args.chunk_size,
        device=args.device,
    )
    model_cfg = NitroGenConfig(chunk_size=args.chunk_size)
    print(f"Training NitroGen on {args.games} for {args.steps} steps ({args.device}) ...")
    model, ema = train(cfg, model_cfg=model_cfg)

    # Save the EMA weights — they are what you evaluate/deploy.
    ema.shadow.to("cpu")
    save_checkpoint(ema.shadow, args.out)
    print(f"Saved EMA checkpoint to {args.out}  ({model.num_parameters()/1e6:.1f}M params)")


if __name__ == "__main__":
    main()
