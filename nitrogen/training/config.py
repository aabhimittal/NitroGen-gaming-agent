"""Training configuration for NitroGen behavior cloning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from nitrogen.envs.toy_game import TRAIN_GAMES


@dataclass
class TrainConfig:
    # data
    games: List[str] = field(default_factory=lambda: list(TRAIN_GAMES))
    episodes_per_game: int = 80
    expert_noise: float = 0.15
    chunk_size: int = 16

    # optimization
    batch_size: int = 64
    steps: int = 1500
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # warmup-stable-decay LR schedule (fractions of total steps)
    warmup_frac: float = 0.05
    decay_frac: float = 0.2

    # exponential moving average of weights
    ema_decay: float = 0.999

    # image augmentation strength (0 disables)
    aug_brightness: float = 0.1
    aug_contrast: float = 0.1

    # bookkeeping
    log_every: int = 50
    seed: int = 0
    device: str = "cpu"
    out_dir: str = "checkpoints"
