"""Behavior-cloning trainer for NitroGen.

NitroGen is trained with **pure behavior cloning** — no rewards, no RL, no
environment interaction during training. The only objective is the flow-matching
loss between the model's velocity field and the expert action chunk (see
:meth:`nitrogen.models.nitrogen.NitroGen.compute_loss`).

This trainer reproduces the paper's recipe at small scale:

    * **AdamW** optimizer with decoupled weight decay,
    * a **warmup-stable-decay (WSD)** learning-rate schedule,
    * **EMA** (exponential moving average) of the weights for evaluation,
    * light **image augmentation** (brightness/contrast jitter) for robustness.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Callable, List

import torch
from torch.utils.data import DataLoader

from nitrogen.data.dataset import FrameActionChunkDataset
from nitrogen.data.synthetic import Episode, collect_dataset
from nitrogen.models.nitrogen import NitroGen, NitroGenConfig
from nitrogen.training.config import TrainConfig


def wsd_lr(step: int, total: int, warmup_frac: float, decay_frac: float) -> float:
    """Warmup-stable-decay multiplier in [0, 1].

    Linear warmup -> constant plateau -> cosine decay to ~0. WSD is popular for
    foundation models because the long stable phase can be extended/resumed
    without recomputing a cosine schedule.
    """
    warmup = max(1, int(total * warmup_frac))
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return step / warmup
    if step < decay_start:
        return 1.0
    progress = (step - decay_start) / max(1, total - decay_start)
    return 0.5 * (1 + math.cos(math.pi * progress))


class EMA:
    """Tracks an exponential moving average of a module's parameters."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def augment(frames: torch.Tensor, brightness: float, contrast: float) -> torch.Tensor:
    """Per-sample brightness/contrast jitter on a (B, 3, H, W) batch in [0, 1]."""
    if brightness <= 0 and contrast <= 0:
        return frames
    b = frames.shape[0]
    device = frames.device
    bright = 1 + (torch.rand(b, 1, 1, 1, device=device) * 2 - 1) * brightness
    cont = 1 + (torch.rand(b, 1, 1, 1, device=device) * 2 - 1) * contrast
    mean = frames.mean(dim=(1, 2, 3), keepdim=True)
    out = (frames - mean) * cont + mean * bright
    return out.clamp(0, 1)


def build_dataset(cfg: TrainConfig) -> List[Episode]:
    return collect_dataset(
        cfg.games,
        episodes_per_game=cfg.episodes_per_game,
        noise=cfg.expert_noise,
        seed=cfg.seed,
    )


def train(
    cfg: TrainConfig | None = None,
    model_cfg: NitroGenConfig | None = None,
    episodes: List[Episode] | None = None,
    on_log: Callable[[int, float, float], None] | None = None,
) -> tuple[NitroGen, EMA]:
    """Run behavior cloning. Returns the trained model and its EMA copy."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    if episodes is None:
        episodes = build_dataset(cfg)
    dataset = FrameActionChunkDataset(episodes, chunk_size=cfg.chunk_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = NitroGen(model_cfg or NitroGenConfig(chunk_size=cfg.chunk_size)).to(device)
    model.train()
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def infinite(dl):
        while True:
            yield from dl

    stream = infinite(loader)
    running = 0.0
    for step in range(cfg.steps):
        frames, chunks = next(stream)
        frames = augment(frames.to(device), cfg.aug_brightness, cfg.aug_contrast)
        chunks = chunks.to(device)

        loss = model.compute_loss(frames, chunks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        lr_mult = wsd_lr(step, cfg.steps, cfg.warmup_frac, cfg.decay_frac)
        for g in opt.param_groups:
            g["lr"] = cfg.lr * lr_mult
        opt.step()
        ema.update(model)

        running += loss.item()
        if (step + 1) % cfg.log_every == 0:
            avg = running / cfg.log_every
            running = 0.0
            if on_log:
                on_log(step + 1, avg, cfg.lr * lr_mult)
            else:
                print(f"step {step+1:5d}/{cfg.steps}  loss {avg:.4f}  lr {cfg.lr*lr_mult:.2e}")

    return model, ema


def save_checkpoint(model: NitroGen, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.cfg}, path)


def load_checkpoint(path: str, map_location="cpu") -> NitroGen:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = NitroGen(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
