"""End-to-end NitroGen quickstart in one file.

Runs the *entire* pipeline at tiny scale so you can watch every concept fire:

    1. build a synthetic video-action dataset (expert rollouts),
    2. behavior-clone a NitroGen policy with flow matching,
    3. evaluate closed-loop success on the multi-game benchmark,
    4. probe cross-game transfer to a held-out (unseen) game.

Run it with::

    python examples/quickstart.py

It is deliberately small (a few minutes on CPU). Bump ``steps`` /
``episodes_per_game`` for stronger policies.
"""

from __future__ import annotations

import torch

from nitrogen.benchmark.evaluate import evaluate_suite, transfer_report
from nitrogen.data.overlay_extraction import evaluate_extraction
from nitrogen.action_space import GamepadAction, BUTTON_NAMES
from nitrogen.training.config import TrainConfig
from nitrogen.training.trainer import train
import numpy as np


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(4)

    # 0) Pillar 1 sanity check: can we recover actions from a controller overlay?
    rng = np.random.default_rng(0)
    probe = [
        GamepadAction(
            left_x=float(rng.uniform(-1, 1)), left_y=float(rng.uniform(-1, 1)),
            buttons={n: bool(rng.random() < 0.3) for n in BUTTON_NAMES},
        )
        for _ in range(100)
    ]
    m = evaluate_extraction(probe)
    print(f"[overlay extraction] joystick R^2={m['joystick_r2']:.3f}  "
          f"button acc={m['button_accuracy']:.3f}")

    # 1-2) Collect data and behavior-clone a policy.
    cfg = TrainConfig(games=["reacher", "dodger"], episodes_per_game=40,
                      steps=800, batch_size=48, log_every=100)
    print("\n[training] behavior cloning with flow matching ...")
    _, ema = train(cfg)
    policy = ema.shadow

    # 3) Benchmark on the games it trained on.
    print("\n[benchmark] closed-loop success on seen games:")
    scores = evaluate_suite(policy, ["reacher", "dodger"], episodes=20)
    for g, s in scores.items():
        print(f"    {g:10s}: {s:.0%}")

    # 4) Transfer: a game the policy never saw during training.
    print("\n[transfer] seen vs unseen games:")
    report = transfer_report(policy, episodes=20)
    for split, sc in report.items():
        print(f"    {split:6s}: " + ", ".join(f"{g}={v:.0%}" for g, v in sc.items()))


if __name__ == "__main__":
    main()
