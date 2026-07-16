"""Closed-loop evaluation — the multi-game benchmark.

Training is offline behavior cloning, but the thing we actually care about is
*closed-loop* competence: drop the policy into a live game and measure task
success. This module rolls the policy out in the toy games with **action
chunking**:

    1. observe one frame,
    2. generate a chunk of ``T`` actions with :meth:`NitroGen.sample_actions`,
    3. execute the first ``replan_every`` of them open-loop,
    4. re-observe and repeat.

Chunking amortizes the (relatively expensive) generative forward pass over
several environment steps and keeps behavior temporally smooth — a key reason
NitroGen predicts chunks rather than single actions.

The public helpers:
    * :func:`evaluate_game`      — success rate on one game.
    * :func:`evaluate_suite`     — success rate across a set of games (the benchmark).
    * :func:`transfer_report`    — seen vs. held-out (unseen) game comparison.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from nitrogen.action_space import decode_action
from nitrogen.envs.toy_game import HELDOUT_GAMES, TRAIN_GAMES, make_game
from nitrogen.models.nitrogen import NitroGen


@torch.no_grad()
def run_episode(
    model: NitroGen,
    game_name: str,
    seed: int,
    replan_every: int = 4,
    sample_steps: int = 10,
) -> bool:
    """Roll one episode with action chunking. Returns whether the task succeeded."""
    env = make_game(game_name)
    frame = env.reset(seed=seed)
    done = False
    info = {"success": False}
    chunk = None
    idx = 0
    while not done:
        if chunk is None or idx >= replan_every:
            raw = model.act(frame, num_steps=sample_steps)  # (T, ACTION_DIM) raw
            chunk = raw
            idx = 0
        action = chunk[idx]
        idx += 1
        frame, done, info = env.step(action)
        if info["success"]:
            # Continuous games latch success but keep running; stop as soon as the
            # task is first accomplished (task-completion is the metric we want).
            return True
    return bool(info["success"])


@torch.no_grad()
def evaluate_game(
    model: NitroGen,
    game_name: str,
    episodes: int = 20,
    replan_every: int = 4,
    sample_steps: int = 10,
    seed_offset: int = 10_000,
) -> float:
    """Success rate over ``episodes`` fresh seeds for a single game."""
    model.eval()
    wins = 0
    for e in range(episodes):
        wins += run_episode(
            model, game_name, seed=seed_offset + e,
            replan_every=replan_every, sample_steps=sample_steps,
        )
    return wins / episodes


def evaluate_suite(
    model: NitroGen,
    games: List[str],
    episodes: int = 20,
    **kwargs,
) -> Dict[str, float]:
    """Success rate per game plus the mean — this dict *is* the benchmark score."""
    scores = {g: evaluate_game(model, g, episodes=episodes, **kwargs) for g in games}
    scores["mean"] = float(np.mean([scores[g] for g in games]))
    return scores


def transfer_report(
    model: NitroGen,
    episodes: int = 20,
    train_games: List[str] | None = None,
    heldout_games: List[str] | None = None,
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    """Compare success on *seen* training games vs *held-out* unseen games.

    This is the zero-shot view. At the paper's scale the model retains non-trivial
    success on unseen games; at this toy scale a policy trained on just two games
    overfits their exact appearance, so zero-shot success on a genuinely novel game
    is typically near zero — the interesting transfer signal here is *few-shot*
    (see :func:`few_shot_transfer`).
    """
    train_games = train_games or TRAIN_GAMES
    heldout_games = heldout_games or HELDOUT_GAMES
    return {
        "seen": evaluate_suite(model, train_games, episodes=episodes, **kwargs),
        "unseen": evaluate_suite(model, heldout_games, episodes=episodes, **kwargs),
    }


def few_shot_transfer(
    pretrained: NitroGen,
    game: str = "chaser",
    n_episodes: int = 12,
    finetune_steps: int = 400,
    eval_episodes: int = 25,
    seed: int = 0,
) -> Dict[str, float]:
    """The paper's real transfer claim: **low-data** adaptation to a new game.

    Fine-tunes the ``pretrained`` policy on a *small* number of demonstrations from
    an unseen ``game`` and compares it to a model trained from scratch on the exact
    same data. The pretrained features should reach much higher success from the
    same handful of episodes — mirroring NitroGen's "up to 52% relative improvement
    on low-data tasks".

    Returns success rates for ``from_scratch`` and ``pretrained`` plus the absolute
    and relative improvement. ``relative_improvement`` is ``None`` when the
    from-scratch baseline is ~0% (the ratio is undefined — a common outcome at this
    toy scale, since a fresh model often learns nothing from a dozen episodes); use
    ``absolute_improvement`` in that case.
    """
    # Imported here to avoid a circular import (trainer imports the model, etc.).
    from nitrogen.data.synthetic import collect_dataset
    from nitrogen.training.config import TrainConfig
    from nitrogen.training.trainer import train

    data = collect_dataset([game], episodes_per_game=n_episodes, seed=seed)
    cfg = TrainConfig(games=[game], steps=finetune_steps, batch_size=48,
                      log_every=finetune_steps, seed=seed)

    _, scratch_ema = train(cfg, episodes=data)
    _, ft_ema = train(cfg, episodes=data, init_model=pretrained)

    scratch = evaluate_game(scratch_ema.shadow, game, episodes=eval_episodes)
    finetuned = evaluate_game(ft_ema.shadow, game, episodes=eval_episodes)
    # Relative improvement is only meaningful when the baseline is non-trivial;
    # otherwise the ratio blows up (dividing by ~0), so report it as undefined.
    rel = (finetuned - scratch) / scratch if scratch > 1e-6 else None
    return {
        "game": game,
        "n_episodes": n_episodes,
        "from_scratch": scratch,
        "pretrained": finetuned,
        "absolute_improvement": finetuned - scratch,
        "relative_improvement": rel,
    }
