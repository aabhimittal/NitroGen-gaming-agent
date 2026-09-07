"""Synthetic gameplay corpus: scripted experts rolled out inside the toy games.

The real NitroGen corpus is ~40,000 hours of internet gameplay footage whose
action labels are read off on-screen controller overlays (see
:mod:`nitrogen.data.overlay_extraction`). We obviously can't ship that, so this
module produces an *analogous* corpus: roll a scripted expert inside each toy
game and record ``(frame, action)`` pairs.

A little action noise is injected on purpose. A perfectly deterministic script
only ever visits the optimal trajectory, so a policy cloned from it has never
seen a slightly-off state and cannot recover from one. Noise widens the state
distribution — the same reason messy human demonstrations clone more robustly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from nitrogen.action_space import ACTION_DIM
from nitrogen.envs.toy_game import make_game


@dataclass
class Episode:
    """One rollout: aligned frames and the *raw* actions taken from them.

    ``actions[i]`` is the action the expert took **after observing**
    ``frames[i]``, which is exactly the alignment behavior cloning needs.
    """

    game: str
    frames: np.ndarray      # (T, H, W, 3) uint8
    actions: np.ndarray     # (T, ACTION_DIM) float32, raw (un-normalized) space
    success: bool = False

    def __len__(self) -> int:
        return int(self.frames.shape[0])


def collect_episode(game: str, seed: int = 0, noise: float = 0.0) -> Episode:
    """Roll the scripted expert for one episode of ``game``."""
    env = make_game(game)
    frame = env.reset(seed=seed)
    frames: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    done, info = False, {"success": False}
    while not done:
        action = env.expert_action(noise=noise)
        frames.append(frame)
        actions.append(np.asarray(action, dtype=np.float32))
        frame, done, info = env.step(action)
    return Episode(
        game=game,
        frames=np.stack(frames).astype(np.uint8),
        actions=np.stack(actions).astype(np.float32),
        success=bool(info.get("success", False)),
    )


def collect_dataset(
    games: Sequence[str],
    episodes_per_game: int = 80,
    noise: float = 0.15,
    seed: int = 0,
) -> List[Episode]:
    """Roll ``episodes_per_game`` episodes for each game with distinct seeds.

    Seeds are offset per game so two games never share a rollout seed, which
    would otherwise correlate their object spawn positions.
    """
    if episodes_per_game <= 0:
        return []
    episodes: List[Episode] = []
    for g_idx, game in enumerate(games):
        base = seed + 1_000 * (g_idx + 1)
        for e in range(episodes_per_game):
            episodes.append(collect_episode(game, seed=base + e, noise=noise))
    return episodes


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
# Episodes have different lengths, so we store one flat frame/action array plus
# the episode boundaries. That keeps the archive a single compressed .npz
# instead of a directory of ragged per-episode arrays.

def save_episodes(episodes: Sequence[Episode], path: str) -> None:
    """Write episodes to a compressed ``.npz`` archive."""
    if not episodes:
        raise ValueError("refusing to save an empty episode list")
    lengths = np.array([len(e) for e in episodes], dtype=np.int64)
    np.savez_compressed(
        path,
        frames=np.concatenate([e.frames for e in episodes], axis=0),
        actions=np.concatenate([e.actions for e in episodes], axis=0),
        lengths=lengths,
        games=np.array([e.game for e in episodes]),
        success=np.array([e.success for e in episodes], dtype=bool),
    )


def load_episodes(path: str) -> List[Episode]:
    """Inverse of :func:`save_episodes`."""
    with np.load(path, allow_pickle=False) as z:
        frames, actions = z["frames"], z["actions"]
        lengths, games, success = z["lengths"], z["games"], z["success"]
    out, start = [], 0
    for i, n in enumerate(lengths):
        n = int(n)
        out.append(
            Episode(
                game=str(games[i]),
                frames=frames[start:start + n],
                actions=actions[start:start + n],
                success=bool(success[i]),
            )
        )
        start += n
    return out


def dataset_stats(episodes: Sequence[Episode]) -> Dict[str, float]:
    """Quick corpus summary — useful to sanity-check a collection run."""
    if not episodes:
        return {"episodes": 0, "frames": 0, "success_rate": 0.0, "mean_length": 0.0}
    lengths = [len(e) for e in episodes]
    return {
        "episodes": len(episodes),
        "frames": int(sum(lengths)),
        "success_rate": float(np.mean([e.success for e in episodes])),
        "mean_length": float(np.mean(lengths)),
        "action_dim": ACTION_DIM,
    }
