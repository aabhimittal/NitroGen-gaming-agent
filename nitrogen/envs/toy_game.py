"""Toy games — a miniature multi-game benchmark.

NitroGen's second pillar is a **multi-game benchmark** that measures *cross-game
generalization*. We obviously can't ship 1,000 real titles, so this module
provides a small family of visually distinct 2D games that nonetheless exercise
the same generalist skill: *look at the frame, move the left stick toward what
matters*. Because every game is controlled through the shared gamepad interface
(:mod:`nitrogen.action_space`), one policy can be trained across several games
and evaluated on a held-out one — exactly the transfer setup from the paper.

Each game implements the same small interface:

    frame = env.reset(seed)                 # (256, 256, 3) uint8
    frame, done, info = env.step(action)    # action: raw gamepad vector
    expert = env.expert_action(noise=...)   # raw gamepad vector (for BC data)

``info["success"]`` reports task completion. Games are intentionally learnable
from a single frame, matching NitroGen's frame-only (history-free) conditioning.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from nitrogen.action_space import ACTION_DIM, GamepadAction
from nitrogen.envs.rendering import Canvas


class ToyGame:
    """Base class. Subclasses set colours, dynamics, expert, and success."""

    name: str = "base"
    max_steps: int = 60
    size: int = 256

    def __init__(self, size: int = 256):
        self.size = size
        self.rng = np.random.default_rng(0)
        self.t = 0

    # -- API -------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self._reset_state()
        return self.render()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, bool, Dict]:
        self.t += 1
        self._apply(action)
        success = self._success()
        done = success or self.t >= self.max_steps
        return self.render(), done, {"success": success, "t": self.t}

    # -- to be implemented by subclasses --------------------------------
    def _reset_state(self):
        raise NotImplementedError

    def _apply(self, action: np.ndarray):
        raise NotImplementedError

    def _success(self) -> bool:
        raise NotImplementedError

    def render(self) -> np.ndarray:
        raise NotImplementedError

    def expert_action(self, noise: float = 0.0) -> np.ndarray:
        raise NotImplementedError

    # -- shared helpers --------------------------------------------------
    @staticmethod
    def _stick_from_direction(dx: float, dy: float, noise: float, rng) -> np.ndarray:
        """Build a raw gamepad vector that pushes the left stick along (dx, dy)."""
        norm = float(np.hypot(dx, dy)) + 1e-8
        lx, ly = dx / norm, dy / norm
        if noise > 0:
            lx += rng.normal(0, noise)
            ly += rng.normal(0, noise)
        a = GamepadAction(left_x=float(np.clip(lx, -1, 1)), left_y=float(np.clip(ly, -1, 1)))
        return a.to_vector()


# ---------------------------------------------------------------------------
# Game 1: Reacher — analog of a 3D "go to the objective" task
# ---------------------------------------------------------------------------
class ReacherGame(ToyGame):
    """Drive the blue avatar onto the gold target using the left stick."""

    name = "reacher"
    max_steps = 60

    def _reset_state(self):
        self.agent = self.rng.uniform(0.15, 0.85, size=2).astype(np.float32)
        self.target = self.rng.uniform(0.15, 0.85, size=2).astype(np.float32)
        # Ensure they don't start on top of each other.
        while np.linalg.norm(self.agent - self.target) < 0.3:
            self.target = self.rng.uniform(0.15, 0.85, size=2).astype(np.float32)
        self.speed = 0.04

    def _apply(self, action: np.ndarray):
        lx, ly = float(action[0]), float(action[1])
        self.agent = np.clip(self.agent + self.speed * np.array([lx, ly]), 0.05, 0.95)

    def _success(self) -> bool:
        return bool(np.linalg.norm(self.agent - self.target) < 0.06)

    def render(self) -> np.ndarray:
        c = Canvas(self.size, background=(20, 24, 40))
        c.fill_gradient((20, 24, 40), (40, 30, 60))
        c.circle(self.target[0], self.target[1], 0.05, (240, 200, 60))   # gold target
        c.circle(self.agent[0], self.agent[1], 0.035, (70, 150, 240))    # blue avatar
        return c.to_array()

    def expert_action(self, noise: float = 0.0) -> np.ndarray:
        d = self.target - self.agent
        return self._stick_from_direction(d[0], d[1], noise, self.rng)


# ---------------------------------------------------------------------------
# Game 2: Dodger — analog of a 2D platformer reflex task
# ---------------------------------------------------------------------------
class DodgerGame(ToyGame):
    """Slide the paddle at the bottom to avoid the falling red hazard."""

    name = "dodger"
    max_steps = 60

    def _reset_state(self):
        self.paddle_x = 0.5
        self.hazard = np.array([self.rng.uniform(0.2, 0.8), 0.0], dtype=np.float32)
        self.hazard_speed = 0.05
        self.paddle_speed = 0.06
        self.alive = True

    def _apply(self, action: np.ndarray):
        lx = float(action[0])
        self.paddle_x = float(np.clip(self.paddle_x + self.paddle_speed * lx, 0.08, 0.92))
        self.hazard[1] += self.hazard_speed
        if self.hazard[1] >= 0.88:  # reached paddle row
            if abs(self.hazard[0] - self.paddle_x) < 0.1:
                self.alive = False
            # respawn hazard at a new column
            self.hazard = np.array([self.rng.uniform(0.15, 0.85), 0.0], dtype=np.float32)

    def _success(self) -> bool:
        # "Success" = survive the full episode.
        return self.alive and self.t >= self.max_steps - 1

    def step(self, action):
        frame, done, info = super().step(action)
        if not self.alive:
            done = True
            info["success"] = False
        return frame, done, info

    def render(self) -> np.ndarray:
        c = Canvas(self.size, background=(12, 20, 20))
        c.fill_gradient((12, 24, 24), (10, 12, 18))
        c.rect(self.paddle_x - 0.1, 0.9, self.paddle_x + 0.1, 0.95, (80, 220, 140))  # paddle
        c.circle(self.hazard[0], self.hazard[1], 0.04, (240, 70, 70))                # hazard
        return c.to_array()

    def expert_action(self, noise: float = 0.0) -> np.ndarray:
        # Move away from the hazard's horizontal position; idle if it's far above.
        if self.hazard[1] < 0.4:
            dx = 0.5 - self.paddle_x  # drift to center when safe
        else:
            dx = self.paddle_x - self.hazard[0]  # flee horizontally
            dx = np.sign(dx) if abs(dx) > 1e-3 else 1.0
        return self._stick_from_direction(dx, 0.0, noise, self.rng)


# ---------------------------------------------------------------------------
# Game 3: Chaser — held-out "unseen" game for transfer experiments
# ---------------------------------------------------------------------------
class ChaserGame(ToyGame):
    """Catch the fleeing green orb. Same 'move toward salient object' skill as
    Reacher but different colours/dynamics — a good zero-shot transfer target."""

    name = "chaser"
    max_steps = 80

    def _reset_state(self):
        self.agent = np.array([0.5, 0.5], dtype=np.float32)
        self.prey = self.rng.uniform(0.2, 0.8, size=2).astype(np.float32)
        self.speed = 0.045
        self.prey_speed = 0.025

    def _apply(self, action: np.ndarray):
        lx, ly = float(action[0]), float(action[1])
        self.agent = np.clip(self.agent + self.speed * np.array([lx, ly]), 0.05, 0.95)
        # Prey drifts away from the agent (slowly), making it a mild pursuit.
        flee = self.prey - self.agent
        n = np.linalg.norm(flee) + 1e-6
        self.prey = np.clip(self.prey + self.prey_speed * flee / n, 0.05, 0.95)

    def _success(self) -> bool:
        return bool(np.linalg.norm(self.agent - self.prey) < 0.06)

    def render(self) -> np.ndarray:
        c = Canvas(self.size, background=(30, 18, 18))
        c.fill_gradient((34, 18, 22), (18, 14, 26))
        c.circle(self.prey[0], self.prey[1], 0.04, (90, 230, 120))    # green prey
        c.circle(self.agent[0], self.agent[1], 0.04, (230, 120, 200)) # pink chaser
        return c.to_array()

    def expert_action(self, noise: float = 0.0) -> np.ndarray:
        d = self.prey - self.agent
        return self._stick_from_direction(d[0], d[1], noise, self.rng)


# ---------------------------------------------------------------------------
# Registry — the "benchmark suite"
# ---------------------------------------------------------------------------
GAME_REGISTRY = {
    ReacherGame.name: ReacherGame,
    DodgerGame.name: DodgerGame,
    ChaserGame.name: ChaserGame,
}

# Convention used by the transfer experiment: train on these, hold out the rest.
TRAIN_GAMES = ["reacher", "dodger"]
HELDOUT_GAMES = ["chaser"]


def make_game(name: str, size: int = 256) -> ToyGame:
    if name not in GAME_REGISTRY:
        raise KeyError(f"Unknown game '{name}'. Available: {list(GAME_REGISTRY)}")
    return GAME_REGISTRY[name](size=size)
