"""Toy games — a miniature multi-game benchmark.

NitroGen's second pillar is a **multi-game benchmark** that measures *cross-game
generalization*. We obviously can't ship 1,000 real titles, so this module
provides a small family of visually distinct games that all exercise the same
generalist skill: *look at the frame and move the left stick relative to the one
salient object on screen* — toward it, or away from it, depending on the game.

Agent-centric framing
---------------------
Every game is rendered **agent-centric**: the controllable avatar is a fixed
reticle at the center of the screen, and the world is drawn relative to it (the
way most first- and third-person games render around the player). Pushing the
stick moves the avatar, which slides the salient object across the view. This is
deliberate: it turns each task into "find the one bright object and push toward
/ away from it", a single-object visuomotor skill that a small vision-action
model can actually *learn* from pixels — while keeping every NitroGen mechanism
intact (pixels-in, standardized-gamepad-out, one frame conditioning a chunk).

Each game implements the same small interface:

    frame = env.reset(seed)                 # (256, 256, 3) uint8
    frame, done, info = env.step(action)    # action: raw gamepad vector
    expert = env.expert_action(noise=...)   # raw gamepad vector (for BC data)

``info["success"]`` reports task completion. Games are learnable from a single
frame, matching NitroGen's frame-only (history-free) conditioning.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from nitrogen.action_space import ACTION_DIM, GamepadAction
from nitrogen.envs.rendering import Canvas

CENTER = np.array([0.5, 0.5], dtype=np.float32)
REL_BOUND = 0.44  # keep the salient object on-screen


class ToyGame:
    """Base class. Subclasses set colours, dynamics, expert, and success."""

    name: str = "base"
    max_steps: int = 60
    size: int = 256
    # Continuous games keep running after a "success" (the objective respawns), so
    # episodes are always full-length. This matters for training: with short
    # episodes, most action chunks would be dominated by end-of-episode padding
    # and the policy would just learn to idle. See docs/07_training.md.
    terminate_on_success: bool = True

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
        done = (self.terminate_on_success and success) or self.t >= self.max_steps
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


class AgentCentricGame(ToyGame):
    """Shared scaffolding: a fixed center reticle + one salient object at ``center + rel``.

    Subclasses control the object's colour, how it drifts each step, and whether
    reaching it is a win (Reacher, Chaser) or a loss (Avoider).
    """

    speed = 0.05                       # how fast the stick moves the avatar
    hit_radius = 0.06                  # contact distance between avatar and object
    spawn_dist = (0.25, 0.40)          # initial |rel| range
    obj_radius = 0.05
    obj_color = (240, 240, 240)
    reticle_color = (225, 228, 240)
    bg_top = (20, 24, 40)
    bg_bottom = (40, 30, 60)
    approach = "toward"                # expert pushes "toward" or "away" from the object

    def _spawn(self):
        ang = self.rng.uniform(0, 2 * np.pi)
        dist = self.rng.uniform(*self.spawn_dist)
        self.rel = np.array([np.cos(ang) * dist, np.sin(ang) * dist], dtype=np.float32)

    def _reset_state(self):
        self.reached = False
        self._spawn()

    def _object_drift(self):
        """Optional per-step motion of the object, by a fixed rule (override).

        Crucially the motion is a deterministic function of the current state
        (never a random jump), so the whole action chunk stays predictable from
        the single conditioning frame — see docs/07_training.md on why that is
        essential for behavior cloning with action chunks.
        """

    def _apply(self, action: np.ndarray):
        # Move the avatar: the object slides opposite to the stick push.
        self.rel = self.rel - self.speed * np.array([action[0], action[1]], dtype=np.float32)
        self._object_drift()
        self.rel = np.clip(self.rel, -REL_BOUND, REL_BOUND)
        if self._dist() < self.hit_radius:
            self.reached = True

    def _dist(self) -> float:
        return float(np.linalg.norm(self.rel))

    def render(self) -> np.ndarray:
        c = Canvas(self.size, background=self.bg_top)
        c.fill_gradient(self.bg_top, self.bg_bottom)
        obj = CENTER + self.rel
        c.circle(float(obj[0]), float(obj[1]), self.obj_radius, self.obj_color)
        c.crosshair(0.5, 0.5, 0.045, self.reticle_color)
        return c.to_array()

    def expert_action(self, noise: float = 0.0) -> np.ndarray:
        # Hold still once we're on the objective (a predictable "do nothing").
        if self.approach == "toward" and self._dist() < self.hit_radius:
            return GamepadAction().to_vector()
        d = self.rel if self.approach == "toward" else -self.rel
        return self._stick_from_direction(float(d[0]), float(d[1]), noise, self.rng)


# ---------------------------------------------------------------------------
# Game 1: Reacher — home the reticle onto a stationary beacon
# ---------------------------------------------------------------------------
class ReacherGame(AgentCentricGame):
    """Home the reticle onto the stationary gold beacon (push the stick toward it).

    The beacon does not move and does not respawn, so from any frame the entire
    future action chunk — steer straight in, then hold on arrival — is a
    deterministic function of the beacon's position on screen."""

    name = "reacher"
    max_steps = 30
    # End the episode on contact: the recorded data is then *pure homing* ("steer
    # toward the beacon"), with no post-arrival "hold" frames. Those holds, if
    # kept, dominate the dataset and collapse the policy into doing nothing.
    terminate_on_success = True
    speed = 0.03
    spawn_dist = (0.30, 0.42)
    hit_radius = 0.06
    obj_color = (240, 200, 60)
    bg_top = (20, 24, 40)
    bg_bottom = (44, 32, 64)
    approach = "toward"

    def _success(self) -> bool:
        return self.reached


# ---------------------------------------------------------------------------
# Game 2: Avoider — keep an incoming asteroid away (push the stick AWAY from it)
# ---------------------------------------------------------------------------
class AvoiderGame(AgentCentricGame):
    """Strafe so the single red asteroid never reaches the reticle.

    The asteroid drifts toward the center by a fixed rule; the player pushes it
    back out. No respawn — one persistent hazard — so the chunk stays predictable.
    """

    name = "avoider"
    max_steps = 34
    speed = 0.05
    hit_radius = 0.07
    spawn_dist = (0.34, 0.42)
    obj_radius = 0.045
    obj_color = (240, 70, 70)
    bg_top = (12, 20, 20)
    bg_bottom = (10, 12, 18)
    approach = "away"
    approach_speed = 0.028   # how fast the asteroid closes in on the reticle

    def _reset_state(self):
        super()._reset_state()
        self.alive = True

    def _object_drift(self):
        d = self._dist()
        if d > 1e-6:
            self.rel = self.rel - self.approach_speed * self.rel / d

    def _apply(self, action: np.ndarray):
        super()._apply(action)
        if self._dist() < self.hit_radius:
            self.alive = False

    def _success(self) -> bool:
        return self.alive and self.t >= self.max_steps - 1

    def step(self, action):
        frame, done, info = super().step(action)
        if not self.alive:
            done = True
            info["success"] = False
        return frame, done, info


# ---------------------------------------------------------------------------
# Game 3: Chaser — held-out "unseen" game for transfer experiments
# ---------------------------------------------------------------------------
class ChaserGame(AgentCentricGame):
    """Catch the fleeing amber orb (push the stick toward it as it runs away).

    This is the **held-out** game used to probe cross-game *transfer*. It is never
    trained on. It shares Reacher's skill — *approach the warm salient object* —
    but presents it in a new game: the target now flees, sits on a different
    background, and is a slightly different shade. A policy that merely memorized
    Reacher fails here; one that learned the transferable "steer toward the warm
    object" behavior still catches the orb zero-shot. The orb flees by a fixed
    rule (away from the avatar), so the action chunk stays predictable."""

    name = "chaser"
    max_steps = 45
    terminate_on_success = True    # end on catch → pure-pursuit data, no idling
    speed = 0.04
    spawn_dist = (0.28, 0.40)
    hit_radius = 0.065
    obj_color = (250, 170, 50)     # warm amber — shares Reacher's "approach" cue
    obj_radius = 0.05
    bg_top = (26, 22, 34)
    bg_bottom = (16, 20, 30)
    approach = "toward"
    flee_speed = 0.015      # the orb drifts away from the avatar each step

    def _object_drift(self):
        d = self._dist()
        if d > 1e-6:
            self.rel = self.rel + self.flee_speed * self.rel / d  # move orb outward

    def _success(self) -> bool:
        return self.reached


# ---------------------------------------------------------------------------
# Registry — the "benchmark suite"
# ---------------------------------------------------------------------------
GAME_REGISTRY = {
    ReacherGame.name: ReacherGame,
    AvoiderGame.name: AvoiderGame,
    ChaserGame.name: ChaserGame,
}

# Convention used by the transfer experiment: train on these, hold out the rest.
TRAIN_GAMES = ["reacher", "avoider"]
HELDOUT_GAMES = ["chaser"]


def make_game(name: str, size: int = 256) -> ToyGame:
    if name not in GAME_REGISTRY:
        raise KeyError(f"Unknown game '{name}'. Available: {list(GAME_REGISTRY)}")
    return GAME_REGISTRY[name](size=size)
