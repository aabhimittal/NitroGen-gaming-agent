"""Standardized gamepad action space.

One of the three pillars of NitroGen is a *single, standardized action interface*
shared across every game. Real games expose wildly different control schemes
(keyboard+mouse, touch, bespoke bindings). NitroGen sidesteps this by mapping
everything onto a **virtual Xbox-style gamepad**: two analog sticks, two analog
triggers, and a set of digital buttons. Because every game speaks the *same*
action language, a single policy can be trained across 1,000+ titles and can
transfer to unseen ones — you never have to re-learn "what a controller is".

For the generative action head we need a *continuous* target vector, so we pack
the heterogeneous controls into one real-valued vector in ``[-1, 1]^ACTION_DIM``:

    * analog sticks  (already in [-1, 1])            -> passed through
    * analog triggers (in [0, 1])                    -> rescaled to [-1, 1]
    * digital buttons (in {0, 1})                    -> mapped to {-1, +1}

Flow matching then models this whole vector as continuous noise->data transport.
At decode time we threshold the button channels back to {0, 1}. This "buttons as
soft continuous values that get rounded" trick is what lets a *diffusion-style*
head emit discrete presses without a separate categorical head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Channel layout
# ---------------------------------------------------------------------------
# Indices into the packed action vector. Keeping them explicit (rather than
# magic numbers scattered around) makes the encode/decode logic auditable.

STICK_NAMES: List[str] = ["left_x", "left_y", "right_x", "right_y"]
TRIGGER_NAMES: List[str] = ["left_trigger", "right_trigger"]
BUTTON_NAMES: List[str] = [
    "A", "B", "X", "Y",           # face buttons
    "LB", "RB",                    # shoulder bumpers
    "dpad_up", "dpad_down", "dpad_left", "dpad_right",
    "start", "back",
]

N_STICK = len(STICK_NAMES)        # 4
N_TRIGGER = len(TRIGGER_NAMES)    # 2
N_BUTTON = len(BUTTON_NAMES)      # 12
ACTION_DIM = N_STICK + N_TRIGGER + N_BUTTON  # 18

# Slices into the packed vector.
STICK_SLICE = slice(0, N_STICK)
TRIGGER_SLICE = slice(N_STICK, N_STICK + N_TRIGGER)
BUTTON_SLICE = slice(N_STICK + N_TRIGGER, ACTION_DIM)


@dataclass
class GamepadAction:
    """A single, human-readable gamepad state.

    Sticks are in ``[-1, 1]`` (0 = centered), triggers in ``[0, 1]``
    (0 = released), and buttons are booleans. This is the *ergonomic* view used
    by environments and the overlay-extraction pipeline; the model works with
    the packed :func:`encode_action` view instead.
    """

    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    buttons: dict = field(default_factory=dict)

    def button(self, name: str) -> bool:
        return bool(self.buttons.get(name, False))

    def to_vector(self) -> np.ndarray:
        """Return the *raw* (un-normalized) vector: sticks[-1,1], triggers[0,1], buttons{0,1}."""
        vec = np.zeros(ACTION_DIM, dtype=np.float32)
        vec[STICK_SLICE] = [self.left_x, self.left_y, self.right_x, self.right_y]
        vec[TRIGGER_SLICE] = [self.left_trigger, self.right_trigger]
        for i, name in enumerate(BUTTON_NAMES):
            vec[N_STICK + N_TRIGGER + i] = 1.0 if self.button(name) else 0.0
        return vec

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "GamepadAction":
        """Inverse of :meth:`to_vector` (expects raw, not normalized, values)."""
        vec = np.asarray(vec, dtype=np.float32)
        buttons = {
            name: bool(vec[N_STICK + N_TRIGGER + i] > 0.5)
            for i, name in enumerate(BUTTON_NAMES)
        }
        return cls(
            left_x=float(vec[0]), left_y=float(vec[1]),
            right_x=float(vec[2]), right_y=float(vec[3]),
            left_trigger=float(vec[4]), right_trigger=float(vec[5]),
            buttons=buttons,
        )


# ---------------------------------------------------------------------------
# Packing to / from the continuous [-1, 1] space the model actually learns
# ---------------------------------------------------------------------------

def encode_action(raw: np.ndarray) -> np.ndarray:
    """Map a *raw* action vector into the normalized ``[-1, 1]`` model space.

    Sticks are already in ``[-1, 1]``. Triggers ``[0, 1] -> [-1, 1]``. Buttons
    ``{0, 1} -> {-1, +1}``. The result is what flow matching treats as "data".
    Works on a single vector ``(ACTION_DIM,)`` or a batch ``(..., ACTION_DIM)``.
    """
    raw = np.asarray(raw, dtype=np.float32)
    out = raw.copy()
    out[..., TRIGGER_SLICE] = raw[..., TRIGGER_SLICE] * 2.0 - 1.0
    out[..., BUTTON_SLICE] = raw[..., BUTTON_SLICE] * 2.0 - 1.0
    return np.clip(out, -1.0, 1.0)


def decode_action(norm: np.ndarray) -> np.ndarray:
    """Inverse of :func:`encode_action`: map model space back to raw controls.

    Triggers are clamped to ``[0, 1]``; buttons are thresholded at 0 (the image
    of 0.5 under encoding) to recover crisp ``{0, 1}`` presses.
    """
    norm = np.asarray(norm, dtype=np.float32)
    out = norm.copy()
    out[..., STICK_SLICE] = np.clip(norm[..., STICK_SLICE], -1.0, 1.0)
    out[..., TRIGGER_SLICE] = np.clip((norm[..., TRIGGER_SLICE] + 1.0) * 0.5, 0.0, 1.0)
    out[..., BUTTON_SLICE] = (norm[..., BUTTON_SLICE] > 0.0).astype(np.float32)
    return out


def null_action() -> np.ndarray:
    """The normalized encoding of "do nothing" (centered sticks, released everything)."""
    return encode_action(GamepadAction().to_vector())
