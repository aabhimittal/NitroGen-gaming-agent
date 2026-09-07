"""Making a generated action legal for real hardware, and safe for a real game.

The action head is a *generative* model. It samples from a learned distribution
over ``[-1, 1]^18`` and nothing in that formulation knows about the physical
gamepad on the other end. Straight from the sampler you get, routinely:

    * a stick at ``(0.98, 0.97)`` — magnitude 1.38, outside the circular gate a
      real thumbstick is physically confined to. A driver clamps it silently, so
      the game sees a *different* direction than the policy intended.
    * tiny non-zero drift on an idle stick, which a real controller's deadzone
      would have swallowed and which reads as constant slow walking.
    * ``dpad_up`` and ``dpad_down`` both "pressed" — impossible on a rocker
      d-pad; the driver's tie-break is undefined.
    * a button flickering on for a single tick because the sampled channel sat
      near the 0.5 decision boundary. On real input stacks that is either a
      dropped press or a machine-gun rebind.
    * a full-deflection reversal in one tick, which is not a motion any human
      hand produces and which some anti-cheat heuristics flag as automation.
    * ``start`` or ``back``, which pauses or exits the game — an agent that
      presses those in an unattended benchmark run just deleted its own episode.

:class:`ActionGuard` is the last stage before the virtual pad: it turns a
sampled vector into one a human hand could plausibly have produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from nitrogen.action_space import (
    ACTION_DIM,
    BUTTON_NAMES,
    BUTTON_SLICE,
    GamepadAction,
    N_STICK,
    N_TRIGGER,
    STICK_SLICE,
    TRIGGER_SLICE,
)

# Pairs a physical rocker d-pad cannot report simultaneously.
EXCLUSIVE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("dpad_up", "dpad_down"),
    ("dpad_left", "dpad_right"),
)


@dataclass
class GamepadLimits:
    """Hardware and policy limits applied to every emitted action.

    Defaults are deliberately conservative — they are what you'd want on an
    unattended run against a real title. Set ``blocked_buttons=()`` if the agent
    genuinely needs the menu buttons.
    """

    stick_deadzone: float = 0.08        # radial, matching a real stick's gate
    trigger_deadzone: float = 0.05
    unit_circle: bool = True            # clamp |stick| <= 1
    stick_max_rate: float = 0.6         # max change per tick, per axis
    trigger_max_rate: float = 0.6
    press_threshold: float = 0.6        # continuous -> pressed (rising edge)
    release_threshold: float = 0.4      # continuous -> released (falling edge)
    min_hold_ticks: int = 2             # debounce: a press lasts at least this
    blocked_buttons: Tuple[str, ...] = ("start", "back")
    resolve_exclusive: bool = True

    def __post_init__(self):
        if not 0.0 <= self.release_threshold <= self.press_threshold <= 1.0:
            raise ValueError(
                "need 0 <= release_threshold <= press_threshold <= 1, got "
                f"{self.release_threshold} / {self.press_threshold}"
            )
        if self.min_hold_ticks < 1:
            raise ValueError("min_hold_ticks must be >= 1")
        unknown = set(self.blocked_buttons) - set(BUTTON_NAMES)
        if unknown:
            raise ValueError(f"unknown blocked buttons: {sorted(unknown)}")


class ActionGuard:
    """Stateful sanitizer: raw sampled action in, hardware-legal action out.

    Stateful because three of the guarantees are inherently temporal — slew
    limiting, press/release hysteresis, and minimum hold time all need the
    previous emitted action. Call :meth:`reset` between episodes so one
    episode's final state can't bleed into the next one's first tick.
    """

    def __init__(self, limits: Optional[GamepadLimits] = None):
        self.limits = limits or GamepadLimits()
        self._blocked_idx = [BUTTON_NAMES.index(b) for b in self.limits.blocked_buttons]
        self.reset()

    def reset(self) -> None:
        self._prev_analog = np.zeros(N_STICK + N_TRIGGER, dtype=np.float32)
        self._held = np.zeros(len(BUTTON_NAMES), dtype=bool)
        self._hold_ticks = np.zeros(len(BUTTON_NAMES), dtype=np.int32)
        self.stats: Dict[str, int] = {
            "calls": 0, "nonfinite": 0, "deadzoned": 0, "circle_clamped": 0,
            "rate_limited": 0, "exclusive_resolved": 0, "buttons_blocked": 0,
            "debounced": 0,
        }

    # -- the pipeline ----------------------------------------------------
    def __call__(self, action: Sequence[float]) -> np.ndarray:
        """Sanitize one *raw* action vector. Returns a new ``(ACTION_DIM,)`` array.

        Input and output are both in raw gamepad space (sticks ``[-1, 1]``,
        triggers ``[0, 1]``, buttons ``{0, 1}``) — the space
        :meth:`nitrogen.NitroGen.act` emits and an environment consumes.
        """
        vec = np.asarray(action, dtype=np.float32).reshape(-1)
        if vec.shape[0] != ACTION_DIM:
            raise ValueError(f"expected {ACTION_DIM} channels, got {vec.shape[0]}")
        self.stats["calls"] += 1

        # A NaN reaching a driver is undefined behaviour; neutral is the only
        # defensible reading of "the model produced nothing".
        bad = ~np.isfinite(vec)
        if bad.any():
            self.stats["nonfinite"] += int(bad.sum())
            vec = np.where(bad, 0.0, vec)

        analog = self._analog(vec)
        buttons = self._buttons(vec)

        out = np.zeros(ACTION_DIM, dtype=np.float32)
        out[:N_STICK + N_TRIGGER] = analog
        out[BUTTON_SLICE] = buttons.astype(np.float32)
        self._prev_analog = analog
        return out

    # -- analog channels -------------------------------------------------
    def _analog(self, vec: np.ndarray) -> np.ndarray:
        lim = self.limits
        sticks = np.clip(vec[STICK_SLICE].copy(), -1.0, 1.0)
        triggers = np.clip(vec[TRIGGER_SLICE].copy(), 0.0, 1.0)

        # Radial deadzone per stick: kill the whole vector, not each axis
        # separately. Axis-wise deadzones are the classic bug that turns a slow
        # diagonal into a pure-cardinal jerk.
        for s in (slice(0, 2), slice(2, 4)):
            mag = float(np.hypot(sticks[s][0], sticks[s][1]))
            if mag < lim.stick_deadzone:
                if mag > 0:
                    self.stats["deadzoned"] += 1
                sticks[s] = 0.0
            elif lim.unit_circle and mag > 1.0:
                self.stats["circle_clamped"] += 1
                sticks[s] = sticks[s] / mag

        triggers = np.where(triggers < lim.trigger_deadzone, 0.0, triggers)

        analog = np.concatenate([sticks, triggers]).astype(np.float32)
        rates = np.concatenate([
            np.full(N_STICK, lim.stick_max_rate, dtype=np.float32),
            np.full(N_TRIGGER, lim.trigger_max_rate, dtype=np.float32),
        ])
        delta = analog - self._prev_analog
        over = np.abs(delta) > rates
        if over.any():
            self.stats["rate_limited"] += int(over.sum())
            analog = self._prev_analog + np.clip(delta, -rates, rates)
        return analog.astype(np.float32)

    # -- digital channels ------------------------------------------------
    def _buttons(self, vec: np.ndarray) -> np.ndarray:
        lim = self.limits
        # Buttons arrive as {0, 1} from decode_action, but a caller may hand us
        # the pre-threshold continuous channel; treating both as a level in
        # [0, 1] lets the hysteresis do real work in the continuous case and
        # degrade to a plain threshold in the discrete one.
        level = np.clip(vec[BUTTON_SLICE], 0.0, 1.0)
        want = np.where(self._held, level > lim.release_threshold, level >= lim.press_threshold)

        # A held button must stay held for min_hold_ticks; anything shorter is a
        # press no input stack would reliably deliver.
        too_soon = self._held & ~want & (self._hold_ticks < lim.min_hold_ticks)
        if too_soon.any():
            self.stats["debounced"] += int(too_soon.sum())
            want = want | too_soon

        if lim.resolve_exclusive:
            for a, b in EXCLUSIVE_PAIRS:
                ia, ib = BUTTON_NAMES.index(a), BUTTON_NAMES.index(b)
                if want[ia] and want[ib]:
                    self.stats["exclusive_resolved"] += 1
                    # Keep the more strongly asserted direction; on an exact tie
                    # release both rather than inventing an arbitrary winner.
                    if level[ia] > level[ib]:
                        want[ib] = False
                    elif level[ib] > level[ia]:
                        want[ia] = False
                    else:
                        want[ia] = want[ib] = False

        for i in self._blocked_idx:
            if want[i]:
                self.stats["buttons_blocked"] += 1
                want[i] = False

        self._hold_ticks = np.where(want, self._hold_ticks + 1, 0).astype(np.int32)
        self._held = want
        return want

    # -- convenience ------------------------------------------------------
    def as_gamepad(self, action: Sequence[float]) -> GamepadAction:
        """Sanitize and return the ergonomic :class:`GamepadAction` view."""
        return GamepadAction.from_vector(self(action))

    def sanitize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Apply the guard across a whole ``(T, ACTION_DIM)`` chunk, in order.

        Order matters: the temporal guarantees only hold if the chunk is walked
        front to back, and doing it here (rather than per emitted tick) lets you
        inspect what the agent *will* do before any of it reaches the game.
        """
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"expected a (T, {ACTION_DIM}) chunk, got shape {arr.shape}")
        return np.stack([self(row) for row in arr]) if arr.shape[0] else arr.copy()
