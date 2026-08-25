"""Reading actions back out of pixels — the controller-overlay pipeline.

Behavior cloning needs ``(frame, action)`` pairs, and the actions are the hard
half: raw gameplay video doesn't say which buttons were pressed. NitroGen's
insight is that a large slice of footage (speedruns, tutorials, "gamepad
viewer" streams) already renders a **live controller overlay**, so the label is
literally on screen. The paper localizes that widget with SIFT/XFeat matching
against ~300 controller templates, then reads it with a fine-tuned SegFormer,
reporting joystick R² = 0.84 and button accuracy = 0.96.

This module implements both directions of that idea at a scale we can actually
ship and verify:

    * :func:`render_overlay` — the *forward* direction. Paint a controller HUD
      (two stick wells, two trigger bars, a grid of button lights) onto a frame,
      exactly what a gamepad viewer does live while a human plays.
    * :class:`OverlayExtractor` — the *inverse* direction. Localize the panel,
      read the stick dots back into ``[-1, 1]``, measure the trigger fills, and
      classify each button light.

Localization uses two magenta **fiducial markers** at the panel corners — our
stand-in for the paper's keypoint/template matching. It is the same contract
(find the widget from its visual signature, then rectify a known layout), just
with a signature we control instead of 300 third-party skins. Round-tripping
random actions through render -> extract reproduces the paper's metric framing;
our numbers are near-perfect because the overlay is clean, and the point is the
*mechanism*, not the score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from nitrogen.action_space import BUTTON_NAMES, GamepadAction

# ---------------------------------------------------------------------------
# Palette. MARKER must be a colour no game renders — pure magenta is the
# classic fiducial choice for exactly that reason.
# ---------------------------------------------------------------------------
MARKER = (255, 0, 255)
PANEL_BG = (13, 17, 23)
WELL_BG = (38, 44, 56)
DOT = (255, 208, 64)
TRIG_TRACK = (40, 44, 54)
TRIG_FILL = (90, 170, 255)
BTN_ON = (86, 226, 138)
BTN_OFF = (46, 52, 64)

MARKER_PX = 3           # fiducial square side, in pixels
BTN_ROWS, BTN_COLS = 2, 6


@dataclass(frozen=True)
class PanelLayout:
    """Pixel geometry of the HUD, derived from the panel bounding box.

    Both the renderer and the extractor build this from the same numbers, so the
    inverse direction never has to guess where anything is — it only has to find
    the box.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @classmethod
    def for_frame(cls, height: int, width: int) -> "PanelLayout":
        return cls(
            x0=int(0.02 * width), y0=int(0.72 * height),
            x1=int(0.98 * width), y1=int(0.97 * height),
        )

    # -- sub-regions -----------------------------------------------------
    @property
    def well_radius(self) -> float:
        return 0.34 * self.h

    def well_center(self, right: bool) -> Tuple[float, float]:
        fx = 0.90 if right else 0.10
        return self.x0 + fx * self.w, self.y0 + 0.5 * self.h

    @property
    def dot_radius(self) -> float:
        return max(1.0, 0.26 * self.well_radius)

    def trigger_box(self, right: bool) -> Tuple[int, int, int, int]:
        fx0, fx1 = (0.775, 0.815) if right else (0.185, 0.225)
        return (
            int(self.x0 + fx0 * self.w), int(self.y0 + 0.15 * self.h),
            int(self.x0 + fx1 * self.w), int(self.y0 + 0.85 * self.h),
        )

    def button_box(self, index: int) -> Tuple[int, int, int, int]:
        """Inner square of button ``index`` (row-major over a 2x6 grid)."""
        row, col = divmod(index, BTN_COLS)
        gx0, gx1, gy0, gy1 = 0.26, 0.74, 0.12, 0.88
        cw = (gx1 - gx0) * self.w / BTN_COLS
        ch = (gy1 - gy0) * self.h / BTN_ROWS
        cx = self.x0 + gx0 * self.w + (col + 0.5) * cw
        cy = self.y0 + gy0 * self.h + (row + 0.5) * ch
        side = max(2.0, 0.55 * min(cw, ch))
        return (
            int(cx - side / 2), int(cy - side / 2),
            int(cx + side / 2), int(cy + side / 2),
        )


# ---------------------------------------------------------------------------
# Forward direction: paint the HUD
# ---------------------------------------------------------------------------

def _fill(buf: np.ndarray, x0: int, y0: int, x1: int, y1: int, color) -> None:
    h, w = buf.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 > x0 and y1 > y0:
        buf[y0:y1, x0:x1] = np.asarray(color, dtype=np.uint8)


def _disc(buf: np.ndarray, cx: float, cy: float, r: float, color) -> None:
    h, w = buf.shape[:2]
    x0, x1 = max(0, int(cx - r) - 1), min(w, int(cx + r) + 2)
    y0, y1 = max(0, int(cy - r) - 1), min(h, int(cy + r) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    buf[y0:y1, x0:x1][mask] = np.asarray(color, dtype=np.uint8)


def render_overlay(frame: np.ndarray, action: GamepadAction) -> np.ndarray:
    """Paint a gamepad-viewer HUD for ``action`` onto a copy of ``frame``.

    ``frame`` is ``(H, W, 3)`` uint8. Returns a new array — the caller's frame is
    never mutated, so this is safe to use inside a rollout loop.
    """
    buf = np.array(frame, dtype=np.uint8, copy=True)
    if buf.ndim != 3 or buf.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) uint8 frame, got shape {frame.shape}")
    h, w = buf.shape[:2]
    lay = PanelLayout.for_frame(h, w)

    _fill(buf, lay.x0, lay.y0, lay.x1, lay.y1, PANEL_BG)
    # Fiducials at two opposite corners pin down the box exactly.
    _fill(buf, lay.x0, lay.y0, lay.x0 + MARKER_PX, lay.y0 + MARKER_PX, MARKER)
    _fill(buf, lay.x1 - MARKER_PX, lay.y1 - MARKER_PX, lay.x1, lay.y1, MARKER)

    sticks = ((action.left_x, action.left_y, False), (action.right_x, action.right_y, True))
    for sx, sy, right in sticks:
        cx, cy = lay.well_center(right)
        _disc(buf, cx, cy, lay.well_radius, WELL_BG)
        reach = lay.well_radius - lay.dot_radius
        sx = float(np.clip(sx, -1.0, 1.0))
        sy = float(np.clip(sy, -1.0, 1.0))
        _disc(buf, cx + sx * reach, cy + sy * reach, lay.dot_radius, DOT)

    for value, right in ((action.left_trigger, False), (action.right_trigger, True)):
        tx0, ty0, tx1, ty1 = lay.trigger_box(right)
        _fill(buf, tx0, ty0, tx1, ty1, TRIG_TRACK)
        value = float(np.clip(value, 0.0, 1.0))
        filled = int(round(value * (ty1 - ty0)))
        if filled > 0:  # trigger bars fill from the bottom, like every gamepad viewer
            _fill(buf, tx0, ty1 - filled, tx1, ty1, TRIG_FILL)

    for i, name in enumerate(BUTTON_NAMES):
        bx0, by0, bx1, by1 = lay.button_box(i)
        _fill(buf, bx0, by0, bx1, by1, BTN_ON if action.button(name) else BTN_OFF)
    return buf


# ---------------------------------------------------------------------------
# Inverse direction: read the HUD back
# ---------------------------------------------------------------------------

class OverlayExtractor:
    """Recover a :class:`GamepadAction` from a frame that carries the HUD.

    ``extract`` returns ``None`` when the panel can't be localized — a *frame
    filter*, not a crash. Real footage constantly loses the overlay (cutscenes,
    menus, scene transitions, the streamer moving the widget), and the paper's
    pipeline drops those frames rather than labeling them with garbage.
    """

    def __init__(self, marker_tolerance: int = 24, min_marker_pixels: int = 4):
        self.marker_tolerance = marker_tolerance
        self.min_marker_pixels = min_marker_pixels

    # -- localization ----------------------------------------------------
    def locate(self, frame: np.ndarray) -> Optional[PanelLayout]:
        """Find the panel bounding box from its fiducial markers."""
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        rgb = arr[..., :3].astype(np.int16)
        target = np.asarray(MARKER, dtype=np.int16)
        mask = np.all(np.abs(rgb - target) <= self.marker_tolerance, axis=-1)
        if int(mask.sum()) < self.min_marker_pixels:
            return None
        ys, xs = np.nonzero(mask)
        lay = PanelLayout(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if lay.w < 8 * MARKER_PX or lay.h < 4 * MARKER_PX:
            return None  # too small to hold a readable layout
        return lay

    # -- readers ---------------------------------------------------------
    @staticmethod
    def _read_stick(rgb: np.ndarray, lay: PanelLayout, right: bool) -> Tuple[float, float]:
        cx, cy = lay.well_center(right)
        r = lay.well_radius
        x0, x1 = max(0, int(cx - r)), min(rgb.shape[1], int(cx + r) + 1)
        y0, y1 = max(0, int(cy - r)), min(rgb.shape[0], int(cy + r) + 1)
        patch = rgb[y0:y1, x0:x1]
        dist = np.linalg.norm(patch - np.asarray(DOT, dtype=np.float32), axis=-1)
        mask = dist < 90.0
        if not mask.any():
            return 0.0, 0.0
        ys, xs = np.nonzero(mask)
        dot_x = xs.mean() + x0
        dot_y = ys.mean() + y0
        reach = max(1e-6, r - lay.dot_radius)
        return (
            float(np.clip((dot_x - cx) / reach, -1.0, 1.0)),
            float(np.clip((dot_y - cy) / reach, -1.0, 1.0)),
        )

    @staticmethod
    def _read_trigger(rgb: np.ndarray, lay: PanelLayout, right: bool) -> float:
        tx0, ty0, tx1, ty1 = lay.trigger_box(right)
        patch = rgb[ty0:ty1, tx0:tx1]
        if patch.size == 0:
            return 0.0
        dist = np.linalg.norm(patch - np.asarray(TRIG_FILL, dtype=np.float32), axis=-1)
        filled_rows = int((dist < 80.0).any(axis=1).sum())
        return float(np.clip(filled_rows / max(1, ty1 - ty0), 0.0, 1.0))

    @staticmethod
    def _read_button(rgb: np.ndarray, lay: PanelLayout, index: int) -> bool:
        bx0, by0, bx1, by1 = lay.button_box(index)
        patch = rgb[by0:by1, bx0:bx1]
        if patch.size == 0:
            return False
        on = np.linalg.norm(patch.mean(axis=(0, 1)) - np.asarray(BTN_ON, dtype=np.float32))
        off = np.linalg.norm(patch.mean(axis=(0, 1)) - np.asarray(BTN_OFF, dtype=np.float32))
        return bool(on < off)

    def extract(self, frame: np.ndarray) -> Optional[GamepadAction]:
        """Read the HUD in ``frame``; ``None`` if no panel is visible."""
        lay = self.locate(frame)
        if lay is None:
            return None
        rgb = np.asarray(frame)[..., :3].astype(np.float32)
        lx, ly = self._read_stick(rgb, lay, right=False)
        rx, ry = self._read_stick(rgb, lay, right=True)
        return GamepadAction(
            left_x=lx, left_y=ly, right_x=rx, right_y=ry,
            left_trigger=self._read_trigger(rgb, lay, right=False),
            right_trigger=self._read_trigger(rgb, lay, right=True),
            buttons={
                name: self._read_button(rgb, lay, i)
                for i, name in enumerate(BUTTON_NAMES)
            },
        )


# ---------------------------------------------------------------------------
# Quality metric — the round-trip that mirrors the paper's reported numbers
# ---------------------------------------------------------------------------

def evaluate_extraction(
    actions: Sequence[GamepadAction],
    size: int = 256,
    base_frame: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Render each action as an overlay, read it back, and score the round trip.

    Returns ``joystick_r2`` (pooled over all four stick channels),
    ``button_accuracy``, ``trigger_mae``, ``localization_rate`` (the fraction of
    frames whose panel was found at all), and ``n``.
    """
    if len(actions) == 0:
        raise ValueError("evaluate_extraction needs at least one action")
    if base_frame is None:
        base_frame = np.full((size, size, 3), 24, dtype=np.uint8)

    extractor = OverlayExtractor()
    true_sticks, pred_sticks = [], []
    true_trig, pred_trig = [], []
    button_hits = button_total = 0
    located = 0

    for action in actions:
        rendered = render_overlay(base_frame, action)
        got = extractor.extract(rendered)
        if got is None:
            continue
        located += 1
        true_sticks += [action.left_x, action.left_y, action.right_x, action.right_y]
        pred_sticks += [got.left_x, got.left_y, got.right_x, got.right_y]
        true_trig += [action.left_trigger, action.right_trigger]
        pred_trig += [got.left_trigger, got.right_trigger]
        for name in BUTTON_NAMES:
            button_hits += int(got.button(name) == action.button(name))
            button_total += 1

    if located == 0:
        return {
            "joystick_r2": 0.0, "button_accuracy": 0.0, "trigger_mae": 1.0,
            "localization_rate": 0.0, "n": 0,
        }

    y = np.asarray(true_sticks, dtype=np.float64)
    p = np.asarray(pred_sticks, dtype=np.float64)
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    # A constant ground truth has no variance to explain; call a perfect
    # reconstruction R2 = 1 rather than dividing by zero.
    r2 = 1.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot

    return {
        "joystick_r2": float(r2),
        "button_accuracy": float(button_hits / max(1, button_total)),
        "trigger_mae": float(np.abs(np.asarray(true_trig) - np.asarray(pred_trig)).mean()),
        "localization_rate": float(located / len(actions)),
        "n": int(located),
    }
