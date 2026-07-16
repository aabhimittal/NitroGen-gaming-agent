"""Tiny numpy software renderer for the toy games.

No game engine, no OpenGL — just enough primitives (filled circles, rectangles,
gradients) to paint distinct-looking 256x256 RGB frames. Keeping rendering in
pure numpy means the whole repo runs anywhere torch runs, with no display.
"""

from __future__ import annotations

import numpy as np


class Canvas:
    """A mutable H x W x 3 uint8 RGB buffer with a handful of draw primitives.

    Coordinates use the *normalized* game convention: ``(x, y)`` in ``[0, 1]``
    with the origin at the top-left, x rightwards, y downwards.
    """

    def __init__(self, size: int = 256, background=(18, 18, 28)):
        self.size = size
        self.buf = np.zeros((size, size, 3), dtype=np.uint8)
        self.buf[:] = np.array(background, dtype=np.uint8)
        # Precompute a pixel coordinate grid for vectorized primitives.
        ys, xs = np.mgrid[0:size, 0:size]
        self._xs = xs.astype(np.float32)
        self._ys = ys.astype(np.float32)

    def fill_gradient(self, top, bottom):
        """Vertical gradient background — cheap way to make games look different."""
        top = np.array(top, dtype=np.float32)
        bottom = np.array(bottom, dtype=np.float32)
        t = np.linspace(0, 1, self.size, dtype=np.float32)[:, None, None]
        grad = (top[None, None] * (1 - t) + bottom[None, None] * t).astype(np.uint8)
        self.buf[:] = grad

    def circle(self, cx: float, cy: float, radius: float, color):
        """Filled circle; center and radius are fractions of the canvas size."""
        cx_px, cy_px, r_px = cx * self.size, cy * self.size, radius * self.size
        mask = (self._xs - cx_px) ** 2 + (self._ys - cy_px) ** 2 <= r_px ** 2
        self.buf[mask] = np.array(color, dtype=np.uint8)

    def rect(self, x0: float, y0: float, x1: float, y1: float, color):
        """Filled axis-aligned rectangle in [0,1] coordinates."""
        s = self.size
        xa, xb = int(min(x0, x1) * s), int(max(x0, x1) * s)
        ya, yb = int(min(y0, y1) * s), int(max(y0, y1) * s)
        xa, xb = np.clip([xa, xb], 0, s)
        ya, yb = np.clip([ya, yb], 0, s)
        self.buf[ya:yb, xa:xb] = np.array(color, dtype=np.uint8)

    def ring(self, cx: float, cy: float, radius: float, thickness: float, color):
        """Hollow circle (annulus) — handy for a fixed reticle at screen center."""
        cx_px, cy_px = cx * self.size, cy * self.size
        r_out, r_in = radius * self.size, max(0.0, (radius - thickness) * self.size)
        d2 = (self._xs - cx_px) ** 2 + (self._ys - cy_px) ** 2
        mask = (d2 <= r_out ** 2) & (d2 >= r_in ** 2)
        self.buf[mask] = np.array(color, dtype=np.uint8)

    def crosshair(self, cx: float, cy: float, radius: float, color):
        """A small reticle: a ring plus a center dot. Marks the (fixed) avatar."""
        self.ring(cx, cy, radius, radius * 0.35, color)
        self.circle(cx, cy, radius * 0.22, color)

    def to_array(self) -> np.ndarray:
        return self.buf.copy()
